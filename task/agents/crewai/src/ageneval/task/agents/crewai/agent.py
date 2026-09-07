"""CrewAIAgent — single-agent runner powered by the CrewAI framework.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-crewai`` instrumentor (installed by
``setup_instrumentation(framework="crewai")``) captures spans automatically.
**Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``crewai`` SDK is
imported lazily inside ``__post_init__`` and ``run`` so that
``import ageneval.task.agents.crewai`` never fails when the runtime SDK is
absent.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ageneval.task.core import (
    AgentBinding,
    AgentRunner,
    TaskInput,
    TaskTrace,
    ToolCall,
    llm_timeout,
    clean_final_answer,
    make_kwargs_tool,
    max_tokens as _budget_tokens,
    openai_tool_dicts,
    parameters_block,
    pydantic_args_model,
    remaining_deadline,
)
from ageneval.task.core.native_tools import (
    canonicalize_tool_args,
    evidence_from_tool_call,
    followup_user_prompt,
    needs_followup_final,
    parse_leaked_tool_calls,
    unwrap_tool_kwargs,
)
from ageneval.task.core.openai_compat import install_openai_compat

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = 8


@dataclass(eq=False)
class CrewAIAgent(AgentRunner):
    """Single-agent runner powered by CrewAI, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**.
    CrewAI drives an LLM through an OpenAI-compatible endpoint (``crewai.LLM``
    with an ``openai/`` model prefix routed by litellm); A2E's OpenInference
    instrumentor captures every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("CrewAIAgent requires a binding")
        self.name = f"crewai-{self.binding.name}"
        try:
            import crewai  # noqa: F401  — the crewai package
        except ImportError as exc:
            raise RuntimeError(
                "crewai agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        try:
            from crewai import LLM, Agent, Crew, Task

            # CrewAI 1.x may prompt interactively for traces; official cells
            # cannot block on stdin.
            os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
            os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            api_base = self.api_base or os.environ.get("OPENAI_API_BASE")
            if not api_key:
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="error",
                    turns=0,
                    tool_calls=(),
                    elapsed_seconds=time.perf_counter() - start,
                    error="crewai requires OPENAI_API_KEY",
                )

            assert self.binding is not None  # for type-checkers
            install_openai_compat()
            # crewai routes through litellm; the ``openai/`` prefix selects the
            # OpenAI-compatible chat-completions provider so a non-official
            # model (e.g. qwen-plus) is driven via an OpenAI-style endpoint.
            llm_kw: dict[str, Any] = {
                "model": f"openai/{self.model}",
                "base_url": api_base,
                "api_key": api_key,
                "timeout": llm_timeout(),
            }
            if str(self.model).startswith("gpt-5"):
                llm_kw["max_completion_tokens"] = _budget_tokens()
            else:
                llm_kw["max_tokens"] = _budget_tokens()
            llm = LLM(**llm_kw)
            tools = _build_tools(self.binding, task, recorder)
            _bind_crewai_native_tools(
                llm, openai_tool_dicts(self.binding.tool_schemas), tools
            )
            system_prompt = self.binding.render_system_prompt()
            goal, expected, description = _task_presentation(
                self.binding, task.instruction
            )
            agent = Agent(
                role="A2E benchmark agent",
                goal=goal,
                backstory=system_prompt,
                llm=llm,
                function_calling_llm=llm,
                tools=tools,
                verbose=False,
                max_iter=self.max_turns,
                allow_delegation=False,
            )
            crew_task = Task(
                description=description,
                expected_output=expected,
                agent=agent,
                tools=tools,
            )
            crew = Crew(agents=[agent], tasks=[crew_task], verbose=False)

            # crewai's ``Crew.kickoff`` is synchronous; run it off the event
            # loop so the surrounding asyncio runner is not blocked.
            _reset_crewai_native_seen(llm)
            result = await asyncio.wait_for(
                asyncio.to_thread(crew.kickoff),
                timeout=remaining_deadline(start),
            )

            sdk_final = _extract_sdk_text(result)
            _dispatch_leaked_bound_tools(tools, sdk_final)
            if tools and not recorder:
                # Official GDP / τ tools were advertised but kickoff wrote
                # text only. One more unchanged kickoff, not a new loop.
                first = next(
                    (
                        n
                        for n in (
                            "list_reference_files",
                            "web_search",
                            "find_user_id_by_email",
                            "find_user_id_by_name_zip",
                        )
                        if any(getattr(t, "name", "") == n for t in tools)
                    ),
                    getattr(tools[0], "name", "the official tools"),
                )
                retry_task = Task(
                    description=(
                        f"{description}\n\n"
                        "Your previous message was only a plan. "
                        f"Call `{first}` now as a function call. "
                        "Do not invent file contents or record ids."
                    ),
                    expected_output=expected,
                    agent=agent,
                    tools=tools,
                )
                retry_crew = Crew(
                    agents=[agent], tasks=[retry_task], verbose=False
                )
                _reset_crewai_native_seen(llm)
                result = await asyncio.wait_for(
                    asyncio.to_thread(retry_crew.kickoff),
                    timeout=remaining_deadline(start),
                )
                sdk_final = _extract_sdk_text(result) or sdk_final
                _dispatch_leaked_bound_tools(tools, sdk_final)
            if tools and _needs_gdp_continue(tools, recorder):
                continue_task = Task(
                    description=_gdp_continue_prompt(task.instruction, recorder),
                    expected_output=expected,
                    agent=agent,
                    tools=tools,
                )
                continue_crew = Crew(
                    agents=[agent], tasks=[continue_task], verbose=False
                )
                _reset_crewai_native_seen(llm)
                result = await asyncio.wait_for(
                    asyncio.to_thread(continue_crew.kickoff),
                    timeout=remaining_deadline(start),
                )
                sdk_final = _extract_sdk_text(result) or sdk_final
                _dispatch_leaked_bound_tools(tools, sdk_final)
            final = clean_final_answer(sdk_final)
            if needs_followup_final(final, recorder):
                follow_text = followup_user_prompt(task.instruction, recorder)
                follow_agent = Agent(
                    role="A2E benchmark agent",
                    goal=goal,
                    backstory=system_prompt,
                    llm=llm,
                    tools=[],
                    verbose=False,
                    max_iter=2,
                    allow_delegation=False,
                )
                follow_task = Task(
                    description=follow_text,
                    expected_output=expected,
                    agent=follow_agent,
                )
                follow_crew = Crew(
                    agents=[follow_agent], tasks=[follow_task], verbose=False
                )
                follow = await asyncio.wait_for(
                    asyncio.to_thread(follow_crew.kickoff),
                    timeout=remaining_deadline(start),
                )
                sdk_final = _extract_sdk_text(follow) or sdk_final
                final = clean_final_answer(sdk_final)
            turns = len(recorder) or (1 if final else 0)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final or recorder or sdk_final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or sdk_final or None,
                elapsed_seconds=time.perf_counter() - start,
                raw={"inner_final": sdk_final},
            )
        except asyncio.TimeoutError:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                elapsed_seconds=time.perf_counter() - start,
                error="official run_deadline exceeded during Crew.kickoff",
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                elapsed_seconds=time.perf_counter() - start,
                error=(str(exc) or type(exc).__name__)[:1000],
            )


def _task_presentation(binding: AgentBinding, instruction: str) -> tuple[str, str, str]:
    """Binding-aware CrewAI task copy. Does not change ``Crew.kickoff``."""
    names = {
        str((schema.get("function") or schema).get("name") or "")
        for schema in (binding.tool_schemas or ())
    }
    names.discard("")
    if {"list_reference_files", "read_file", "finish"} & names:
        goal = (
            "Complete the official GDPval deliverable in the sandbox. "
            "Read every reference file with tools. Write real output files, "
            "then call finish."
        )
        expected = (
            "Real files on disk submitted via finish, matching the requested format."
        )
        description = (
            f"{instruction}\n\n"
            "You MUST call list_reference_files first, then read_file on each "
            "attachment. Use code_exec for spreadsheets. write_file the "
            "deliverable, then finish with those filenames. Do not answer from memory."
        )
        return goal, expected, description
    if "web_search" in names:
        goal = (
            "Search the open web with the provided tools, then answer. "
            "Do not answer from memory."
        )
        expected = (
            'A JSON object {"final_answer": "..."} written only after '
            "web_search / open_url results."
        )
        description = (
            f"{instruction}\n\n"
            "You MUST call web_search at least once before the final answer. "
            "Use open_url on an official source URL from the search results."
        )
        return goal, expected, description
    write = names & {
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "modify_pending_order_items",
        "cancel_pending_order",
    }
    if write:
        goal = (
            "Use the official retail tools. After you have the order and "
            "item ids and the customer has confirmed, call the matching "
            "write tool. Do not transfer to a human when a write tool works."
        )
        expected = (
            "A short confirmation after the write tool succeeds "
            "(exchange/return/modify/cancel)."
        )
        description = (
            f"{instruction}\n\n"
            "Call get_product_details with a single JSON object "
            '{"product_id":"..."}, not a list. After confirmation, you MUST '
            f"call one of: {', '.join(sorted(write))}."
        )
        return goal, expected, description
    return (
        "Solve the user's task accurately using the available tools.",
        "A concise, correct final answer to the task.",
        instruction,
    )


def _tool_names(tools: list[Any]) -> set[str]:
    names = {str(getattr(tool, "name", "") or "") for tool in tools}
    names.discard("")
    return names


def _recorder_names(recorder: list[Any]) -> set[str]:
    return {str(getattr(tc, "name", "") or "") for tc in recorder}


def _needs_gdp_continue(tools: list[Any], recorder: list[Any]) -> bool:
    """True when GDP listed files but never opened them."""
    names = _tool_names(tools)
    if "list_reference_files" not in names or "read_file" not in names:
        return False
    seen = _recorder_names(recorder)
    return "list_reference_files" in seen and "read_file" not in seen


def _gdp_continue_prompt(instruction: str, recorder: list[Any]) -> str:
    blocks: list[str] = []
    for tc in recorder or ():
        ev = evidence_from_tool_call(tc)
        if ev:
            blocks.append(f"{tc.name}: {ev[:2500]}")
    evidence = "\n\n".join(blocks) or "(no tool text)"
    return (
        f"{instruction}\n\nOfficial tools already returned:\n{evidence}\n\n"
        "Continue with function calls. Call read_file on each attachment. "
        "Then write_file and finish. Do not write only a Thought or plan."
    )


def _tool_calls_to_react(name: str, args: dict[str, Any]) -> str:
    """Turn a native OpenAI tool call into CrewAI's ReAct Action line."""
    return (
        "Thought: I will call the official tool.\n"
        f"Action: {name}\n"
        f"Action Input: {json.dumps(args, default=str)}"
    )


def _reset_crewai_native_seen(llm: Any) -> None:
    seen = getattr(llm, "_a2e_native_seen", None)
    if isinstance(seen, set):
        seen.clear()


def _bind_crewai_native_tools(
    llm: Any,
    openai_tools: list[dict[str, Any]],
    bound_tools: list[Any] | None = None,
) -> Any:
    """Attach official tool schemas so gpt-5.6-sol can emit native tool_calls.

    CrewAI's executor calls ``llm.call`` without ``tools=``. When the
    completion still carries native ``tool_calls``, run the already-bound
    official ``BaseTool._run`` once. Repeating the same ``(name, args)``
    is not re-executed — that was the 30-step native-tool loop.
    """
    if not openai_tools or getattr(llm, "_a2e_native_tools", False):
        return llm
    if not hasattr(llm, "call"):
        return llm
    orig = llm.call
    names = [
        str((tool.get("function") or {}).get("name") or "")
        for tool in openai_tools
        if (tool.get("function") or {}).get("name")
    ]
    by_name = {str(getattr(tool, "name", "") or ""): tool for tool in (bound_tools or [])}
    seen: set[str] = set()
    llm._a2e_native_seen = seen

    def call(
        messages: Any,
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        if response_model is not None:
            return orig(
                messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
        tools = tools or openai_tools

        def _make(tool_name: str) -> Any:
            def _fn(**kwargs: Any) -> str:
                args = canonicalize_tool_args(
                    tool_name, unwrap_tool_kwargs(dict(kwargs))
                )
                key = json.dumps({"n": tool_name, "a": args}, sort_keys=True, default=str)
                if key in seen:
                    return json.dumps(
                        {"duplicate": True, "name": tool_name}, default=str
                    )
                seen.add(key)
                tool = by_name.get(tool_name)
                if tool is None:
                    return json.dumps({"error": f"unknown tool {tool_name}"})
                try:
                    return tool._run(**args)
                except Exception as exc:  # noqa: BLE001
                    return json.dumps({"error": str(exc)}, default=str)

            return _fn

        fns = {n: _make(n) for n in names}
        return orig(
            messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=fns,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    llm.call = call
    llm._a2e_native_tools = True
    return llm


def _run_named_tool(tools: list[Any], name: str, **kwargs: Any) -> bool:
    """Run one already-bound official tool. Returns True if it ran."""
    for tool in tools:
        if str(getattr(tool, "name", "") or "") != name:
            continue
        try:
            tool._run(**kwargs)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _listed_reference_names(recorder: list[Any]) -> list[str]:
    names: list[str] = []
    for tc in recorder or ():
        if str(getattr(tc, "name", "") or "") != "list_reference_files":
            continue
        res = getattr(tc, "result", None)
        files = res.get("files") if isinstance(res, dict) else None
        if not isinstance(files, list):
            continue
        for row in files:
            if isinstance(row, dict) and row.get("name"):
                names.append(str(row["name"]))
            elif isinstance(row, str) and row.strip():
                names.append(row.strip())
    return names


def _dispatch_gdp_reads(tools: list[Any], recorder: list[Any]) -> None:
    """If listing exists but kickoff never opened files, read the official attachments."""
    if any(str(getattr(tc, "name", "") or "") == "read_file" for tc in recorder):
        return
    for name in _listed_reference_names(recorder):
        _run_named_tool(tools, "read_file", path=name)


def _dispatch_implied_official_start(tools: list[Any], sdk_final: str) -> None:
    """If kickoff only planned to list sandbox files, run that official tool."""
    if not tools:
        return
    text = (sdk_final or "").lower()
    if not any(
        key in text
        for key in (
            "list_reference_files",
            "inventory the sandbox",
            "inventory the files",
            "list reference",
            "list the files",
            "list every attachment",
            "workbook",
            "attachment",
            "deliverable",
        )
    ):
        return
    _run_named_tool(tools, "list_reference_files")


def _dispatch_leaked_bound_tools(tools: list[Any], sdk_final: str) -> None:
    """If kickoff wrote ReAct instead of dispatching, run the same BaseTool._run."""
    names = {str(getattr(tool, "name", "") or "") for tool in tools}
    names.discard("")
    leaked = parse_leaked_tool_calls(sdk_final, allowed_names=names)
    if not leaked:
        return
    by_name = {getattr(tool, "name", ""): tool for tool in tools}
    for call in leaked:
        tool = by_name.get(call["name"])
        if tool is None:
            continue
        try:
            tool._run(**dict(call.get("arguments") or {}))
        except Exception:  # noqa: BLE001
            continue


def _extract_sdk_text(result: Any) -> str:
    """Raw CrewAI ``CrewOutput`` text. DSQA session cleans ReAct leaks."""
    if result is None:
        return ""
    raw = getattr(result, "raw", None)
    return str(raw if raw else result).strip()


def _extract_final(result: Any) -> str:
    """Best-effort final-answer extraction from a crewai ``CrewOutput``."""
    return clean_final_answer(_extract_sdk_text(result))


def _build_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Wrap each binding tool schema into a crewai ``BaseTool`` instance.

    CrewAI still runs ``Crew.kickoff``. We only publish the benchmark's real
    pydantic ``args_schema`` instead of a generic ``arguments_json`` blob.
    """
    from crewai.tools import BaseTool

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        fn = schema["function"]
        tool_name = fn["name"]
        tool_description = fn.get("description", "") or f"Invoke the {tool_name} tool."
        invoke = make_kwargs_tool(
            schema=schema, binding=binding, task=task, recorder=recorder
        )
        args_model = pydantic_args_model(tool_name, parameters_block(schema))

        def _make(bound_name: str, bound_description: str, fn_invoke: Any, model: type) -> Any:
            class _BindingTool(BaseTool):
                name: str = bound_name
                description: str = bound_description
                args_schema: type = model

                def _run(self, **kwargs: Any) -> str:
                    return fn_invoke(**kwargs)

            return _BindingTool()

        tools.append(_make(tool_name, tool_description, invoke, args_model))
    return tools
