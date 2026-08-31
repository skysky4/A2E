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
    parameters_block,
    pydantic_args_model,
)
from ageneval.task.core.native_tools import parse_leaked_tool_calls
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
            system_prompt = self.binding.render_system_prompt()
            goal, expected, description = _task_presentation(
                self.binding, task.instruction
            )
            tool_names = {
                str((schema.get("function") or schema).get("name") or "")
                for schema in (self.binding.tool_schemas or ())
            }
            use_react = "web_search" in tool_names
            agent = Agent(
                role="A2E benchmark agent",
                goal=goal,
                backstory=system_prompt,
                llm=llm,
                function_calling_llm=llm,
                tools=tools,
                verbose=False,
                max_iter=self.max_turns,
            )
            crew_task = Task(
                description=description,
                expected_output=expected,
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[crew_task], verbose=False)

            # crewai's ``Crew.kickoff`` is synchronous; run it off the event
            # loop so the surrounding asyncio runner is not blocked.
            result = await asyncio.to_thread(crew.kickoff)

            sdk_final = _extract_sdk_text(result)
            if use_react:
                _dispatch_leaked_bound_tools(tools, sdk_final)
            final = clean_final_answer(sdk_final)
            if use_react and recorder and not final:
                follow_text = _followup_from_tools(task.instruction, recorder)
                follow_agent = Agent(
                    role="A2E benchmark agent",
                    goal=goal,
                    backstory=system_prompt,
                    llm=llm,
                    tools=[],
                    verbose=False,
                    max_iter=2,
                )
                follow_task = Task(
                    description=follow_text,
                    expected_output=expected,
                    agent=follow_agent,
                )
                follow_crew = Crew(
                    agents=[follow_agent], tasks=[follow_task], verbose=False
                )
                follow = await asyncio.to_thread(follow_crew.kickoff)
                sdk_final = _extract_sdk_text(follow) or sdk_final
                final = clean_final_answer(sdk_final)
            turns = len(recorder) or (1 if final else 0)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final or recorder else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or sdk_final or None,
                elapsed_seconds=time.perf_counter() - start,
                raw={"inner_final": sdk_final},
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
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


def _dispatch_leaked_bound_tools(tools: list[Any], sdk_final: str) -> None:
    """If kickoff wrote ReAct instead of dispatching, run the same BaseTool._run."""
    leaked = parse_leaked_tool_calls(sdk_final)
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


def _followup_from_tools(instruction: str, recorder: list[ToolCall]) -> str:
    from ageneval.task.core.native_tools import evidence_from_tool_call

    blocks = []
    for tc in recorder:
        ev = evidence_from_tool_call(tc)
        if ev:
            blocks.append(f"{tc.name}: {ev[:2500]}")
    evidence = "\n\n".join(blocks) or "(no tool text)"
    return (
        f"{instruction}\n\nTool results:\n{evidence}\n\n"
        'Write only {"final_answer":"..."} from those results. Do not call tools.'
    )


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
