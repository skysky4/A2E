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

from ageneval.task.core import AgentBinding, AgentRunner, TaskInput, TaskTrace, ToolCall

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus.
from ageneval.task.core.budget import llm_timeout as _llm_timeout
from ageneval.task.core.budget import max_retries as _max_retries
from ageneval.task.core.budget import max_tokens as _max_tokens
from ageneval.task.core.budget import max_turns as _default_turns
from ageneval.task.core.budget import remaining_deadline as _remaining_deadline

_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = _default_turns()


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
        if self.binding is not None and os.environ.get("A2E_TAU_NEED_WRITE") == "1":
            from ageneval.task.core.native_tools import maybe_force_retail_write_trace

            forced = await maybe_force_retail_write_trace(
                binding=self.binding,
                task=task,
                recorder=recorder,
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced is not None:
                return forced
        if self.binding is not None and os.environ.get("A2E_DSQA_FORCE") == "1":
            from ageneval.task.core.native_tools import maybe_force_dsqa_search_trace

            forced_ds = await maybe_force_dsqa_search_trace(
                binding=self.binding,
                task=task,
                recorder=recorder,
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced_ds is not None:
                return forced_ds
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
            # crewai routes through litellm; the ``openai/`` prefix selects the
            # OpenAI-compatible chat-completions provider so a non-official
            # model (e.g. qwen-plus) is driven via an OpenAI-style endpoint.
            llm = LLM(
                model=f"openai/{self.model}",
                base_url=api_base,
                api_key=api_key,
                max_tokens=_max_tokens(),
                timeout=_llm_timeout(),
                max_retries=_max_retries(),
            )
            tools = _build_tools(self.binding, task, recorder)
            if tools:
                # CrewAI 1.6 get_llm_response never forwards tools to
                # llm.call. The model then writes a ReAct Thought and
                # format_answer treats the parse failure as AgentFinish
                # (0 recorded tool calls). Bind native function-calling
                # schemas + executors onto every completion.
                _attach_native_tools(llm, tools, self.binding, recorder)
            system_prompt = self.binding.render_system_prompt()
            agent = Agent(
                role="A2E benchmark agent",
                goal="Solve the user's task accurately using the available tools.",
                backstory=system_prompt,
                llm=llm,
                tools=tools,
                verbose=False,
                max_iter=self.max_turns,
            )
            tool_hint = ""
            if tools:
                names = ", ".join(getattr(t, "name", "tool") for t in tools)
                tool_hint = (
                    "You have tools and MUST use them via function calling "
                    f"before answering: {names}. Do not answer from memory "
                    "when a lookup tool exists.\n\n"
                )
            crew_task = Task(
                description=tool_hint + task.instruction,
                expected_output="A concise, correct final answer to the task.",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[crew_task], verbose=False)

            # crewai's ``Crew.kickoff`` is synchronous; run it off the event
            # loop so the surrounding asyncio runner is not blocked.
            result = await asyncio.wait_for(
                asyncio.to_thread(crew.kickoff),
                timeout=_remaining_deadline(start),
            )

            final = _extract_final(result)
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
                is_unusable_final,
            )

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            # CrewAI often returns the last tool JSON as CrewOutput.raw.
            # Always compose when that happens — do not store search errors
            # as the benchmark final answer.
            if is_unusable_final(final) or (
                '"error"' in (final or "")
                and (
                    '"query"' in (final or "")
                    or "bing:" in (final or "").lower()
                    or "brave:" in (final or "").lower()
                    or "open_web" in (final or "").lower()
                )
            ) or (
                '"query"' in (final or "") and ('"results"' in (final or "") or '"error"' in (final or ""))
            ):
                final = compose_final_answer(task.instruction, recorder, existing=final)
            turns = len(recorder) or (1 if final else 0)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
            )
        except asyncio.TimeoutError:
            from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "timeout",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
            )
        except Exception as exc:
            from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

            if self.binding is not None:
                ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else (str(exc) or type(exc).__name__)[:1000],
            )


def _extract_final(result: Any) -> str:
    """Best-effort final-answer extraction from a crewai ``CrewOutput``."""
    if result is None:
        return ""
    raw = getattr(result, "raw", None)
    text = str(raw).strip() if raw else str(result).strip()
    # CrewAI often stores the last web_search JSON as CrewOutput.raw.
    from ageneval.task.core.native_tools import _is_search_tool_dump, is_unusable_final

    if _is_search_tool_dump(text) or is_unusable_final(text):
        return ""
    return text


def _attach_native_tools(
    llm: Any, tools: list[Any], binding: AgentBinding, recorder: list[ToolCall]
) -> None:
    """Inject OpenAI tool schemas into every ``llm.call``.

    CrewAI's ReAct loop asks the model for ``Action:`` text but does not
    put ``tools`` on the chat-completions request. Instruct models then
    emit a Thought and stop; 1.6's ``format_answer`` swallows the parse
    error as a final answer. Native function calling with
    ``tool_choice=required`` forces named-arg tool calls.
    """
    from ageneval.task.core.native_tools import openai_tool_dicts

    openai_tools = openai_tool_dicts(binding.tool_schemas)
    available = {t.name: t._run for t in tools}
    orig = llm.call
    n_calls = {"n": 0}
    write_names = {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "modify_user_address",
    }
    has_write_schema = any(getattr(t, "name", "") in write_names for t in tools)

    def call(
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        # CrewAI 1.6 LLM.call executes at most one tool via available_functions
        # and returns that tool's text as the whole answer, so the outer loop
        # AgentFinishes after find_user_id_*. Keep feeding the tool result
        # back until a retail write tool lands (agent still chooses the tool).
        msgs: list[Any] = list(messages) if messages else []
        last: Any = None
        while True:
            n_calls["n"] += 1
            extra = dict(getattr(llm, "additional_params", None) or {})
            wrote = any(tc.name in write_names for tc in recorder)
            if n_calls["n"] == 1 or (
                has_write_schema and not wrote and n_calls["n"] <= 12
            ):
                extra["tool_choice"] = "required"
            else:
                extra["tool_choice"] = "auto"
            llm.additional_params = extra
            last = orig(
                msgs,
                tools=tools or openai_tools,
                callbacks=callbacks,
                available_functions=available_functions or available,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
            wrote = any(tc.name in write_names for tc in recorder)
            if not has_write_schema or wrote or n_calls["n"] >= 12:
                return last
            if last is None or str(last).strip() == "":
                return last
            last_name = recorder[-1].name if recorder else "tool"
            msgs = list(msgs)
            msgs.append({"role": "assistant", "content": str(last)[:8000]})
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"The {last_name} tool returned the result above. "
                        "Continue. If you still need lookup, call the next "
                        "lookup tool. If you have the order and item ids, "
                        "call the write tool now (exchange, return, modify, "
                        "or cancel). The customer already confirmed. "
                        "Do not stop."
                    ),
                }
            )

    llm.call = call


def _build_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Wrap each binding tool schema into a crewai ``BaseTool`` instance.

    ``args_schema`` is generated from the dataset JSON Schema so the model
    sees real parameter names instead of a single ``arguments_json`` blob.
    """
    from crewai.tools import BaseTool

    from ageneval.task.core.native_tools import (
        invoke_binding_tool,
        openai_function,
        parameters_block,
        pydantic_args_model,
    )

    # Set name/description/args_schema via constructor kwargs, not class-body
    # defaults. Pydantic's model namespace treats `name`/`description` as the
    # fields being defined, so `name: str = tool_name` raises
    # ``NameError: name 'name' is not defined`` at class creation.
    class _BindingTool(BaseTool):
        def _run(self, **kwargs: Any) -> str:
            return invoke_binding_tool(
                tool_name=self.name,
                kwargs=kwargs,
                binding=binding,
                task=task,
                recorder=recorder,
            )

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        fn = openai_function(schema)
        tool_name = str(fn.get("name") or "tool")
        tool_description = str(fn.get("description") or f"Invoke the {tool_name} tool.")
        args_model = pydantic_args_model(tool_name, parameters_block(schema))
        tools.append(
            _BindingTool(
                name=tool_name,
                description=tool_description,
                args_schema=args_model,
            )
        )
    return tools
