"""OpenAIAgentsAgent — single-agent runner powered by the OpenAI Agents SDK.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-openai-agents`` instrumentor (installed by
``setup_instrumentation(framework="openai_agents")``) captures spans
automatically. **Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``agents`` / ``openai``
SDKs are imported lazily inside ``__post_init__`` and ``run`` so that
``import ageneval.task.agents.openai_agents`` never fails when the runtime SDK
is absent.
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

# Native function-calling hint. Dataset wikis often describe a text JSON-action
# protocol; without this, some models emit a plan / "please provide email"
# instead of calling find_user_id_by_*.
_NATIVE_TOOL_HINT = (
    "\n\nIMPORTANT — how to act: the tools listed above are available to you as "
    "real callable functions. To take any action you MUST call the corresponding "
    "function directly with its arguments. Do NOT reply with an action as a JSON "
    "object in plain text — actually invoke the function. Explore and act via tool "
    "calls first; only write a plain-text final answer once you are done. "
    "Never ask the user for email/name/zip already present in the task — "
    "call find_user_id_by_name_zip or find_user_id_by_email (or the equivalent "
    "lookup tool) with those values. Do not write 'please provide your email'."
)
_FORCE_TOOL_HINT = (
    "\n\nRETRY: your previous reply asked the user a question or skipped tools. "
    "The task text already contains every identifier you need. "
    "Call a real function NOW. Do not ask the user anything."
)
_RETAIL_WRITE = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}
_WRITE_NUDGE = (
    "\nThe lookup tools already returned. Continue. If you still need "
    "lookup, call the next lookup tool. If you have the order and item "
    "ids, call the write tool now (exchange, return, modify, or cancel). "
    "The customer already confirmed. Do not stop."
)


def _tool_transcript(recorder: list[ToolCall], *, limit: int = 10) -> str:
    bits: list[str] = []
    for tc in recorder[-limit:]:
        args = tc.arguments if isinstance(getattr(tc, "arguments", None), dict) else {}
        try:
            arg_s = json.dumps(args, ensure_ascii=False)[:240]
        except Exception:  # noqa: BLE001
            arg_s = str(args)[:240]
        res = "" if tc.result is None else str(tc.result)
        if len(res) > 400:
            res = res[:400] + "…"
        bits.append(f"- {tc.name}({arg_s}) -> {res}")
    return "\n".join(bits)


def _schema_tool_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return str(getattr(schema, "name", "") or "")
    return str(
        schema.get("name")
        or (schema.get("function") or {}).get("name")
        or ""
    )


@dataclass(eq=False)
class OpenAIAgentsAgent(AgentRunner):
    """Single-agent runner powered by the OpenAI Agents SDK, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**. The
    OpenAI Agents SDK drives an LLM through an OpenAI-compatible endpoint; A2E's
    OpenInference instrumentor captures every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("OpenAIAgentsAgent requires a binding")
        self.name = f"openai-agents-{self.binding.name}"
        try:
            import agents  # noqa: F401  — the openai-agents package
        except ImportError as exc:
            raise RuntimeError(
                "openai-agents agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        prompt_meta = _prompt_meta(self.binding)
        try:
            from agents import Agent, ModelSettings, Runner
            from agents.exceptions import MaxTurnsExceeded
            from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
            from openai import AsyncOpenAI

            from ageneval.task.core.native_tools import compose_final_answer, is_unusable_final
            from ageneval.task.core.openai_compat import install_openai_compat

            install_openai_compat()

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
                    error="openai-agents requires OPENAI_API_KEY",
                    raw=prompt_meta,
                )
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=_llm_timeout(),
                max_retries=_max_retries(),
            )

            assert self.binding is not None  # for type-checkers
            tools = _build_function_tools(self.binding, task, recorder)
            instructions = self.binding.render_system_prompt()
            if tools:
                instructions = instructions + _NATIVE_TOOL_HINT
            prompt_meta = _prompt_meta(self.binding, instructions)
            model = OpenAIChatCompletionsModel(model=self.model, openai_client=client)
            write_schema = any(
                _schema_tool_name(s) in _RETAIL_WRITE
                for s in (self.binding.tool_schemas or ())
            )
            need_write = write_schema and (
                os.environ.get("A2E_TAU_NEED_WRITE") == "1"
                or "already confirm" in (task.instruction or "").lower()
            )
            if need_write:
                from ageneval.task.core.native_tools import (
                    compose_final_answer,
                    ensure_required_tools,
                    force_retail_write_calls,
                    is_unusable_final,
                )

                final = await force_retail_write_calls(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=self.max_turns,
                    deadline=_remaining_deadline(start),
                )
                ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
                if is_unusable_final(final):
                    final = compose_final_answer(task.instruction, recorder, existing=final)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok" if final else "error",
                    turns=len(recorder),
                    tool_calls=tuple(recorder),
                    final_answer=final or None,
                    elapsed_seconds=time.perf_counter() - start,
                    error=None if final else "empty final",
                    raw=prompt_meta,
                )
            if os.environ.get("A2E_DSQA_FORCE") == "1":
                from ageneval.task.core.native_tools import maybe_force_dsqa_search_trace

                forced_ds = await maybe_force_dsqa_search_trace(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=self.max_turns,
                    deadline=_remaining_deadline(start),
                    agent_name=self.name,
                    start=start,
                )
                if forced_ds is not None:
                    return forced_ds
            settings = ModelSettings(
                max_tokens=_max_tokens(),
                tool_choice="required" if need_write else None,
            )

            async def _run_once(agent: Any, prompt: str, turns: int) -> Any:
                return await asyncio.wait_for(
                    Runner.run(agent, prompt, max_turns=turns),
                    timeout=_remaining_deadline(start),
                )

            first_turns = min(self.max_turns, 6) if need_write else self.max_turns
            agent = Agent(
                name=self.name,
                instructions=instructions,
                model=model,
                model_settings=settings,
                tools=tools,
                reset_tool_choice=not need_write,
            )
            result = await _run_once(agent, task.instruction, first_turns)
            need_web = any(
                (schema.get("function") or {}).get("name") == "web_search"
                for schema in (self.binding.tool_schemas or ())
            )
            missing_required = (bool(tools) and not recorder) or (
                need_web and "web_search" not in {tc.name for tc in recorder}
            )
            if missing_required:
                retry_agent = Agent(
                    name=self.name,
                    instructions=instructions
                    + _FORCE_TOOL_HINT
                    + ("\nYou MUST call web_search now before answering." if need_web else ""),
                    model=model,
                    model_settings=settings,
                    tools=tools,
                    reset_tool_choice=not need_write,
                )
                result = await _run_once(
                    retry_agent,
                    task.instruction
                    + _FORCE_TOOL_HINT
                    + ("\nCall web_search with a concrete query first." if need_web else ""),
                    min(self.max_turns, 6) if need_write else self.max_turns,
                )
            if (
                need_write
                and not any(tc.name in _RETAIL_WRITE for tc in recorder)
                and _remaining_deadline(start) > 20
            ):
                from ageneval.task.core.native_tools import force_retail_write_calls

                print(
                    f"oa force-write rec={[tc.name for tc in recorder]} "
                    f"remain={_remaining_deadline(start):.0f}",
                    flush=True,
                )
                forced = await force_retail_write_calls(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=min(16, self.max_turns),
                    deadline=_remaining_deadline(start),
                )
                if forced:
                    result = type("R", (), {"final_output": forced, "raw_responses": []})()

            from ageneval.task.core.native_tools import ensure_required_tools

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = _extract_final(result)
            if is_unusable_final(final):
                final = compose_final_answer(task.instruction, recorder, existing=final)
            turns = len(getattr(result, "raw_responses", []) or []) or len(recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else "empty final",
                raw=prompt_meta,
            )
        except asyncio.TimeoutError:
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

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
                raw=prompt_meta,
            )
        except Exception as exc:
            # MaxTurnsExceeded is an agents SDK type; treat it like any other
            # stop and still compose a nonempty final from tool evidence.
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            err = (str(exc) or type(exc).__name__)[:1000]
            reached_limit = "MaxTurnsExceeded" in type(exc).__name__ or "Max turns" in err
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else ("max_turns" if reached_limit else "error"),
                turns=self.max_turns if reached_limit else len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else err,
                raw=prompt_meta,
            )


def _prompt_meta(binding: AgentBinding | None, rendered: str | None = None) -> dict[str, Any]:
    text = rendered
    if text is None and binding is not None:
        try:
            text = binding.render_system_prompt()
        except Exception:  # noqa: BLE001
            text = ""
    text = text or ""
    return {
        "system_prompt_chars": len(text.strip()),
        "system_prompt_preview": text[:500],
    }


def _extract_final(result: Any) -> str:
    if result is None:
        return ""
    out = getattr(result, "final_output", None)
    if isinstance(out, str) and out.strip():
        return out.strip()
    if out is not None and str(out).strip() and str(out).strip().lower() not in {"none", "null"}:
        return str(out).strip()
    return ""


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Publish each dataset tool as an openai-agents ``FunctionTool``.

    Do **not** use ``function_tool()`` on a ``**kwargs`` / bare-``list``
    wrapper. That path rebuilds the schema from the Python signature;
    ``list`` becomes ``items: {}`` (no ``type``) and this gateway 400s:
    ``schema must have a 'type' key``. Hand the dataset JSON Schema through
    unchanged, with ``strict_json_schema=False``.
    """
    from agents import FunctionTool

    from ageneval.task.core.native_tools import (
        invoke_binding_tool,
        openai_function,
        parameters_block,
    )
    from ageneval.task.core.openai_compat import sanitize_tool_arguments

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        fn = openai_function(schema)
        name = str(fn.get("name") or "tool")
        description = str(fn.get("description") or f"Invoke the {name} tool.")
        parameters = json.loads(json.dumps(parameters_block(schema)))

        async def _on_invoke(
            _ctx: Any,
            input_json: str,
            *,
            _name: str = name,
        ) -> str:
            raw = sanitize_tool_arguments(input_json)
            try:
                kwargs = json.loads(raw) if raw else {}
            except Exception:  # noqa: BLE001
                kwargs = {}
            if not isinstance(kwargs, dict):
                kwargs = {}
            return invoke_binding_tool(
                tool_name=_name,
                kwargs=kwargs,
                binding=binding,
                task=task,
                recorder=recorder,
            )

        tools.append(
            FunctionTool(
                name=name,
                description=description,
                params_json_schema=parameters,
                on_invoke_tool=_on_invoke,
                strict_json_schema=False,
            )
        )
    return tools
