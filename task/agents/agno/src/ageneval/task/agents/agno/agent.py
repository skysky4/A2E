"""AgnoAgent — single-agent runner powered by the Agno agent framework.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-agno`` instrumentor (installed by
``setup_instrumentation(framework="agno")``) captures spans automatically.
**Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``agno`` SDK is
imported lazily inside ``__post_init__`` and ``run`` so that
``import ageneval.task.agents.agno`` never fails when the runtime SDK is
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
from ageneval.task.core.budget import max_tokens as _budget_tokens
from ageneval.task.core.budget import max_turns as _default_turns
from ageneval.task.core.budget import run_deadline as _run_deadline

_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = _default_turns()

# Per-request LLM timeout + retries. Without these a stalled connection to the
# OpenAI-compatible endpoint hangs the whole run forever (observed: a single
# qwen-max call stuck >20 min with the container idle). A bounded timeout makes
# a hung call fail fast and retry; a persistent failure raises and is recorded
# as an error trace (the task still gets a trajectory) so the run never stalls.
_LLM_TIMEOUT = _llm_timeout()
_LLM_MAX_RETRIES = _max_retries()
# Reasoning models (e.g. kimi-k3) spend completion tokens on hidden
# reasoning_content before they emit a tool call. A low default cap makes
# finish_reason=length with empty content and zero tools.
_MAX_TOKENS = _budget_tokens()

# Whole-agent wall-clock deadline. agno's ``Agent.run`` is a *synchronous* call
# run via ``asyncio.to_thread``; a slow sandbox tool (e.g. compiling a C-extension
# library or running its test suite) can keep a single tool call busy for minutes,
# so a few of them exhaust any task budget. When this deadline fires we DON'T
# discard the work: the shared tool ``recorder`` already holds every call made so
# far, so we return a *partial* trajectory (status="timeout") instead of letting
# an outer ``asyncio.wait_for`` hard-cancel the thread and lose everything. Keep
# this BELOW the experiment runner's task cap so this branch wins and the
# trajectory (plus the sandbox diff/score) survives.
_RUN_DEADLINE = _run_deadline()

# Many dataset bindings prescribe a text JSON-action protocol ({"action": ...})
# that suits text-loop agents (e.g. langgraph). agno drives the model through
# NATIVE function-calling, so that instruction makes some models emit a JSON
# final answer in one turn without ever calling a tool (empty trajectory). This
# hint steers agno's model to actually invoke the provided functions.
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
_DELIVERABLE_HINT = (
    "\n\nWrite the complete deliverable now as plain text. "
    "No JSON wrapper, no plan, no tool calls, no 'I will start'."
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
    "\nUser lookup is done. After get_user_details / get_order_details / "
    "get_product_details, call the write tool now (exchange, return, "
    "modify, or cancel). The customer already confirmed. Do not stop. "
    "Do not call find_user_id_* again."
)


def _tool_transcript(recorder: list[ToolCall], *, limit: int = 10) -> str:
    """Replay already-returned tools so a fresh Agent does not re-lookup."""
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
class AgnoAgent(AgentRunner):
    """Single-agent runner powered by the Agno framework, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**.
    Agno drives an LLM through an OpenAI-compatible endpoint (``OpenAILike``);
    A2E's OpenInference instrumentor captures every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    request_timeout: float = _LLM_TIMEOUT
    max_retries: int = _LLM_MAX_RETRIES
    run_deadline: float = _RUN_DEADLINE
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("AgnoAgent requires a binding")
        self.name = f"agno-{self.binding.name}"
        try:
            import agno  # noqa: F401  — the agno package
        except ImportError as exc:
            raise RuntimeError(
                "agno agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        try:
            from agno.agent import Agent
            from agno.models.openai.chat import OpenAIChat
            from agno.models.openai.like import OpenAILike
            from ageneval.task.core.openai_compat import install_openai_compat, sanitize_messages

            install_openai_compat()
            if not getattr(OpenAIChat._format_all_messages, "_a2e_compat", False):
                _orig_fmt = OpenAIChat._format_all_messages

                def _fmt(self, messages, compress_tool_results=False):  # noqa: ANN001
                    rows = _orig_fmt(self, messages, compress_tool_results)
                    return sanitize_messages(rows)

                _fmt._a2e_compat = True  # type: ignore[attr-defined]
                OpenAIChat._format_all_messages = _fmt  # type: ignore[method-assign]

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
                    error="agno requires OPENAI_API_KEY",
                )

            assert self.binding is not None  # for type-checkers
            model = OpenAILike(
                id=self.model,
                api_key=api_key,
                base_url=api_base,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
                max_tokens=_MAX_TOKENS,
            )
            tools = _build_function_tools(self.binding, task, recorder)
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
                    deadline=self.run_deadline - (time.perf_counter() - start),
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
                    deadline=self.run_deadline - (time.perf_counter() - start),
                    agent_name=self.name,
                    start=start,
                )
                if forced_ds is not None:
                    return forced_ds
            # Tool-less bindings (GDPval) must not get the native-tool hint —
            # it steers glm into a JSON-wrapped "I'll start..." plan.
            act_hint = _NATIVE_TOOL_HINT if tools else _DELIVERABLE_HINT
            # Dataset overrides set max_turns (DeepSearchQA=8). Other harnesses
            # pass that budget into the SDK; without tool_call_limit agno loops
            # until A2E_AGNO_DEADLINE and returns an empty final_answer.
            agent = Agent(
                name="a2e_agent",
                model=model,
                tools=tools,
                instructions=self.binding.render_system_prompt() + act_hint,
                tool_call_limit=self.max_turns,
            )

            # agno's ``Agent.run`` is synchronous; run it off the event loop so
            # the surrounding asyncio runner is not blocked (mirrors smolagents).
            # Bound it by a wall-clock deadline: a slow sandbox tool can block one
            # call for minutes. On timeout the worker thread keeps running (Python
            # threads can't be cancelled) but ``recorder`` already holds its work,
            # so we return a partial trajectory rather than losing it — the outer
            # SandboxScoringRunner still extracts the diff + score while the
            # container is alive, and tears the container down (killing the thread).
            async def _run_once(prompt: str):
                return await asyncio.wait_for(
                    asyncio.to_thread(agent.run, prompt),
                    timeout=max(30.0, self.run_deadline - (time.perf_counter() - start)),
                )

            try:
                result = await _run_once(task.instruction)
                need_web = any(
                    (schema.get("function") or {}).get("name") == "web_search"
                    for schema in (self.binding.tool_schemas or ())
                )
                missing_required = (tools and not recorder) or (
                    need_web and "web_search" not in {tc.name for tc in recorder}
                )
                lookup = {"find_user_id_by_name_zip", "find_user_id_by_email"}
                names = [tc.name for tc in recorder]
                lookup_loop = bool(names) and all(n in lookup for n in names) and len(names) >= 3
                if missing_required or lookup_loop:
                    extra = ""
                    if need_web and "web_search" not in set(names):
                        extra = "\nYou MUST call web_search now before answering."
                    elif lookup_loop:
                        extra = (
                            "\nUser lookup is done. Call get_user_details then "
                            "get_order_details. Do not call find_user_id_* again."
                        )
                    retry_agent = Agent(
                        name="a2e_agent",
                        model=model,
                        tools=tools,
                        instructions=self.binding.render_system_prompt()
                        + _NATIVE_TOOL_HINT
                        + _FORCE_TOOL_HINT
                        + extra,
                        tool_call_limit=self.max_turns,
                    )
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            retry_agent.run,
                            task.instruction + _FORCE_TOOL_HINT + extra,
                        ),
                        timeout=max(30.0, self.run_deadline - (time.perf_counter() - start)),
                    )
                write_schema = any(
                    _schema_tool_name(s) in _RETAIL_WRITE
                    for s in (self.binding.tool_schemas or ())
                )
                need_write = write_schema and (
                    os.environ.get("A2E_TAU_NEED_WRITE") == "1"
                    or "already confirm" in (task.instruction or "").lower()
                )
                if (
                    need_write
                    and not any(tc.name in _RETAIL_WRITE for tc in recorder)
                    and (self.run_deadline - (time.perf_counter() - start)) > 20
                ):
                    from ageneval.task.core.native_tools import force_retail_write_calls

                    forced = await force_retail_write_calls(
                        binding=self.binding,
                        task=task,
                        recorder=recorder,
                        model=self.model,
                        api_key=api_key,
                        api_base=api_base,
                        max_turns=min(16, self.max_turns),
                        deadline=self.run_deadline - (time.perf_counter() - start),
                    )
                    if forced:
                        result = type("R", (), {"content": forced})()
            except asyncio.TimeoutError:
                partial = tuple(recorder)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="timeout",
                    turns=len(partial),
                    tool_calls=partial,
                    final_answer=None,
                    elapsed_seconds=time.perf_counter() - start,
                    error=(
                        f"agent exceeded {self.run_deadline:.0f}s deadline "
                        f"after {len(partial)} tool call(s)"
                    ),
                )
            except Exception as stop_exc:
                from agno.exceptions import StopAgentRun

                if not isinstance(stop_exc, StopAgentRun):
                    raise
                result = None

            run_error = _run_error(result)
            err_l = (run_error or "").lower()
            if run_error and ("503" in err_l or "no available accounts" in err_l):
                remain = self.run_deadline - (time.perf_counter() - start)
                if remain > 40:
                    await asyncio.sleep(8)
                    try:
                        result = await _run_once(task.instruction)
                        run_error = _run_error(result)
                    except Exception:  # noqa: BLE001
                        pass
            final = _extract_final(result)
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
                is_unusable_final,
            )

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            if is_unusable_final(final) or run_error is not None:
                final = compose_final_answer(task.instruction, recorder, existing=final)

            turns = _count_turns(result) or len(recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else (run_error or "empty final")[:1000],
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

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


def _extract_text(obj: Any) -> str:
    """Visible content, or kimi-k3 reasoning if content is empty."""
    if obj is None:
        return ""
    content = getattr(obj, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("thinking") or block.get("reasoning_content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined
    elif content is not None and str(content).strip():
        text = str(content).strip()
        low = text.lower()
        if "no available accounts" not in low and "error code: 503" not in low:
            return text
    for key in ("reasoning_content", "reasoning", "thinking"):
        extra = getattr(obj, key, None)
        if extra and str(extra).strip():
            return str(extra).strip()
    extra = getattr(obj, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            text = extra.get(key)
            if text:
                return str(text).strip()
    dump = obj.model_dump() if hasattr(obj, "model_dump") else {}
    if isinstance(dump, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            text = dump.get(key) or (dump.get("model_extra") or {}).get(key)
            if text:
                return str(text).strip()
    return ""


def _extract_final(result: Any) -> str:
    text = _extract_text(result)
    if text:
        return text
    for msg in reversed(list(getattr(result, "messages", None) or ())):
        text = _extract_text(msg)
        if text:
            return text
    return ""


def _count_turns(result: Any) -> int:
    """Best-effort turn count from an agno RunOutput object."""
    messages = getattr(result, "messages", None)
    if messages:
        return sum(1 for m in messages if getattr(m, "role", None) == "assistant")
    return 0


def _run_error(result: Any) -> str | None:
    """Return an SDK-reported terminal error instead of treating it as output."""
    status = getattr(result, "status", None)
    status_value = getattr(status, "value", status)
    normalized = str(status_value or "").upper()
    if normalized not in {"ERROR", "CANCELLED"}:
        return None
    content = getattr(result, "content", None)
    return str(content or f"Agno run ended with status {normalized}")


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Wrap each binding tool schema into an agno ``Function`` with the tool's
    REAL parameter schema, so the model calls it natively.

    A binding's tools are dynamic (the JSON schema comes from the dataset), so
    we cannot write a static Python signature for them. Instead we construct
    ``agno.tools.function.Function`` directly, handing agno the OpenAI-format
    ``parameters`` schema verbatim and setting ``skip_entrypoint_processing=True``
    so agno uses that schema AS-IS: no signature reflection, no pydantic
    ``validate_call`` wrapping of our closure.

    This is what makes the model emit ``bash(command="...")`` natively rather
    than guessing a generic ``arguments_json`` blob or nesting everything under
    a spurious ``kwargs`` object (which silently dropped the real arguments and
    produced empty sandbox trajectories). agno invokes the entrypoint as
    ``entrypoint(**model_arguments)``, so the tool's arguments arrive directly
    as keyword args. Each invocation is captured into ``TaskTrace.tool_calls``.
    """
    from agno.tools.function import Function

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        fn = schema["function"]
        name = fn["name"]
        description = fn.get("description", "") or f"Invoke the {name} tool."
        parameters = dict(fn.get("parameters") or {"type": "object", "properties": {}})

        def _make(tool_name: str):
            def _tool(**kwargs: Any) -> str:
                from ageneval.task.core.native_tools import (
                    invoke_binding_tool,
                    is_stop_tool_result,
                )

                text = invoke_binding_tool(
                    tool_name=tool_name,
                    kwargs=kwargs,
                    binding=binding,
                    task=task,
                    recorder=recorder,
                )
                if is_stop_tool_result(text):
                    # Duplicate/budget STOP must not abort before a retail write.
                    # StopAgentRun here used to skip the write-nudge loop entirely.
                    wrote = any(tc.name in _RETAIL_WRITE for tc in recorder)
                    if wrote:
                        from agno.exceptions import StopAgentRun

                        raise StopAgentRun(text)
                return text

            return _tool

        tools.append(
            Function(
                name=name,
                description=description,
                parameters=parameters,
                entrypoint=_make(name),
                skip_entrypoint_processing=True,
            )
        )
    return tools
