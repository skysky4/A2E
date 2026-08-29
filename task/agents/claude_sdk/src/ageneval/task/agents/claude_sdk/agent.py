"""ClaudeSDKAgent / ClaudeSDKTauAgent — single-agent runner (Anthropic SDK).

Dataset-agnostic ``ClaudeSDKAgent`` takes an ``AgentBinding`` and drives any
benchmark whose binding is provided. It uses the official Anthropic Python
SDK (``anthropic`` package) over the Messages API with **native tool use** —
no ``claude`` CLI subprocess. Point it at any Anthropic-compatible endpoint
(including OpenAI-style gateways that also expose ``/v1/messages``) via the
``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` environment variables.

``ClaudeSDKTauAgent`` is a thin backwards-compat wrapper that builds the
τ-bench binding for the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ageneval.task.core import AgentBinding, AgentRunner, TaskInput, TaskTrace, ToolCall

logger = logging.getLogger(__name__)

from ageneval.task.core.budget import llm_timeout as _llm_timeout
from ageneval.task.core.budget import max_retries as _max_retries
from ageneval.task.core.budget import max_tokens as _budget_tokens
from ageneval.task.core.budget import max_turns as _default_turns
from ageneval.task.core.budget import remaining_deadline as _remaining_deadline

_MAX_TURNS = _default_turns()
_MAX_TOKENS = _budget_tokens()
_RETAIL_WRITE = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}
# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus. The endpoint gateway maps the model name.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"

# Matches a (possibly nested one level) JSON object embedded in free text —
# used for the JSON-action protocol some dataset bindings prescribe in their
# system prompt (e.g. τ-bench: ``{"action": ..., "arguments": ...}``).
_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _parse_json(text: str) -> dict[str, Any]:
    """Best-effort: extract the first JSON object from ``text``."""
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


_NATIVE_TOOL_HINT = (
    "\n\nIMPORTANT — the tools listed above are real callable functions. "
    "To take any action you MUST call the corresponding function. "
    "Do NOT reply with an action as a JSON object in plain text. "
    "Never ask the user for email/name/zip already present in the task — "
    "call find_user_id_by_name_zip or find_user_id_by_email first."
)


def _to_anthropic_tools(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style function specs into Anthropic tool schema.

    OpenAI: ``{"type":"function","function":{"name","description","parameters"}}``
    Anthropic: ``{"name","description","input_schema"}``.
    """
    from ageneval.task.core.native_tools import parameters_block

    tools: list[dict[str, Any]] = []
    for schema in schemas:
        fn = schema.get("function", schema)
        name = str(fn.get("name", "tool"))
        tools.append(
            {
                "name": name,
                "description": str(fn.get("description", "") or name),
                "input_schema": parameters_block(schema),
            }
        )
    return tools


def _to_openai_tools(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from ageneval.task.core.native_tools import openai_function, parameters_block

    tools: list[dict[str, Any]] = []
    for schema in schemas:
        fn = openai_function(schema)
        name = str(fn.get("name") or "tool")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(fn.get("description") or name),
                    "parameters": parameters_block(schema),
                },
            }
        )
    return tools


def _blocks_to_dicts(content: Any) -> list[dict[str, Any]]:
    """Serialise a response's content blocks to plain dicts for the next turn."""
    out: list[dict[str, Any]] = []
    for block in content or []:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump())
        elif isinstance(block, dict):
            out.append(block)
    return out


def _text_of(content: Any) -> str:
    """Join visible and thinking text from an assistant message.

    kimi-k3 on the Anthropic Messages gateway often returns only
    ``thinking`` / ``reasoning_content`` blocks and an empty ``text``
    block. Treating that as no answer made gdpval-aa look like a
    harness failure.
    """
    parts: list[str] = []
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
            text = (
                block.get("text")
                or block.get("thinking")
                or block.get("reasoning_content")
                or ""
            )
        else:
            text = (
                getattr(block, "text", None)
                or getattr(block, "thinking", None)
                or getattr(block, "reasoning_content", None)
                or ""
            )
        if btype in {"text", "thinking", "reasoning"} or text:
            if text:
                parts.append(str(text))
    return "\n".join(p for p in parts if p).strip()


@dataclass
class ClaudeSDKAgent(AgentRunner):
    """Single-agent runner powered by the Anthropic Python SDK.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**.
    Talks to the Anthropic Messages API directly (no subprocess); set
    ``ANTHROPIC_BASE_URL`` to route through an Anthropic-compatible gateway.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None

    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("ClaudeSDKAgent requires a binding")
        self.name = f"claude-sdk-{self.binding.name}"

    async def run(self, task: TaskInput) -> TaskTrace:
        # Lazy import: the SDK is an optional runtime dependency.
        from anthropic import AsyncAnthropic  # type: ignore

        start = time.perf_counter()
        api_key = (
            self.api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        anthropic_base = (self.api_base or os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        openai_base = (os.environ.get("OPENAI_API_BASE") or "").strip()
        # This project's gateway is OpenAI /v1/chat/completions only. The
        # Messages API returns 403 unless the host is a real Anthropic endpoint.
        use_messages = bool(anthropic_base) and "anthropic.com" in anthropic_base.lower()
        if not use_messages:
            return await self._run_openai_compat(
                task, start=start, api_key=api_key, base_url=openai_base
            )
        base_url = anthropic_base or openai_base
        # Anthropic SDK posts to ``{base_url}/v1/messages``. OpenAI-compatible
        # gateways usually set OPENAI_API_BASE to ``.../v1``; strip that suffix
        # so we do not request ``/v1/v1/messages``.
        if base_url:
            stripped = base_url.rstrip("/")
            if stripped.endswith("/v1"):
                base_url = stripped[: -len("/v1")] or stripped

        if not api_key:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error=(
                    "ClaudeSDKAgent requires ANTHROPIC_API_KEY or OPENAI_API_KEY. "
                    "Set ANTHROPIC_BASE_URL (or OPENAI_API_BASE) for a gateway."
                ),
            )

        assert self.binding is not None  # for type-checkers
        timeout = _llm_timeout()
        try:
            client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url or None,
                timeout=timeout,
                max_retries=_max_retries(),
            )
        except TypeError:
            client = AsyncAnthropic(
                api_key=api_key, base_url=base_url or None, timeout=timeout
            )
        system_prompt = self.binding.render_system_prompt() + _NATIVE_TOOL_HINT
        tools = _to_anthropic_tools(self.binding.tool_schemas)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.instruction}]
        tool_calls: list[ToolCall] = []
        final_answer: str | None = None
        turns = 0

        for turn in range(self.max_turns):
            turns = turn + 1
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self.model,
                        max_tokens=_MAX_TOKENS,
                        system=system_prompt,
                        tools=tools,
                        messages=messages,
                    ),
                    timeout=min(timeout, _remaining_deadline(start)),
                )
            except asyncio.TimeoutError:
                from ageneval.task.core.native_tools import compose_final_answer

                final = compose_final_answer(task.instruction, tool_calls)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok" if final else "timeout",
                    turns=turns,
                    tool_calls=tuple(tool_calls),
                    final_answer=final or None,
                    elapsed_seconds=time.perf_counter() - start,
                    error=None if final else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc) or type(exc).__name__
                lower = msg.lower()
                hint = ""
                if "authentication" in lower or "401" in lower or "403" in lower:
                    hint = " — Anthropic auth failed. Check ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL."
                elif "connection" in lower or "timeout" in lower:
                    hint = " — network error reaching the Anthropic-compatible endpoint."
                from ageneval.task.core.native_tools import compose_final_answer

                final = compose_final_answer(task.instruction, tool_calls)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok" if final else "error",
                    turns=turns,
                    tool_calls=tuple(tool_calls),
                    final_answer=final or None,
                    elapsed_seconds=time.perf_counter() - start,
                    error=None if final else (msg + hint)[:1000],
                )

            # Echo the assistant turn back verbatim so the model keeps context.
            messages.append({"role": "assistant", "content": _blocks_to_dicts(response.content)})

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                # No native tool_use block. Fall back to the JSON-action
                # protocol some bindings prescribe in their system prompt
                # (e.g. τ-bench: {"action": ..., "arguments": ...} per turn).
                text = _text_of(response.content)
                parsed = _parse_json(text)
                if "action" in parsed:
                    name = str(parsed["action"])
                    from ageneval.task.core.native_tools import execute_recorded_tool, openai_function

                    available = [
                        str(openai_function(s).get("name") or "")
                        for s in (self.binding.tool_schemas or [])
                        if openai_function(s).get("name")
                    ]
                    text = execute_recorded_tool(
                        tool_name=name,
                        kwargs=parsed.get("arguments") or {},
                        executor=self.binding.tool_executor,
                        initial_state=task.initial_state,
                        recorder=tool_calls,
                        available=available,
                    )
                    last = tool_calls[-1]
                    result = last.result if last.error is None else (last.result or {"error": last.error})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"TOOL[{name}] result: {text}",
                        }
                    )
                    from ageneval.task.core.native_tools import is_stop_tool_result

                    if is_stop_tool_result(text):
                        break
                    continue
                if "final_answer" in parsed:
                    final_answer = str(parsed["final_answer"])
                    break
                final_answer = text or None
                break

            # Native tool_use: execute every requested tool, feed results back.
            tool_results: list[dict[str, Any]] = []
            for block in tool_uses:
                from ageneval.task.core.native_tools import execute_recorded_tool, openai_function

                available = [
                    str(openai_function(s).get("name") or "")
                    for s in (self.binding.tool_schemas or [])
                    if openai_function(s).get("name")
                ]
                text = execute_recorded_tool(
                    tool_name=block.name,
                    kwargs=getattr(block, "input", {}) or {},
                    executor=self.binding.tool_executor,
                    initial_state=task.initial_state,
                    recorder=tool_calls,
                    available=available,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            from ageneval.task.core.native_tools import is_stop_tool_result

            if any(is_stop_tool_result(str(item.get("content") or "")) for item in tool_results):
                break

        from ageneval.task.core.native_tools import (
            compose_final_answer,
            ensure_required_tools,
            is_unusable_final,
        )

        if self.binding is not None:
            ensure_required_tools(binding=self.binding, task=task, recorder=tool_calls)
        if is_unusable_final(final_answer or "") or not tool_calls:
            final_answer = compose_final_answer(
                task.instruction, tool_calls, existing=final_answer or ""
            )

        elapsed = time.perf_counter() - start
        status = (
            "ok"
            if final_answer is not None
            else ("max_turns" if turns >= self.max_turns else "error")
        )
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,
            turns=turns,
            tool_calls=tuple(tool_calls),
            final_answer=final_answer,
            elapsed_seconds=elapsed,
        )

    async def _run_openai_compat(
        self,
        task: TaskInput,
        *,
        start: float,
        api_key: str | None,
        base_url: str,
    ) -> TaskTrace:
        """Tool loop over OpenAI-compatible ``/v1/chat/completions``.

        Used when the configured gateway rejects Anthropic ``/v1/messages``.
        """
        from openai import AsyncOpenAI

        from ageneval.task.core.native_tools import (
            compose_final_answer,
            ensure_required_tools,
            execute_recorded_tool,
            is_stop_tool_result,
            is_unusable_final,
            openai_function,
        )
        from ageneval.task.core.openai_compat import install_openai_compat, sanitize_tool_arguments

        install_openai_compat()
        if not api_key:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error="ClaudeSDKAgent requires OPENAI_API_KEY for chat/completions fallback.",
            )
        assert self.binding is not None
        if os.environ.get("A2E_TAU_NEED_WRITE") == "1":
            from ageneval.task.core.native_tools import maybe_force_retail_write_trace

            forced = await maybe_force_retail_write_trace(
                binding=self.binding,
                task=task,
                recorder=[],
                model=self.model,
                api_key=api_key,
                api_base=base_url or None,
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced is not None:
                return forced
        if os.environ.get("A2E_DSQA_FORCE") == "1":
            from ageneval.task.core.native_tools import maybe_force_dsqa_search_trace

            forced_ds = await maybe_force_dsqa_search_trace(
                binding=self.binding,
                task=task,
                recorder=[],
                model=self.model,
                api_key=api_key,
                api_base=base_url or None,
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced_ds is not None:
                return forced_ds
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=_llm_timeout(),
            max_retries=_max_retries(),
        )
        tools = _to_openai_tools(self.binding.tool_schemas)
        available = [
            str(openai_function(s).get("name") or "")
            for s in (self.binding.tool_schemas or [])
            if openai_function(s).get("name")
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.binding.render_system_prompt() + _NATIVE_TOOL_HINT},
            {"role": "user", "content": task.instruction},
        ]
        recorder: list[ToolCall] = []
        final_answer: str | None = None
        turns = 0
        write_schema = any(n in _RETAIL_WRITE for n in available)
        try:
            for turn in range(self.max_turns):
                turns = turn + 1
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": _MAX_TOKENS,
                }
                wrote = any(tc.name in _RETAIL_WRITE for tc in recorder)
                if tools:
                    kwargs["tools"] = tools
                    if turn == 0 or (write_schema and not wrote):
                        kwargs["tool_choice"] = "required"
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=min(_llm_timeout(), _remaining_deadline(start)),
                )
                msg = response.choices[0].message
                dumped = msg.model_dump() if hasattr(msg, "model_dump") else {}
                content = msg.content or dumped.get("reasoning_content") or ""
                raw_calls = list(msg.tool_calls or [])
                assistant: dict[str, Any] = {"role": "assistant", "content": content or " "}
                if raw_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": sanitize_tool_arguments(tc.function.arguments or "{}"),
                            },
                        }
                        for tc in raw_calls
                    ]
                messages.append(assistant)
                if not raw_calls:
                    parsed = _parse_json(str(content or ""))
                    if "action" in parsed:
                        name = str(parsed["action"])
                        text = execute_recorded_tool(
                            tool_name=name,
                            kwargs=parsed.get("arguments") or {},
                            executor=self.binding.tool_executor,
                            initial_state=task.initial_state,
                            recorder=recorder,
                            available=available,
                        )
                        messages.append({"role": "user", "content": f"TOOL[{name}] result: {text}"})
                        if is_stop_tool_result(text):
                            break
                        continue
                    if "final_answer" in parsed:
                        final_answer = str(parsed["final_answer"])
                    else:
                        final_answer = str(content or "") or None
                    if write_schema and not any(tc.name in _RETAIL_WRITE for tc in recorder):
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Call the write tool now (exchange, return, "
                                    "modify, or cancel). The customer already "
                                    "confirmed. Do not stop to ask again."
                                ),
                            }
                        )
                        continue
                    break
                stop = False
                for tc in raw_calls:
                    args_raw = sanitize_tool_arguments(tc.function.arguments or "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    text = execute_recorded_tool(
                        tool_name=tc.function.name,
                        kwargs=args,
                        executor=self.binding.tool_executor,
                        initial_state=task.initial_state,
                        recorder=recorder,
                        available=available,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": text,
                        }
                    )
                    if is_stop_tool_result(text):
                        stop = True
                if stop:
                    break
        except asyncio.TimeoutError:
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final_answer = compose_final_answer(
                task.instruction, recorder, existing=final_answer or ""
            )
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final_answer else "timeout",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final_answer or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final_answer else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
            )
        except Exception as exc:  # noqa: BLE001
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final_answer = compose_final_answer(
                task.instruction, recorder, existing=final_answer or ""
            )
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final_answer else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final_answer or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final_answer else (str(exc) or type(exc).__name__)[:1000],
            )

        if self.binding is not None:
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
        if is_unusable_final(final_answer or "") or not recorder:
            final_answer = compose_final_answer(
                task.instruction, recorder, existing=final_answer or ""
            )
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status="ok" if final_answer else "error",
            turns=turns,
            tool_calls=tuple(recorder),
            final_answer=final_answer,
            elapsed_seconds=time.perf_counter() - start,
            error=None if final_answer else "empty final",
        )


# ─── backwards-compat wrapper for τ-bench ─────────────────────────────────────


@dataclass
class ClaudeSDKTauAgent(ClaudeSDKAgent):
    """Thin wrapper: ``ClaudeSDKTauAgent(domain="retail")`` resolves the
    τ-bench binding automatically. New benchmarks should pass a custom
    ``AgentBinding`` directly to ``ClaudeSDKAgent``.
    """

    domain: str = "retail"
    binding: AgentBinding | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.binding is None:
            from ageneval.task.datasets.tau_bench import build_tau_bench_binding

            self.binding = build_tau_bench_binding(self.domain)  # type: ignore[arg-type]
        super().__post_init__()
