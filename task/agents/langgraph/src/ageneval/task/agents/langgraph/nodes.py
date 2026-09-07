"""LangGraph node implementations — dataset-agnostic.

Each node wraps itself in an explicit OpenInference ``AGENT`` span so the
A2E UI surfaces three distinct agents per task (router / executor /
responder) instead of a generic CHAIN tree.

The nodes only know about the active ``AgentBinding`` (per-call argument).
Switching benchmarks therefore costs **one** new ``binding.py`` in
``task/datasets/<bench>/`` — these node files never change.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api

from ageneval.task.core import AgentBinding, TaskInput
from ageneval.task.core.native_tools import (
    canonicalize_tool_args,
    parse_leaked_tool_calls,
    unwrap_tool_kwargs,
)

logger = logging.getLogger(__name__)

_AGENT_KIND_KEY = SpanAttributes.OPENINFERENCE_SPAN_KIND
_AGENT_KIND_VAL = OpenInferenceSpanKindValues.AGENT.value
_TOOL_KIND_VAL = OpenInferenceSpanKindValues.TOOL.value
_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)

_tracer = trace_api.get_tracer(__name__)


def router_node(*, state: dict[str, Any], llm: Any, binding: AgentBinding) -> dict[str, Any]:
    """ROUTER agent: picks the next tool call (or signals "done")."""
    with _tracer.start_as_current_span("agent.router") as span:
        span.set_attribute(_AGENT_KIND_KEY, _AGENT_KIND_VAL)
        span.set_attribute("agent.name", "router")
        span.set_attribute("a2e.binding", binding.name)

        task: TaskInput = state["task"]
        history = state.get("tool_calls", [])
        system = binding.render_system_prompt()
        tool_names = [
            str(schema.get("function", {}).get("name", ""))
            for schema in binding.tool_schemas
            if schema.get("function", {}).get("name")
        ]
        user = _router_user_prompt(
            task=task,
            history=history,
            tool_names=tool_names,
        )

        reply_text = _invoke_llm(llm, system, user, span)
        leaked = parse_leaked_tool_calls(reply_text, allowed_names=set(tool_names))
        if leaked:
            call = leaked[0]
            return {
                "next_action": {
                    "name": str(call.get("name") or ""),
                    "arguments": dict(call.get("arguments") or {}),
                },
            }
        parsed = _parse_json(reply_text)
        if "action" in parsed:
            return {
                "next_action": {
                    "name": str(parsed["action"]),
                    "arguments": parsed.get("arguments", {}) or {},
                },
            }
        if "final_answer" in parsed:
            return {"final_answer": str(parsed["final_answer"]), "next_action": None}
        logger.warning("router got unstructured reply, terminating")
        return {"final_answer": reply_text.strip() or "(no answer)", "next_action": None}


def executor_run(*, state: dict[str, Any], binding: AgentBinding) -> dict[str, Any]:
    """EXECUTOR agent: dispatches the chosen tool via ``binding.tool_executor``.

    The binding's ``tool_executor`` is the **only** dataset-specific code
    in this graph. Wrapped in its own AGENT span so the trace clearly
    shows when each tool ran.
    """
    action = state.get("next_action") or {}
    name = str(action.get("name") or "noop")
    raw_args = action.get("arguments", {}) or {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    args = canonicalize_tool_args(name, unwrap_tool_kwargs(raw_args))
    task: TaskInput = state["task"]

    args_json = json.dumps(args, default=str)
    with _tracer.start_as_current_span("agent.executor") as span:
        span.set_attribute(_AGENT_KIND_KEY, _AGENT_KIND_VAL)
        span.set_attribute("agent.name", "executor")
        # Nested TOOL-kind span so A2E renders the tool call as a
        # first-class step in the trajectory (with input / output panels),
        # not just opaque attributes on the executor agent span.
        with _tracer.start_as_current_span(f"tool.{name}") as tool_span:
            tool_span.set_attribute(_AGENT_KIND_KEY, _TOOL_KIND_VAL)
            tool_span.set_attribute(SpanAttributes.TOOL_NAME, name)
            tool_span.set_attribute(SpanAttributes.TOOL_PARAMETERS, args_json)
            tool_span.set_attribute(SpanAttributes.INPUT_VALUE, args_json)
            try:
                result = binding.tool_executor(name, args, task.initial_state)
            except Exception as exc:  # noqa: BLE001
                tool_span.record_exception(exc)
                span.record_exception(exc)
                result = {"error": str(exc)}
            result_json = json.dumps(result, default=str)
            tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result_json)
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.parameters", args_json)
        span.set_attribute("tool.result", result_json)

    tool_calls = list(state.get("tool_calls", []))
    tool_calls.append({"name": name, "arguments": args, "result": result})
    return {
        "tool_calls": tool_calls,
        "next_action": None,
        "turns": int(state.get("turns", 0)) + 1,
    }


def responder_node(*, state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """RESPONDER agent: composes the final natural-language answer."""
    with _tracer.start_as_current_span("agent.responder") as span:
        span.set_attribute(_AGENT_KIND_KEY, _AGENT_KIND_VAL)
        span.set_attribute("agent.name", "responder")

        existing = state.get("final_answer")
        if existing:
            return {"final_answer": existing}

        task: TaskInput = state["task"]
        history = state.get("tool_calls", [])
        system = (
            "You are the responder. Summarise what was done for the customer in one short paragraph."
        )
        user = (
            f"Customer request: {task.instruction}\n"
            f"Tool calls made: {json.dumps(history, default=str)}\n"
            "Write a concise customer-facing reply."
        )
        text = _invoke_llm(llm, system, user, span)
        return {"final_answer": text.strip() or "(no answer)"}


# ─── helpers ──────────────────────────────────────────────────────────────────


def _invoke_llm(llm: Any, system: str, user: str, span: trace_api.Span) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        ai_msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001
        span.record_exception(exc)
        raise
    return getattr(ai_msg, "content", "") or ""


def _router_user_prompt(
    *,
    task: TaskInput,
    history: list[dict[str, Any]],
    tool_names: list[str] | None = None,
) -> str:
    available = ", ".join(tool_names or []) or "(none)"
    tool_policy = (
        "Use an available tool before finishing when a tool can advance the task.\n"
        if tool_names
        else ""
    )
    return (
        f"Customer instruction: {task.instruction}\n"
        f"Initial state: {json.dumps(_public_state(task.initial_state), default=str)}\n"
        f"History so far: {json.dumps(history, default=str)}\n"
        f"Available action names: {available}\n"
        f"{tool_policy}"
        "Return exactly one JSON object with no prose or Markdown.\n"
        'To call a tool, return {"action":"<available action name>",'
        '"arguments":{...}}.\n'
        'To finish, return {"final_answer":"<answer>"}.\n'
        "Pick the next action."
    )


def _public_state(state: Any) -> Any:
    """Drop live DB / sandbox objects so the router prompt stays official-small."""
    if not isinstance(state, dict):
        return state
    skip = {"__tau_db__", "_gdp_sandbox"}
    out: dict[str, Any] = {}
    for key, val in state.items():
        if key in skip or str(key).startswith("_"):
            continue
        if key == "reference_files" and isinstance(val, dict):
            out[key] = sorted(str(name) for name in val)
            continue
        out[key] = val
    return out


def _parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw[:4].lower().startswith("json"):
            raw = raw[4:].lstrip()
    start = raw.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    match = _JSON_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
