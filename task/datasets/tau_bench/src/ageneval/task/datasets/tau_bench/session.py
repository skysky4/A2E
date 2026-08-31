"""Official Sierra τ-bench conversation around an unchanged harness.

Harnesses such as Google ADK treat the first customer-facing text as the
final answer. Official τ-bench does not: that text is ``respond``, and the
user simulator replies. This wrapper re-invokes the *same* ``AgentRunner``
with the next user utterance and a shared ``initial_state`` (the live DB).
No harness ``run()`` loop is modified.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall

from ageneval.task.datasets.tau_bench.runtime import WRITE_TOOLS, _is_tool_error
from ageneval.task.datasets.tau_bench.user_sim import STOP_TOKEN, load_user

_CONFIRM_MARKERS = (
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "go ahead",
    "please proceed",
    "please do",
    "confirm",
    "approved",
)


def wrap_tau_official_session(agent: AgentRunner) -> AgentRunner:
    if isinstance(agent, TauOfficialSession):
        return agent
    return TauOfficialSession(inner=agent)


class TauOfficialSession(AgentRunner):
    """One official user-sim conversation; ``inner`` is the unchanged harness."""

    def __init__(self, inner: AgentRunner, *, user_strategy: str | None = None) -> None:
        self.inner = inner
        self.user_strategy = user_strategy
        self.name = getattr(inner, "name", "tau-session")
        self.binding = getattr(inner, "binding", None)

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        state = task.initial_state
        if not isinstance(state, dict):
            state = dict(state or {})
        hidden = (task.instruction or "").strip()
        user = load_user(self.user_strategy)
        try:
            opening = user.reset(hidden)
        except Exception as exc:  # noqa: BLE001
            from ageneval.task.datasets.tau_bench.user_sim import NaiveUserSimulationEnv

            if isinstance(user, NaiveUserSimulationEnv):
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="error",
                    turns=0,
                    elapsed_seconds=time.perf_counter() - start,
                    error=f"tau user simulator reset failed: {exc}"[:1000],
                )
            # Official LLM user is preferred; quota/gateway failures fall back
            # so the harness can still produce a trajectory.
            user = NaiveUserSimulationEnv()
            opening = user.reset(hidden)

        if STOP_TOKEN in opening:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=0,
                final_answer=opening,
                elapsed_seconds=time.perf_counter() - start,
                raw={"tau_user_strategy": type(user).__name__, "tau_responds": 0},
            )

        transcript: list[tuple[str, str]] = [("customer", opening)]
        tools: list[ToolCall] = []
        seen_keys: set[tuple[str, str]] = set()
        final = ""
        turns = 0
        last_text = ""
        idle = 0
        recovered = 0
        max_responds = _max_responds()
        error: str | None = None
        status = "ok"

        for respond_i in range(max_responds):
            visible = _agent_visible(transcript)
            inner_task = TaskInput(
                task_id=task.task_id,
                instruction=visible,
                initial_state=state,
                metadata=dict(task.metadata or {}),
            )
            try:
                trace = await self.inner.run(inner_task)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:1000]
                status = "error"
                break

            turns += int(trace.turns or 0)
            for tc in trace.tool_calls or ():
                key = (tc.name, json.dumps(dict(tc.arguments or {}), sort_keys=True, default=str))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                tools.append(tc)
                transcript.append(("tool", _tool_line(tc)))
            text = (trace.final_answer or "").strip()
            if text:
                final = text
                transcript.append(("agent", text))

            if trace.status == "error" and not tools:
                status = "error"
                error = trace.error
                break
            if not text:
                idle += 1
                if idle >= 2:
                    status = trace.status or "error"
                    error = error or trace.error
                    break
                continue

            if text == last_text and not (trace.tool_calls or ()):
                idle += 1
                if idle >= 2:
                    status = "ok"
                    break
            else:
                idle = 0
            last_text = text

            if _has_successful_write(tools):
                try:
                    nxt = user.step(text)
                except Exception:  # noqa: BLE001
                    nxt = STOP_TOKEN
                if STOP_TOKEN in (nxt or "") or not (nxt or "").strip():
                    status = "ok"
                    break
                transcript.append(("customer", (nxt or "").strip()))
                continue

            if _looks_like_false_complete(text) and recovered < 2:
                recovered += 1
                transcript.append(
                    (
                        "customer",
                        "You described a write but did not call the write tool. "
                        "Call exchange, return, modify, or cancel now.",
                    )
                )
                continue
            if (
                any(tc.name == "transfer_to_human_agents" for tc in tools)
                and recovered < 2
            ):
                recovered += 1
                transcript.append(
                    (
                        "customer",
                        "Do not transfer me to a human. Finish with your write "
                        "tools (exchange/return/modify/cancel).",
                    )
                )
                continue

            try:
                nxt = user.step(text)
            except Exception as exc:  # noqa: BLE001
                error = f"tau user simulator step failed: {exc}"[:1000]
                status = "error"
                break
            nxt = (nxt or "").strip()
            if not nxt or STOP_TOKEN in nxt:
                if recovered < 2 and not _has_successful_write(tools):
                    recovered += 1
                    nxt = (
                        "That is not finished. Please complete my request with your "
                        "tools. I do not want a human transfer."
                    )
                else:
                    status = "ok"
                    break
            transcript.append(("customer", nxt))
            if respond_i == max_responds - 1:
                status = "max_turns"

        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,  # type: ignore[arg-type]
            turns=turns,
            tool_calls=tuple(tools),
            final_answer=final or None,
            elapsed_seconds=time.perf_counter() - start,
            error=error,
            raw={
                "tau_user_strategy": type(user).__name__,
                "tau_responds": sum(1 for role, _ in transcript if role == "customer"),
                "tau_hidden_instruction": True,
                "tau_write": _has_successful_write(tools),
            },
        )


def _max_responds() -> int:
    return max(1, int(os.environ.get("A2E_TAU_MAX_RESPONDS", "12")))


def _has_successful_write(tools: list[ToolCall]) -> bool:
    for tc in tools:
        if tc.name not in WRITE_TOOLS:
            continue
        if tc.error or _is_tool_error(tc.result):
            continue
        return True
    return False


_FALSE_COMPLETE = (
    "has been submitted",
    "exchange has been",
    "successfully submitted",
    "request has been submitted",
    "completed the exchange",
    "already submitted",
    "i've submitted",
    "i have submitted",
)


def _looks_like_false_complete(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in _FALSE_COMPLETE)


def _looks_like_confirm(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t in {"yes", "y", "ok", "okay", "sure", "yeah", "yep"}:
        return True
    return any(marker in t for marker in _CONFIRM_MARKERS)


_LOOKUP_TOOLS = {
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "get_reservation_details",
}


def _tool_line(tc: ToolCall) -> str:
    args = tc.arguments if isinstance(tc.arguments, dict) else {}
    try:
        arg_s = ", ".join(f"{k}={args[k]!r}" for k in list(args)[:8])
    except Exception:  # noqa: BLE001
        arg_s = str(args)[:200]
    result = "" if tc.result is None else (
        json.dumps(tc.result, default=str) if not isinstance(tc.result, str) else tc.result
    )
    extra = ""
    if tc.name == "get_order_details" and isinstance(tc.result, dict):
        ids = [
            str(it.get("item_id"))
            for it in (tc.result.get("items") or [])
            if isinstance(it, dict) and it.get("item_id")
        ]
        if ids:
            extra = f" item_ids_on_order={ids}"
    limit = 4000 if tc.name in _LOOKUP_TOOLS else 400
    if len(result) > limit:
        result = result[:limit] + "…"
    err = f" error={tc.error}" if tc.error else ""
    return f"{tc.name}({arg_s}) -> {result}{extra}{err}"


def _agent_visible(transcript: list[tuple[str, str]]) -> str:
    """Replay the conversation as the next user message.

    Harnesses start a fresh session on every ``run()``. Official τ keeps one
    message list; this transcript is the binding-side substitute so the
    model does not re-ask for facts it already used.
    """
    if len(transcript) == 1 and transcript[0][0] == "customer":
        return transcript[0][1]
    last_customer = ""
    for role, text in reversed(transcript):
        if role == "customer":
            last_customer = text
            break
    lines = [
        "Continue this customer-service conversation. "
        "The customer script is hidden; only the lines below are visible. "
        "Call tools when you need records. Do not repeat a tool with the same "
        "arguments — reuse the [Tool] results below.",
        "",
    ]
    for role, text in transcript:
        if role == "customer":
            lines.append(f"Customer: {text}")
        elif role == "agent":
            lines.append(f"Agent: {text}")
        else:
            lines.append(f"[Tool] {text}")
    lines.append("")
    failed_write = False
    for role, text in transcript:
        if role == "tool" and any(w in text for w in WRITE_TOOLS) and "error" in text.lower():
            failed_write = True
    if failed_write:
        lines.append(
            "A write tool returned an error. Fix the arguments and call it again. "
            "item_ids must be item_ids_on_order from get_order_details, not "
            "other product variants. new_item_ids must be available:true "
            "variants of the same product. payment_method_id is the id field "
            "(credit_card_… / gift_card_…), not the card last-four."
        )
    elif _looks_like_confirm(last_customer):
        lines.append(
            "The customer has confirmed. Call the matching write tool now. "
            "Do not transfer to a human when exchange, return, cancel, or "
            "modify tools can fulfill the request."
        )
    else:
        lines.append("Respond to the latest Customer line.")
    return "\n".join(lines)
