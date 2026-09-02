"""Sierra-style τ-bench conversation around an unchanged agent harness."""

from __future__ import annotations

import json
import os
import time

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
_LOOKUP_TOOLS = {
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "get_reservation_details",
}


def wrap_tau_official_session(
    agent: AgentRunner,
    *,
    user_strategy: str | None = None,
    user_model: str | None = None,
    user_error_policy: str = "fail",
    max_responds: int | None = None,
) -> AgentRunner:
    if isinstance(agent, TauOfficialSession):
        return agent
    return TauOfficialSession(
        inner=agent,
        user_strategy=user_strategy,
        user_model=user_model,
        user_error_policy=user_error_policy,
        max_responds=max_responds,
    )


class TauOfficialSession(AgentRunner):
    """Run one user-sim conversation by reusing the same agent and task state."""

    def __init__(
        self,
        inner: AgentRunner,
        *,
        user_strategy: str | None = None,
        user_model: str | None = None,
        user_error_policy: str = "fail",
        max_responds: int | None = None,
    ) -> None:
        if user_error_policy not in {"fail", "fallback_naive"}:
            raise ValueError("user_error_policy must be 'fail' or 'fallback_naive'")
        if max_responds is not None and max_responds < 1:
            raise ValueError("max_responds must be positive")
        self.inner = inner
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_error_policy = user_error_policy
        self.max_responds = max_responds or _max_responds()
        self.name = getattr(inner, "name", "tau-session")
        self.binding = getattr(inner, "binding", None)

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        state = task.initial_state
        if not isinstance(state, dict):
            state = dict(state or {})
        user = (
            load_user(self.user_strategy, self.user_model)
            if self.user_model
            else load_user(self.user_strategy)
        )
        try:
            opening = user.reset((task.instruction or "").strip())
        except Exception as exc:
            from ageneval.task.datasets.tau_bench.user_sim import NaiveUserSimulationEnv

            if isinstance(user, NaiveUserSimulationEnv) or self.user_error_policy == "fail":
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="error",
                    turns=0,
                    elapsed_seconds=time.perf_counter() - start,
                    error=f"tau user simulator reset failed: {exc}"[:1000],
                )
            user = NaiveUserSimulationEnv()
            opening = user.reset((task.instruction or "").strip())

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
        error: str | None = None
        status = "ok"

        for respond_index in range(self.max_responds):
            inner_task = TaskInput(
                task_id=task.task_id,
                instruction=_agent_visible(transcript),
                initial_state=state,
                metadata=dict(task.metadata or {}),
            )
            try:
                trace = await self.inner.run(inner_task)
            except Exception as exc:
                error = str(exc)[:1000]
                status = "error"
                break

            turns += int(trace.turns or 0)
            for tool_call in trace.tool_calls or ():
                key = (
                    tool_call.name,
                    json.dumps(
                        dict(tool_call.arguments or {}),
                        sort_keys=True,
                        default=str,
                    ),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                tools.append(tool_call)
                transcript.append(("tool", _tool_line(tool_call)))

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
            if text == last_text and not trace.tool_calls:
                idle += 1
                if idle >= 2:
                    break
            else:
                idle = 0
            last_text = text

            if _has_successful_write(tools):
                try:
                    next_user = user.step(text)
                except Exception as exc:
                    if self.user_error_policy == "fail":
                        error = f"tau user simulator step failed: {exc}"[:1000]
                        status = "error"
                        break
                    next_user = STOP_TOKEN
                if STOP_TOKEN in (next_user or "") or not (next_user or "").strip():
                    break
                transcript.append(("customer", (next_user or "").strip()))
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
            if any(call.name == "transfer_to_human_agents" for call in tools) and recovered < 2:
                recovered += 1
                transcript.append(
                    (
                        "customer",
                        "Do not transfer me to a human. Finish with your write tools.",
                    )
                )
                continue

            try:
                next_user = (user.step(text) or "").strip()
            except Exception as exc:
                error = f"tau user simulator step failed: {exc}"[:1000]
                status = "error"
                break
            if not next_user or STOP_TOKEN in next_user:
                if recovered < 2 and not _has_successful_write(tools):
                    recovered += 1
                    next_user = (
                        "That is not finished. Complete my request with your tools; "
                        "I do not want a human transfer."
                    )
                else:
                    break
            transcript.append(("customer", next_user))
            if respond_index == self.max_responds - 1:
                status = "max_turns"

        from ageneval.task.datasets.tau_bench.reward import data_hash

        tau_db = state.get("__tau_db__")
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
                "tau_user_model": self.user_model,
                "tau_user_error_policy": self.user_error_policy,
                "tau_responds": sum(1 for role, _ in transcript if role == "customer"),
                "tau_hidden_instruction": True,
                "tau_write": _has_successful_write(tools),
                "tau_domain": state.get("__tau_domain__"),
                "tau_data_hash": data_hash(tau_db) if isinstance(tau_db, dict) else None,
                "tau_spoken": " ".join(
                    text for role, text in transcript if role == "agent"
                ),
            },
        )


def _max_responds() -> int:
    return max(1, int(os.environ.get("A2E_TAU_MAX_RESPONDS", "12")))


def _has_successful_write(tools: list[ToolCall]) -> bool:
    return any(
        call.name in WRITE_TOOLS
        and not call.error
        and not _is_tool_error(call.result)
        for call in tools
    )


def _looks_like_false_complete(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _FALSE_COMPLETE)


def _looks_like_confirm(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in {"yes", "y", "ok", "okay", "sure", "yeah", "yep"}:
        return True
    return any(marker in lowered for marker in _CONFIRM_MARKERS)


def _tool_line(tool_call: ToolCall) -> str:
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    try:
        rendered_args = ", ".join(f"{key}={args[key]!r}" for key in list(args)[:8])
    except Exception:
        rendered_args = str(args)[:200]
    if tool_call.result is None:
        result = ""
    elif isinstance(tool_call.result, str):
        result = tool_call.result
    else:
        result = json.dumps(tool_call.result, default=str)
    extra = ""
    if tool_call.name == "get_order_details" and isinstance(tool_call.result, dict):
        item_ids = [
            str(item.get("item_id"))
            for item in tool_call.result.get("items") or ()
            if isinstance(item, dict) and item.get("item_id")
        ]
        if item_ids:
            extra = f" item_ids_on_order={item_ids}"
    limit = 4000 if tool_call.name in _LOOKUP_TOOLS else 400
    if len(result) > limit:
        result = result[:limit] + "…"
    error = f" error={tool_call.error}" if tool_call.error else ""
    return f"{tool_call.name}({rendered_args}) -> {result}{extra}{error}"


def _agent_visible(transcript: list[tuple[str, str]]) -> str:
    if len(transcript) == 1 and transcript[0][0] == "customer":
        return transcript[0][1]
    last_customer = next(
        (text for role, text in reversed(transcript) if role == "customer"),
        "",
    )
    lines = [
        "Continue this customer-service conversation. The customer script is "
        "hidden; only the lines below are visible. Call tools when needed and "
        "reuse prior [Tool] results instead of repeating calls.",
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
    failed_write = any(
        role == "tool"
        and any(name in text for name in WRITE_TOOLS)
        and "error" in text.lower()
        for role, text in transcript
    )
    if failed_write:
        lines.append("A write tool failed. Correct its arguments and call it again.")
    elif _looks_like_confirm(last_customer):
        lines.append(
            "The customer confirmed. Call the matching write tool now instead of "
            "transferring to a human."
        )
    else:
        lines.append("Respond to the latest Customer line.")
    return "\n".join(lines)


__all__ = ["TauOfficialSession", "wrap_tau_official_session"]
