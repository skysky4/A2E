"""Official Sierra τ-bench conversation around an unchanged harness.

Harnesses such as Google ADK treat the first customer-facing text as the
final answer. Official τ-bench does not: that text is ``respond``, and the
LLM user simulator replies until ``###STOP###``. This wrapper re-invokes the
same ``AgentRunner`` with the next user utterance and a shared live DB.
No harness ``run()`` loop is modified. There is no naive user fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.budget import remaining_deadline, run_deadline
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall

from ageneval.task.datasets.tau_bench.runtime import WRITE_TOOLS, _is_tool_error
from ageneval.task.datasets.tau_bench.user_sim import (
    STOP_TOKEN,
    load_user,
    looks_like_hidden_script,
)


def wrap_tau_official_session(agent: AgentRunner) -> AgentRunner:
    if isinstance(agent, TauOfficialSession):
        return agent
    return TauOfficialSession(inner=agent)


class TauOfficialSession(AgentRunner):
    """One official LLM user-sim conversation; ``inner`` is the unchanged harness."""

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
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error=f"official tau user simulator reset failed: {exc}"[:1000],
                raw={
                    "tau_user_strategy": type(user).__name__,
                    "tau_hidden_instruction": True,
                    "tau_opening": None,
                },
            )
        if looks_like_hidden_script(opening):
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error="official user simulator leaked the hidden Sierra script",
                raw={
                    "tau_user_strategy": type(user).__name__,
                    "tau_hidden_instruction": True,
                    "tau_opening": opening,
                },
            )
        if not opening.strip():
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error="official user simulator returned an empty opening",
                raw={
                    "tau_user_strategy": type(user).__name__,
                    "tau_hidden_instruction": True,
                    "tau_opening": opening,
                },
            )
        if STOP_TOKEN in opening:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=0,
                final_answer=opening,
                elapsed_seconds=time.perf_counter() - start,
                raw={
                    "tau_user_strategy": type(user).__name__,
                    "tau_responds": 0,
                    "tau_hidden_instruction": True,
                    "tau_opening": opening,
                },
            )

        transcript: list[tuple[str, str]] = [("customer", opening)]
        tools: list[ToolCall] = []
        seen_keys: set[tuple[str, str]] = set()
        final = ""
        turns = 0
        last_text = ""
        idle = 0
        episode_budget = _episode_budget()
        episode_used = 0
        error: str | None = None
        status = "ok"

        for respond_i in range(episode_budget):
            remaining = episode_budget - episode_used
            if remaining <= 0:
                status = "max_turns"
                break
            if time.perf_counter() - start >= run_deadline():
                status = "error"
                error = (
                    f"official run_deadline {run_deadline():.0f}s exceeded "
                    f"after {episode_used} episode actions"
                )
                break
            _set_inner_budget(self.inner, remaining)
            visible = _agent_visible(transcript)
            inner_task = TaskInput(
                task_id=task.task_id,
                instruction=visible,
                initial_state=state,
                metadata=dict(task.metadata or {}),
            )
            try:
                trace = await asyncio.wait_for(
                    self.inner.run(inner_task),
                    timeout=remaining_deadline(start),
                )
            except asyncio.TimeoutError:
                error = (
                    f"official run_deadline {run_deadline():.0f}s exceeded "
                    f"during inner harness run"
                )
                status = "error"
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:1000]
                status = "error"
                break

            n_new = 0
            for tc in trace.tool_calls or ():
                key = (tc.name, json.dumps(dict(tc.arguments or {}), sort_keys=True, default=str))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                tools.append(tc)
                transcript.append(("tool", _tool_line(tc)))
                n_new += 1
            text = (trace.final_answer or "").strip()
            if text:
                final = text
                transcript.append(("agent", text))
            used = max(int(trace.turns or 0), n_new + (1 if text else 0), 1)
            episode_used += used
            turns = episode_used

            if trace.status == "error" and not tools and not text:
                # Official τ ``respond`` is plain customer-facing text. Some
                # harnesses mark that turn ``error`` because they cleaned the
                # text as a QA final. Continue the user-sim whenever there is
                # a reply; only abort a silent failure.
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

            try:
                nxt = user.step(text)
            except Exception as exc:  # noqa: BLE001
                error = f"official tau user simulator step failed: {exc}"[:1000]
                status = "error"
                break
            nxt = (nxt or "").strip()
            if looks_like_hidden_script(nxt):
                error = "official user simulator leaked the hidden Sierra script"
                status = "error"
                break
            if STOP_TOKEN in nxt:
                status = "ok"
                break
            if not nxt:
                # Official Sierra only stops on ###STOP###, never on empty.
                idle += 1
                if idle >= 2:
                    status = "ok"
                    break
                continue
            transcript.append(("customer", nxt))
            if episode_used >= episode_budget:
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
                "tau_opening": next(
                    (text for role, text in transcript if role == "customer"), opening
                ),
                "tau_write": _has_successful_write(tools),
                "tau_episode_actions": episode_used,
                "tau_episode_budget": episode_budget,
            },
        )


def _episode_budget() -> int:
    """Official τ ``max_num_steps`` is the episode action budget (default 30)."""
    raw = os.environ.get("A2E_MAX_TURNS") or os.environ.get("A2E_TAU_MAX_RESPONDS") or "30"
    return max(1, int(raw))


def _set_inner_budget(inner: object, remaining: int) -> None:
    """Give the unchanged harness only the leftover official episode steps."""
    leftover = max(1, int(remaining))
    for obj in (inner, getattr(inner, "inner", None)):
        if obj is None:
            continue
        for attr in ("max_turns", "max_steps"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, leftover)
                except Exception:  # noqa: BLE001
                    continue


def _has_successful_write(tools: list[ToolCall]) -> bool:
    for tc in tools:
        if tc.name not in WRITE_TOOLS:
            continue
        if tc.error or _is_tool_error(tc.result):
            continue
        return True
    return False


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
    model does not re-ask for facts it already used. The hidden script is
    never included.
    """
    if len(transcript) == 1 and transcript[0][0] == "customer":
        return transcript[0][1]
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
    lines.append("Respond to the latest Customer line.")
    return "\n".join(lines)
