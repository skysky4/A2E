from __future__ import annotations

import asyncio

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall
from ageneval.task.datasets.tau_bench.session import TauOfficialSession
from ageneval.task.datasets.tau_bench.user_sim import (
    STOP_TOKEN,
    NaiveUserSimulationEnv,
)


class _PlanThenAct(AgentRunner):
    name = "fake-agent"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, task: TaskInput) -> TaskTrace:
        self.calls += 1
        if self.calls == 1:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                final_answer="I will look up the order.",
            )
        task.initial_state["__tau_db__"] = {"orders": {"#W1": {"status": "cancelled"}}}
        task.initial_state["__tau_domain__"] = "retail"
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status="ok",
            turns=1,
            tool_calls=(
                ToolCall(
                    name="cancel_pending_order",
                    arguments={"order_id": "#W1", "reason": "requested"},
                    result={"ok": True},
                ),
            ),
            final_answer="The order is cancelled.",
        )


def test_naive_session_continues_after_first_agent_response() -> None:
    inner = _PlanThenAct()
    trace = asyncio.run(
        TauOfficialSession(inner, user_strategy="naive").run(
            TaskInput(
                task_id="retail-1",
                instruction="Cancel my order.",
                initial_state={},
            )
        )
    )
    assert inner.calls >= 2
    assert [call.name for call in trace.tool_calls] == ["cancel_pending_order"]
    assert trace.raw["tau_hidden_instruction"] is True
    assert trace.raw["tau_data_hash"]
    assert "look up the order" in trace.raw["tau_spoken"]
    assert "order is cancelled" in trace.raw["tau_spoken"]


def test_stop_token_ends_without_running_inner(monkeypatch) -> None:
    class _StopUser(NaiveUserSimulationEnv):
        def reset(self, instruction: str | None = None) -> str:
            return STOP_TOKEN

    import ageneval.task.datasets.tau_bench.session as session_module

    monkeypatch.setattr(session_module, "load_user", lambda strategy=None: _StopUser())
    inner = _PlanThenAct()
    trace = asyncio.run(
        TauOfficialSession(inner, user_strategy="naive").run(
            TaskInput(task_id="retail-2", instruction="hidden")
        )
    )
    assert inner.calls == 0
    assert trace.turns == 0
    assert STOP_TOKEN in (trace.final_answer or "")


def test_user_simulator_failure_is_not_silently_downgraded(monkeypatch) -> None:
    class _BrokenUser:
        def reset(self, instruction: str | None = None) -> str:
            raise RuntimeError("unsupported user model")

    import ageneval.task.datasets.tau_bench.session as session_module

    monkeypatch.setattr(session_module, "load_user", lambda strategy=None, model=None: _BrokenUser())
    trace = asyncio.run(
        TauOfficialSession(
            _PlanThenAct(),
            user_strategy="llm",
            user_model="deepseek-v4-flash",
            user_error_policy="fail",
        ).run(TaskInput(task_id="retail-1", instruction="Cancel my order."))
    )

    assert trace.status == "error"
    assert trace.turns == 0
    assert "unsupported user model" in (trace.error or "")
