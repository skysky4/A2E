"""Official user-sim session (harness loop untouched)."""

from __future__ import annotations

import asyncio

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall
from ageneval.task.datasets.tau_bench.session import TauOfficialSession
from ageneval.task.datasets.tau_bench.user_sim import NaiveUserSimulationEnv, STOP_TOKEN


class _PlanThenAct(AgentRunner):
    """Mimics Google ADK: first run is plan text, later runs call a tool."""

    name = "fake-adk"

    def __init__(self) -> None:
        self.n = 0

    async def run(self, task: TaskInput) -> TaskTrace:
        self.n += 1
        if self.n == 1:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                final_answer="I will authenticate you first.",
            )
        if self.n == 2:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=2,
                tool_calls=(
                    ToolCall(
                        name="find_user_id_by_name_zip",
                        arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
                        result="yusuf_rossi_2320",
                    ),
                ),
                final_answer="I found your account. Shall I proceed with the exchange?",
            )
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status="ok",
            turns=1,
            final_answer="The exchange is complete.",
        )


def test_naive_session_continues_after_plan_text():
    agent = _PlanThenAct()
    session = TauOfficialSession(inner=agent, user_strategy="naive")
    task = TaskInput(
        task_id="retail-0000",
        instruction="You are Yusuf Rossi in 19122.",
        initial_state={},
    )
    trace = asyncio.run(session.run(task))
    assert [tc.name for tc in trace.tool_calls] == ["find_user_id_by_name_zip"]
    assert "exchange is complete" in (trace.final_answer or "").lower()
    assert agent.n >= 2
    assert trace.status in {"ok", "max_turns"}


def test_false_complete_without_write_asks_for_tool():
    class _ClaimWrite(AgentRunner):
        name = "fake-crewai"
        n = 0

        async def run(self, task: TaskInput) -> TaskTrace:
            self.n += 1
            if self.n == 1:
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok",
                    turns=1,
                    tool_calls=(
                        ToolCall(name="get_order_details", arguments={"order_id": "#W1"}, result={}),
                    ),
                    final_answer="The exchange has been submitted successfully.",
                )
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                tool_calls=(
                    ToolCall(
                        name="exchange_delivered_order_items",
                        arguments={"order_id": "#W1"},
                        result={"ok": True},
                    ),
                ),
                final_answer="Done.",
            )

    session = TauOfficialSession(inner=_ClaimWrite(), user_strategy="naive")
    trace = asyncio.run(
        session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
    )
    assert session.inner.n >= 2
    assert any(tc.name == "exchange_delivered_order_items" for tc in trace.tool_calls)


def test_user_stop_without_write_recovers_once():
    class _StopAfterFirst(_PlanThenAct):
        pass

    class _StopUser(NaiveUserSimulationEnv):
        def step(self, content: str) -> str:
            return STOP_TOKEN

    import ageneval.task.datasets.tau_bench.session as sess

    orig = sess.load_user
    sess.load_user = lambda strategy=None, model=None: _StopUser()
    try:
        agent = _StopAfterFirst()
        session = TauOfficialSession(inner=agent, user_strategy="naive")
        asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
        assert agent.n >= 2
    finally:
        sess.load_user = orig


def test_llm_user_reset_falls_back_to_naive():
    import ageneval.task.datasets.tau_bench.session as sess

    class _Boom:
        def reset(self, instruction=None):
            raise RuntimeError("quota")

    orig = sess.load_user
    sess.load_user = lambda strategy=None, model=None: _Boom()
    try:
        agent = _PlanThenAct()
        session = TauOfficialSession(inner=agent, user_strategy="llm")
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="You are Yusuf Rossi in 19122.", initial_state={}))
        )
        assert agent.n >= 1
        assert trace.status in {"ok", "max_turns"}
        assert trace.error is None
    finally:
        sess.load_user = orig


def test_stop_token_ends_without_inner_run():
    class _StopUser(NaiveUserSimulationEnv):
        def reset(self, instruction=None):
            return STOP_TOKEN

    session = TauOfficialSession(inner=_PlanThenAct(), user_strategy="naive")
    session.inner  # keep
    # inject stop user via monkeypatch of load_user in the module
    import ageneval.task.datasets.tau_bench.session as sess

    orig = sess.load_user
    sess.load_user = lambda strategy=None, model=None: _StopUser()
    try:
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert trace.turns == 0
    assert trace.tool_calls == ()
    assert STOP_TOKEN in (trace.final_answer or "")
