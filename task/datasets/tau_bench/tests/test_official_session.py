"""Official LLM user-sim session (harness loop untouched, no naive fallback)."""

from __future__ import annotations

import asyncio

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall
from ageneval.task.datasets.tau_bench.session import TauOfficialSession
from ageneval.task.datasets.tau_bench.user_sim import STOP_TOKEN


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
            tool_calls=(
                ToolCall(
                    name="exchange_delivered_order_items",
                    arguments={"order_id": "#W1"},
                    result={"ok": True},
                ),
            ),
            final_answer="The exchange is complete.",
        )


class _ScriptedUser:
    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.n = 0

    def reset(self, instruction: str | None = None) -> str:
        self.n += 1
        return self.lines.pop(0) if self.lines else STOP_TOKEN

    def step(self, content: str) -> str:
        self.n += 1
        return self.lines.pop(0) if self.lines else STOP_TOKEN


def _patch_user(fake):
    import ageneval.task.datasets.tau_bench.session as sess

    orig = sess.load_user
    sess.load_user = lambda strategy=None, model=None: fake
    return orig, sess


def test_official_session_continues_after_plan_text():
    agent = _PlanThenAct()
    fake = _ScriptedUser(
        [
            "Hi, I need help with an order.",
            "My name is Yusuf Rossi, zip 19122.",
            "Yes, please go ahead.",
            STOP_TOKEN,
        ]
    )
    orig, sess = _patch_user(fake)
    try:
        session = TauOfficialSession(inner=agent)
        task = TaskInput(
            task_id="retail-0000",
            instruction="You are Yusuf Rossi in 19122.",
            initial_state={},
        )
        trace = asyncio.run(session.run(task))
    finally:
        sess.load_user = orig
    assert [tc.name for tc in trace.tool_calls] == [
        "find_user_id_by_name_zip",
        "exchange_delivered_order_items",
    ]
    assert "exchange is complete" in (trace.final_answer or "").lower()
    assert agent.n >= 2
    assert trace.status in {"ok", "max_turns"}
    assert trace.raw["tau_user_strategy"] == "_ScriptedUser"
    assert not str(trace.raw.get("tau_opening") or "").startswith("You are Yusuf")


def test_llm_user_reset_failure_is_error():
    import ageneval.task.datasets.tau_bench.session as sess

    class _Boom:
        def reset(self, instruction=None):
            raise RuntimeError("quota")

    orig = sess.load_user
    sess.load_user = lambda strategy=None, model=None: _Boom()
    try:
        agent = _PlanThenAct()
        session = TauOfficialSession(inner=agent)
        trace = asyncio.run(
            session.run(
                TaskInput(
                    task_id="x",
                    instruction="You are Yusuf Rossi in 19122.",
                    initial_state={},
                )
            )
        )
        assert agent.n == 0
        assert trace.status == "error"
        assert trace.error and "reset failed" in trace.error
        assert trace.raw["tau_hidden_instruction"] is True
    finally:
        sess.load_user = orig


def test_leaked_opening_is_error():
    orig, sess = _patch_user(
        _ScriptedUser(
            ["You are Yusuf Rossi in 19122. Please exchange order #W2378156."]
        )
    )
    try:
        session = TauOfficialSession(inner=_PlanThenAct())
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert trace.status == "error"
    assert trace.error and "leaked" in trace.error
    assert trace.turns == 0


def test_stop_token_ends_without_inner_run():
    orig, sess = _patch_user(_ScriptedUser([STOP_TOKEN]))
    agent = _PlanThenAct()
    try:
        session = TauOfficialSession(inner=agent)
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert agent.n == 0
    assert trace.turns == 0
    assert trace.tool_calls == ()
    assert STOP_TOKEN in (trace.final_answer or "")


def test_session_continues_after_harness_error_respond():
    class _AskZip(AgentRunner):
        name = "ask"

        def __init__(self) -> None:
            self.n = 0

        async def run(self, task: TaskInput) -> TaskTrace:
            self.n += 1
            if self.n == 1:
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="error",
                    turns=1,
                    final_answer="Please provide your ZIP code so I can authenticate you.",
                )
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                tool_calls=(
                    ToolCall(
                        name="find_user_id_by_name_zip",
                        arguments={"zip": "19122"},
                        result="u1",
                    ),
                ),
                final_answer="Found your account.",
            )

    agent = _AskZip()
    orig, sess = _patch_user(_ScriptedUser(["Hi", "My zip is 19122.", STOP_TOKEN]))
    try:
        session = TauOfficialSession(inner=agent)
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert agent.n >= 2
    assert [tc.name for tc in trace.tool_calls] == ["find_user_id_by_name_zip"]
    assert trace.status in {"ok", "max_turns"}


def test_empty_user_step_is_not_stop():
    class _Once(AgentRunner):
        name = "once"

        def __init__(self) -> None:
            self.n = 0

        async def run(self, task: TaskInput) -> TaskTrace:
            self.n += 1
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                final_answer=f"Please provide your ZIP. ({self.n})",
            )

    agent = _Once()
    orig, sess = _patch_user(
        _ScriptedUser(["Hi, I need help.", "", "My zip is 19122.", STOP_TOKEN])
    )
    try:
        session = TauOfficialSession(inner=agent)
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert agent.n >= 2
    assert trace.status in {"ok", "max_turns"}


def test_session_passes_remaining_episode_budget(monkeypatch):
    monkeypatch.setenv("A2E_MAX_TURNS", "10")

    class _Spy(AgentRunner):
        name = "spy"

        def __init__(self) -> None:
            self.max_turns = 99
            self.seen: list[int] = []

        async def run(self, task: TaskInput) -> TaskTrace:
            self.seen.append(int(self.max_turns))
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=3,
                tool_calls=(
                    ToolCall(name="find_user_id_by_name_zip", arguments={}, result="u1"),
                ),
                final_answer="Working on it.",
            )

    agent = _Spy()
    orig, sess = _patch_user(
        _ScriptedUser(["Hi", "zip 19122", "yes", STOP_TOKEN])
    )
    try:
        session = TauOfficialSession(inner=agent)
        asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert agent.seen
    assert agent.seen[0] == 10
    if len(agent.seen) > 1:
        assert agent.seen[1] == 7


def test_session_stops_on_official_run_deadline(monkeypatch):
    monkeypatch.setenv("A2E_RUN_DEADLINE", "0.05")

    class _Slow(AgentRunner):
        name = "slow"

        async def run(self, task: TaskInput) -> TaskTrace:
            await asyncio.sleep(0.2)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                final_answer="still going",
            )

    orig, sess = _patch_user(_ScriptedUser(["Hi", "zip 19122", STOP_TOKEN]))
    try:
        session = TauOfficialSession(inner=_Slow())
        trace = asyncio.run(
            session.run(TaskInput(task_id="x", instruction="hidden", initial_state={}))
        )
    finally:
        sess.load_user = orig
    assert trace.status == "error"
    assert "run_deadline" in (trace.error or "")
