from __future__ import annotations

import asyncio

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall
from ageneval.task.datasets.deepsearchqa.session import DeepSearchOfficialSession


class _StaticAgent(AgentRunner):
    name = "fake"

    def __init__(self, **trace_values: object) -> None:
        self.trace_values = trace_values

    async def run(self, task: TaskInput) -> TaskTrace:
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            **self.trace_values,  # type: ignore[arg-type]
        )


def test_session_keeps_harness_tools_and_unwraps_final() -> None:
    session = DeepSearchOfficialSession(
        _StaticAgent(
            status="ok",
            turns=2,
            tool_calls=(
                ToolCall(
                    name="web_search",
                    arguments={"query": "NHS"},
                    result={"results": []},
                ),
            ),
            final_answer='{"final_answer":"Hypermobility"}',
        )
    )
    trace = asyncio.run(
        session.run(TaskInput(task_id="deepsearchqa-1", instruction="question"))
    )
    assert [call.name for call in trace.tool_calls] == ["web_search"]
    assert trace.final_answer == "Hypermobility"
    assert trace.status == "ok"


def test_session_drops_react_leak_without_injecting_tools() -> None:
    session = DeepSearchOfficialSession(
        _StaticAgent(
            status="ok",
            turns=1,
            final_answer='to=web_search code: {"query":"NHS"}',
        )
    )
    trace = asyncio.run(
        session.run(TaskInput(task_id="deepsearchqa-2", instruction="question"))
    )
    assert trace.tool_calls == ()
    assert trace.final_answer is None
    assert trace.status == "error"


def test_session_drops_user_none_placeholder() -> None:
    trace = asyncio.run(
        DeepSearchOfficialSession(
            _StaticAgent(status="ok", turns=1, final_answer="user: None")
        ).run(TaskInput(task_id="deepsearchqa-3", instruction="question"))
    )
    assert trace.final_answer is None
    assert trace.status == "error"
