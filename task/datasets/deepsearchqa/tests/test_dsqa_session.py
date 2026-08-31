"""DeepSearchQA session: clean harness output only — no injected tools."""

from __future__ import annotations

import asyncio

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.binding import AgentBinding
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall
from ageneval.task.datasets.deepsearchqa.session import DeepSearchOfficialSession


def _binding() -> AgentBinding:
    def _exec(name, arguments, state):
        return {"results": [{"url": "https://www.nhs.uk/x", "title": "NHS"}]}

    return AgentBinding(
        name="deepsearchqa",
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        tool_executor=_exec,
        system_prompt_builder=lambda _tools: "",
    )


class _Passthrough(AgentRunner):
    name = "fake"

    def __init__(self, binding: AgentBinding, **trace_kw: object) -> None:
        self.binding = binding
        self.trace_kw = trace_kw

    async def run(self, task: TaskInput) -> TaskTrace:
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            **self.trace_kw,  # type: ignore[arg-type]
        )


def test_session_does_not_inject_tools_when_harness_skipped():
    binding = _binding()
    session = DeepSearchOfficialSession(
        inner=_Passthrough(binding, status="ok", turns=1, final_answer="Cervical spondylosis.")
    )
    task = TaskInput(task_id="deepsearchqa-v0001", instruction="NHS neck question")
    trace = asyncio.run(session.run(task))
    assert list(trace.tool_calls) == []
    assert trace.final_answer == "Cervical spondylosis."
    assert trace.status == "ok"


def test_session_drops_react_and_user_none_without_composing():
    binding = _binding()
    tools = (
        ToolCall(
            name="web_search",
            arguments={"query": "NHS"},
            result={"results": [{"url": "https://www.nhs.uk/x"}]},
        ),
    )
    for dirty in (
        'to=web_search code: {"query":"nhs"}',
        "user: None",
    ):
        session = DeepSearchOfficialSession(
            inner=_Passthrough(
                binding, status="ok", turns=1, tool_calls=tools, final_answer=dirty
            )
        )
        trace = asyncio.run(session.run(TaskInput(task_id="x", instruction="NHS neck")))
        assert [tc.name for tc in trace.tool_calls] == ["web_search"]
        assert not trace.final_answer
        assert trace.status == "error"


def test_session_keeps_harness_tools_and_clean_final():
    binding = _binding()
    session = DeepSearchOfficialSession(
        inner=_Passthrough(
            binding,
            status="ok",
            turns=2,
            tool_calls=(
                ToolCall(
                    name="web_search",
                    arguments={"query": "NHS"},
                    result={"results": [{"url": "https://www.nhs.uk/x"}]},
                ),
            ),
            final_answer='{"final_answer":"Hypermobility"}',
        )
    )
    trace = asyncio.run(
        session.run(TaskInput(task_id="deepsearchqa-v0001", instruction="NHS"))
    )
    assert [tc.name for tc in trace.tool_calls] == ["web_search"]
    assert trace.final_answer == "Hypermobility"
    assert trace.status == "ok"
