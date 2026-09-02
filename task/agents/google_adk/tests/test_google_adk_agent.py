from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from ageneval.task.agents.google_adk.agent import _build_function_tools
from ageneval.task.core import AgentBinding, TaskInput


def _binding(executor: Any) -> AgentBinding:
    schemas: Sequence[Mapping[str, Any]] = (
        {
            "type": "function",
            "function": {
                "name": "blocking_tool",
                "description": "A deliberately blocking test tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": "Value to return."}},
                    "required": ["value"],
                },
            },
        },
    )
    return AgentBinding(
        name="test",
        tool_schemas=schemas,
        tool_executor=executor,
        system_prompt_builder=lambda _schemas: "test",
    )


@pytest.mark.asyncio
async def test_google_adk_tool_does_not_block_event_loop() -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()

    def executor(_name: str, arguments: Mapping[str, Any], _state: Mapping[str, Any]) -> Any:
        worker_started.set()
        release_worker.wait(10)
        return {"value": arguments["value"]}

    task = TaskInput(task_id="async-tool", instruction="test")
    recorder = []
    tool = _build_function_tools(_binding(executor), task, recorder)[0]

    assert inspect.iscoroutinefunction(tool.func)
    assert list(inspect.signature(tool.func).parameters) == ["value"]

    started = time.monotonic()
    call = asyncio.create_task(tool.func(value="ok"))
    while not worker_started.is_set():
        await asyncio.sleep(0.001)
    await asyncio.sleep(0.01)
    assert time.monotonic() - started < 0.1
    assert not call.done()

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    release_worker.set()
