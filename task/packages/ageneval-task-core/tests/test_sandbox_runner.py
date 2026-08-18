from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace
from ageneval.task.core.sandbox_runner import SandboxScoringRunner


class _ImmediateAgent(AgentRunner):
    name = "immediate"

    async def run(self, task: TaskInput) -> TaskTrace:
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status="success",
            turns=0,
        )


class _FakeSandbox:
    def exec(self, _cmd: list[str]) -> SimpleNamespace:
        return SimpleNamespace(stdout="")


@pytest.mark.asyncio
async def test_scorer_is_offloaded_from_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def fake_sandbox_session(_spec: object):
        yield _FakeSandbox()

    monkeypatch.setattr("ageneval.task.sandbox.sandbox_session", fake_sandbox_session)

    offloaded: list[object] = []

    async def fake_to_thread(
        function: Callable[..., Any], *args: object, **kwargs: object
    ) -> Any:
        offloaded.append(function)
        return function(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    def score(_task: TaskInput, _sandbox: object, _patch: str) -> dict[str, bool]:
        return {"resolved": True}

    runner = SandboxScoringRunner(inner=_ImmediateAgent(), score_fn=score)
    task = TaskInput(
        task_id="slow-score",
        instruction="test",
        sandbox={"type": "local"},
    )

    result = await runner.run(task)
    assert offloaded == [score]
    assert result.raw["resolved"] is True
