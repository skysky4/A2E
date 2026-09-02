from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.grading import GraderSpec
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

    async def fake_daemon_thread(
        function: Callable[..., Any], *args: object, **kwargs: object
    ) -> Any:
        offloaded.append(function)
        kwargs.pop("thread_name", None)
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "ageneval.task.core.sandbox_runner.run_sync_in_daemon_thread",
        fake_daemon_thread,
    )

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


@pytest.mark.asyncio
async def test_inline_grader_embeds_normalized_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def fake_sandbox_session(_spec: object):
        yield _FakeSandbox()

    monkeypatch.setattr("ageneval.task.sandbox.sandbox_session", fake_sandbox_session)

    async def fake_daemon_thread(
        function: Callable[..., Any], *args: object, **kwargs: object
    ) -> Any:
        kwargs.pop("thread_name", None)
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "ageneval.task.core.sandbox_runner.run_sync_in_daemon_thread",
        fake_daemon_thread,
    )

    def score(_task: TaskInput, _sandbox: object, _patch: str) -> dict[str, object]:
        return {"score": 1.0, "resolved": True}

    runner = SandboxScoringRunner(
        inner=_ImmediateAgent(),
        grader=GraderSpec(
            id="sandbox_score",
            grade=score,
            mode="inline",
            source="official harness",
        ),
    )
    result = await runner.run(
        TaskInput(
            task_id="inline-score",
            instruction="test",
            sandbox={"type": "local"},
        )
    )
    assert result.raw["resolved"] is True
    assert result.raw["grade_report"]["score"] == 1.0
    assert result.raw["grade_report"]["metadata"]["source"] == "official harness"


def test_requires_exactly_one_scoring_path() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SandboxScoringRunner(inner=_ImmediateAgent())
