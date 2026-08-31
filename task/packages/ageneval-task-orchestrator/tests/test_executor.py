from __future__ import annotations

import asyncio
import json
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.core import TaskTrace
from ageneval.task.orchestrator.executor import (
    _execute_attempt,
    _execute_isolated_attempt,
    _run_cell_async,
)
from ageneval.task.orchestrator.schema import LifecycleEvent, TrialResult
from opentelemetry.sdk.trace import TracerProvider


def test_trial_executor_runs_agent_grader_and_releases_all_permits(monkeypatch) -> None:
    import ageneval.task.runners as runners
    from ageneval.task.core import GraderSpec
    from ageneval.task.runners import AGENTS, DATASETS

    class Agent:
        name = "fake"

        async def run(self, task):
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok",
                turns=1,
                final_answer="A",
            )

    monkeypatch.setitem(
        AGENTS,
        "fake-harness",
        {"build": lambda **_kwargs: Agent(), "framework": "none"},
    )
    monkeypatch.setitem(
        DATASETS,
        "fake-benchmark",
        {"bind": lambda **_kwargs: object(), "kind": "qa"},
    )
    monkeypatch.setattr(
        runners,
        "grader_for_dataset",
        lambda _dataset: GraderSpec(
            id="exact_match",
            grade=lambda output, expected: float(
                output.get("final_answer")
                == (expected.get("expected_outputs") or [None])[0]
            ),
        ),
    )
    profile = ModelProfile.model_validate(
        {
            "id": "fake-model",
            "provider": "fake",
            "model": "fake-model",
            "protocol": "openai_chat_completions",
            "connection": {"api_key_env": "FAKE_KEY"},
            "concurrency": {"group": "fake", "max_sessions": 1},
        }
    )
    resolved = ResolvedModel(profile=profile, api_key="secret")
    events = []

    async def lifecycle(event, attempt):
        events.append((event, attempt))

    async def scenario():
        return await _execute_attempt(
            cell={
                "campaign_id": "campaign-1",
                "cell_id": "cell-1",
                "benchmark": "fake-benchmark",
                "harness": "fake-harness",
                "retry": {"max_retries": 0},
            },
            trial={
                "trial_id": "trial-1",
                "task_id": "task-1",
                "repetition": 1,
            },
            task_payload={
                "task_id": "task-1",
                "instruction": "answer",
                "expected_outputs": ["A"],
            },
            benchmark={
                "graders": [
                    {"id": "exact_match", "mode": "posthoc", "required": True}
                ]
            },
            resolved_model=resolved,
            provider=TracerProvider(),
            lifecycle=lifecycle,
            attempt=1,
            timeout_seconds=5,
        )

    result = asyncio.run(scenario())
    assert result.status == "local_complete"
    assert result.grades[0].score == 1.0
    assert events == [
        (LifecycleEvent.AGENT_START, 1),
        (LifecycleEvent.AGENT_END, 1),
        (LifecycleEvent.VERIFICATION_START, 1),
        (LifecycleEvent.VERIFICATION_END, 1),
    ]


def test_isolated_executor_uses_ipc_without_persisting_secret(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json, sys
from datetime import datetime, timezone
payload = json.load(sys.stdin)
now = datetime.now(timezone.utc).isoformat()
print("framework diagnostic")
print(json.dumps({
    "trial_id": payload["trial"]["trial_id"],
    "cell_id": payload["cell"]["cell_id"],
    "task_id": payload["trial"]["task_id"],
    "repetition": payload["trial"]["repetition"],
    "attempt": payload["attempt"],
    "status": "local_complete",
    "started_at": now,
    "ended_at": now
}))
""",
        encoding="utf-8",
    )
    profile = ModelProfile.model_validate(
        {
            "id": "isolated",
            "provider": "test",
            "model": "isolated",
            "protocol": "openai_chat_completions",
            "connection": {"api_key_env": "ISOLATED_KEY"},
            "concurrency": {"group": "isolated", "max_sessions": 1},
        }
    )
    resolved = ResolvedModel(profile=profile, api_key="do-not-write-me")
    events = []

    async def lifecycle(event, attempt):
        events.append((event, attempt))
    payload = {
        "cell": {
            "campaign_id": "campaign-1",
            "cell_id": "cell-1",
            "retry": {"max_retries": 0},
        },
        "benchmark": {"graders": []},
        "project_name": "project-1",
        "isolated_python": sys.executable,
        "isolated_script": str(worker),
        "isolated_pythonpath": "",
        "timeout_seconds": 5,
    }

    result = asyncio.run(
        _execute_isolated_attempt(
            payload=payload,
            trial={"trial_id": "trial-1", "task_id": "task-1", "repetition": 1},
            task_payload={"task_id": "task-1", "instruction": "answer"},
            resolved_model=resolved,
            lifecycle=lifecycle,
            attempt=1,
        )
    )
    assert result.status == "local_complete"
    assert events == [
        (LifecycleEvent.AGENT_START, 1),
        (LifecycleEvent.AGENT_END, 1),
    ]
    assert "do-not-write-me" not in json.dumps(payload)


def test_cell_worker_uses_one_start_end_around_retry_loop(monkeypatch) -> None:
    profile = ModelProfile.model_validate(
        {
            "id": "worker-model",
            "provider": "test",
            "model": "worker-model",
            "protocol": "openai_chat_completions",
            "connection": {"api_key_env": "WORKER_KEY"},
            "concurrency": {"group": "worker", "max_sessions": 1},
        }
    )
    resolved = ResolvedModel(profile=profile, api_key="secret")
    attempts = []

    async def fake_attempt(*, trial, lifecycle, attempt, payload, **_kwargs):
        attempts.append(attempt)
        await lifecycle(LifecycleEvent.AGENT_START, attempt)
        await lifecycle(LifecycleEvent.AGENT_END, attempt)
        now = datetime.now(timezone.utc)
        return TrialResult(
            trial_id=trial["trial_id"],
            cell_id=payload["cell"]["cell_id"],
            task_id=trial["task_id"],
            repetition=trial["repetition"],
            attempt=attempt,
            status="failed" if attempt == 1 else "local_complete",
            error="temporary" if attempt == 1 else None,
            error_type="ConnectionError" if attempt == 1 else None,
            retryable=attempt == 1,
            started_at=now,
            ended_at=now,
        )

    monkeypatch.setattr(
        "ageneval.task.orchestrator.executor._execute_isolated_attempt", fake_attempt
    )
    commands: queue.Queue = queue.Queue()
    events: queue.Queue = queue.Queue()
    payload = {
        "cell": {
            "campaign_id": "campaign-1",
            "cell_id": "cell-1",
            "retry": {
                "max_retries": 1,
                "min_wait_seconds": 0,
                "max_wait_seconds": 0,
            },
        },
        "resolved_model": resolved.model_dump(mode="python"),
        "project_name": "project-1",
        "benchmark": {"graders": []},
        "tasks_by_id": {"task-1": {"task_id": "task-1", "instruction": "x"}},
        "isolated": True,
    }

    async def scenario():
        worker = asyncio.create_task(_run_cell_async(payload, commands, events))
        lifecycle = []
        results = []

        async def next_event():
            async with asyncio.timeout(2):
                while True:
                    try:
                        return events.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.001)

        ready = await next_event()
        assert ready["type"] == "worker_ready"
        commands.put(
            {
                "type": "run_trial",
                "trial": {
                    "trial_id": "trial-1",
                    "task_id": "task-1",
                    "repetition": 1,
                    "attempt": 1,
                },
            }
        )
        while True:
            message = await next_event()
            if message["type"] == "lifecycle":
                lifecycle.append(message["event"])
                commands.put(
                    {
                        "type": "lifecycle_response",
                        "request_id": message["request_id"],
                        "ok": True,
                    }
                )
            elif message["type"] == "trial_result":
                results.append(message)
                if message["final"]:
                    commands.put({"type": "shutdown"})
                    break
        await asyncio.wait_for(worker, timeout=2)
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert not pending, pending
        return lifecycle, results

    lifecycle, results = asyncio.run(scenario())
    assert attempts == [1, 2]
    assert lifecycle == [
        "START",
        "AGENT_START",
        "AGENT_END",
        "AGENT_START",
        "AGENT_END",
        "END",
    ]
    assert [message["final"] for message in results] == [False, True]
