from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.orchestrator.concurrency import PermitBroker, RuntimeMetrics
from ageneval.task.orchestrator.process import TrialProcessError, TrialProcessRunner
from ageneval.task.orchestrator.schema import LifecycleEvent


def _model(limit: int) -> ResolvedModel:
    return ResolvedModel(
        profile=ModelProfile.model_validate(
            {
                "id": "process-test",
                "provider": "test",
                "model": "process-test",
                "upstream_protocol": "openai_chat_completions",
                "connection": {"api_key_env": "PROCESS_TEST_KEY"},
                "concurrency": {"group": "test", "max_sessions": limit},
            }
        ),
        api_key="never-serialized",
    )


def test_synchronous_work_really_overlaps_in_trial_processes(tmp_path: Path) -> None:
    concurrency = 8
    worker = Path(__file__).resolve().parents[3] / "examples/run_trial_process_probe.py"

    async def scenario() -> tuple[dict, dict]:
        release = tmp_path / "release"
        broker = PermitBroker({"model:test": concurrency})
        runtime = RuntimeMetrics()

        async def run_one(index: int) -> None:
            owner = f"owner:{index}"

            async def lifecycle(event: LifecycleEvent, _attempt: int | None) -> None:
                if event == LifecycleEvent.AGENT_START:
                    await broker.acquire(owner, "model:test")
                elif event == LifecycleEvent.AGENT_END:
                    broker.release(owner, "model:test")

            def activity(name: str, state: str) -> None:
                runtime.activity(owner, name, state)
                if (
                    runtime.snapshot()["activities"]["probe:blocking"]["active"]
                    == concurrency
                ):
                    release.touch(exist_ok=True)

            def process_event(state: str, pid: int) -> None:
                runtime.process_event(state, pid)
                if state != "start":
                    runtime.release_activities(owner)

            runner = TrialProcessRunner(
                script=worker,
                cancellation_grace_seconds=1,
                lifecycle=lifecycle,
                activity=activity,
                process_event=process_event,
            )
            result = await runner.run(
                payload={
                    "cell": {"campaign_id": "test", "cell_id": "cell"},
                    "trial": {
                        "trial_id": f"trial-{index}",
                        "task_id": f"task-{index}",
                        "repetition": 1,
                    },
                    "attempt": 1,
                    "release_path": str(release),
                    "blocking_seconds": 0.1,
                    "probe_timeout": 10,
                },
                resolved_model=_model(concurrency),
                python=sys.executable,
                pythonpath="",
                stderr_path=tmp_path / f"{index}.log",
                result_path=tmp_path / f"{index}.result.json",
                timeout_seconds=15,
            )
            assert result.status == "local_complete"

        await asyncio.gather(*(run_one(index) for index in range(concurrency)))
        return broker.snapshot(), runtime.snapshot()

    permits, metrics = asyncio.run(scenario())
    assert permits["model:test"] == {
        "limit": concurrency,
        "active": 0,
        "high_water": concurrency,
    }
    assert metrics["trial_processes"]["active"] == 0
    assert metrics["trial_processes"]["high_water"] == concurrency
    assert metrics["activities"]["probe:blocking"]["active"] == 0
    assert metrics["activities"]["probe:blocking"]["high_water"] == concurrency


def test_cancelling_trial_terminates_process_and_clears_activity(tmp_path: Path) -> None:
    worker = Path(__file__).resolve().parents[3] / "examples/run_trial_process_probe.py"

    async def scenario() -> dict:
        runtime = RuntimeMetrics()
        entered = asyncio.Event()
        owner = "cancel-owner"

        async def lifecycle(_event: LifecycleEvent, _attempt: int | None) -> None:
            return None

        def activity(name: str, state: str) -> None:
            runtime.activity(owner, name, state)
            if state == "start":
                entered.set()

        def process_event(state: str, pid: int) -> None:
            runtime.process_event(state, pid)
            if state != "start":
                runtime.release_activities(owner)

        runner = TrialProcessRunner(
            script=worker,
            cancellation_grace_seconds=0.2,
            lifecycle=lifecycle,
            activity=activity,
            process_event=process_event,
        )
        task = asyncio.create_task(
            runner.run(
                payload={
                    "cell": {"campaign_id": "test", "cell_id": "cell"},
                    "trial": {
                        "trial_id": "cancelled-trial",
                        "task_id": "cancelled-task",
                        "repetition": 1,
                    },
                    "attempt": 1,
                    "release_path": str(tmp_path / "never-release"),
                    "blocking_seconds": 30,
                    "probe_timeout": 30,
                },
                resolved_model=_model(1),
                python=sys.executable,
                pythonpath="",
                stderr_path=tmp_path / "cancel.log",
                result_path=tmp_path / "cancel.result.json",
                timeout_seconds=None,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return runtime.snapshot()

    metrics = asyncio.run(scenario())
    assert metrics["trial_processes"]["active"] == 0
    assert metrics["activities"]["probe:blocking"]["active"] == 0


def test_real_trial_worker_uses_duplex_protocol_without_leaking_secret(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[3] / "examples/run_isolated_trial.py"
    resolved = _model(1)

    async def scenario():
        runner = TrialProcessRunner(
            script=script,
            cancellation_grace_seconds=1,
            lifecycle=lambda _event, _attempt: asyncio.sleep(0),
        )
        return await runner.run(
            payload={
                "cell": {
                    "campaign_id": "protocol-test",
                    "cell_id": "cell",
                    "benchmark": "intentionally-missing",
                    "harness": "langgraph",
                    "retry": {"max_retries": 0},
                },
                "trial": {
                    "trial_id": "protocol-trial",
                    "task_id": "protocol-task",
                    "repetition": 1,
                },
                "task": {
                    "task_id": "protocol-task",
                    "instruction": "none",
                    "metadata": {},
                },
                "benchmark": {"graders": []},
                "profile": resolved.profile.public_dict(),
                "base_url": None,
                "project_name": "protocol-test",
                "otel_endpoint": None,
                "attempt": 1,
                "timeout_seconds": 5,
            },
            resolved_model=resolved,
            python=sys.executable,
            pythonpath="",
            stderr_path=tmp_path / "stderr.log",
            result_path=tmp_path / "process-result.json",
            timeout_seconds=10,
        )

    result = asyncio.run(scenario())
    assert result.status == "failed"
    assert result.error_type == "KeyError"
    assert "never-serialized" not in (tmp_path / "stderr.log").read_text()


def test_result_larger_than_jsonl_limit_uses_atomic_file(tmp_path: Path) -> None:
    worker = Path(__file__).resolve().parents[3] / "examples/run_trial_process_probe.py"
    release = tmp_path / "release"
    release.touch()
    result_path = tmp_path / "large-process-result.json"

    async def scenario():
        runner = TrialProcessRunner(
            script=worker,
            cancellation_grace_seconds=1,
            lifecycle=lambda _event, _attempt: asyncio.sleep(0),
        )
        return await runner.run(
            payload={
                "cell": {"campaign_id": "test", "cell_id": "cell"},
                "trial": {
                    "trial_id": "large-trial",
                    "task_id": "large-task",
                    "repetition": 1,
                },
                "attempt": 1,
                "release_path": str(release),
                "blocking_seconds": 0,
                "probe_timeout": 2,
                "result_blob_bytes": 2_000_000,
            },
            resolved_model=_model(1),
            python=sys.executable,
            pythonpath="",
            stderr_path=tmp_path / "large.log",
            result_path=result_path,
            timeout_seconds=10,
        )

    result = asyncio.run(scenario())
    assert len(result.output["blob"]) == 2_000_000
    assert result_path.stat().st_size > 2_000_000


def test_result_ready_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    worker = tmp_path / "bad_digest_worker.py"
    worker.write_text(
        """
import json, os, sys
json.loads(sys.stdin.buffer.readline())
path = os.environ['A2E_TRIAL_RESULT_PATH']
with open(path, 'w') as stream:
    stream.write('{}')
message = {'type': 'result_ready', 'path': os.path.basename(path),
           'size_bytes': 2, 'sha256': '0' * 64}
sys.stdout.write(json.dumps(message) + '\\n')
sys.stdout.flush()
""".lstrip(),
        encoding="utf-8",
    )

    async def scenario():
        runner = TrialProcessRunner(
            script=worker,
            cancellation_grace_seconds=1,
            lifecycle=lambda _event, _attempt: asyncio.sleep(0),
        )
        return await runner.run(
            payload={
                "cell": {"campaign_id": "test", "cell_id": "cell"},
                "trial": {
                    "trial_id": "bad-trial",
                    "task_id": "bad-task",
                    "repetition": 1,
                },
                "attempt": 1,
            },
            resolved_model=_model(1),
            python=sys.executable,
            pythonpath="",
            stderr_path=tmp_path / "bad.log",
            result_path=tmp_path / "bad-result.json",
            timeout_seconds=10,
        )

    with pytest.raises(TrialProcessError, match="digest"):
        asyncio.run(scenario())
