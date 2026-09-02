from __future__ import annotations

import asyncio
import multiprocessing
import queue
import sys
from pathlib import Path

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.orchestrator.concurrency import PermitBroker
from ageneval.task.orchestrator.executor import run_cell_worker
from ageneval.task.orchestrator.schema import LifecycleEvent, TrialResult


def _resolved_model() -> ResolvedModel:
    profile = ModelProfile.model_validate(
        {
            "id": "ipc-model",
            "provider": "test",
            "model": "ipc-model",
            "protocol": "openai_chat_completions",
            "connection": {"api_key_env": "IPC_KEY"},
            "concurrency": {"group": "ipc", "max_sessions": 1},
        }
    )
    return ResolvedModel(profile=profile, api_key="secret")


def test_spawned_worker_cancels_cleanly_and_releases_controller_permits(
    tmp_path: Path,
) -> None:
    child = tmp_path / "slow_child.py"
    child.write_text(
        "import time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    context = multiprocessing.get_context("spawn")
    commands = context.Queue()
    events = context.Queue()
    payload = {
        "cell": {
            "campaign_id": "campaign-1",
            "cell_id": "cell-1",
            "retry": {"max_retries": 0},
        },
        "resolved_model": _resolved_model().model_dump(mode="python"),
        "project_name": "project-1",
        "benchmark": {"graders": []},
        "tasks_by_id": {"task-1": {"task_id": "task-1", "instruction": "wait"}},
        "isolated": True,
        "isolated_python": sys.executable,
        "isolated_script": str(child),
        "isolated_pythonpath": "",
        "timeout_seconds": 30,
    }
    process = context.Process(target=run_cell_worker, args=(payload, commands, events))
    process.start()

    async def scenario() -> tuple[list[str], TrialResult]:
        broker = PermitBroker({"global": 1, "model:ipc": 1})
        lifecycle: list[str] = []
        final: TrialResult | None = None
        cancelled = False
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

        async def next_message() -> dict:
            async with asyncio.timeout(5):
                while True:
                    try:
                        return events.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)

        while True:
            message = await next_message()
            kind = message["type"]
            if kind == "worker_ready":
                continue
            if kind == "lifecycle":
                event = LifecycleEvent(message["event"])
                lifecycle.append(event.value)
                owner = f"{message['cell_id']}:{message['trial_id']}"
                if event == LifecycleEvent.START:
                    await broker.acquire(owner, "global")
                elif event == LifecycleEvent.AGENT_START:
                    await broker.acquire(owner, "model:ipc")
                elif event == LifecycleEvent.AGENT_END:
                    broker.release(owner, "model:ipc")
                elif event in {LifecycleEvent.CANCEL, LifecycleEvent.END}:
                    broker.release_all(owner)
                commands.put(
                    {
                        "type": "lifecycle_response",
                        "request_id": message["request_id"],
                        "ok": True,
                    }
                )
                if event == LifecycleEvent.AGENT_START and not cancelled:
                    cancelled = True
                    commands.put({"type": "cancel"})
            elif kind == "trial_result" and message["final"]:
                final = TrialResult.model_validate(message["result"])
            elif kind == "worker_done":
                break
        assert final is not None
        assert all(item["active"] == 0 for item in broker.snapshot().values())
        return lifecycle, final

    lifecycle, final = asyncio.run(scenario())
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    commands.close()
    events.close()

    assert process.exitcode == 0
    assert final.status == "cancelled"
    # Cancellation can land immediately after AGENT_START is granted, before
    # the attempt records AGENT_END. CANCEL/END are therefore the mandatory
    # Harbor-style backstop that releases the model lease.
    assert lifecycle == ["START", "AGENT_START", "CANCEL", "END"]
