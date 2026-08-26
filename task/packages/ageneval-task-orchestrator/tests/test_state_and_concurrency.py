from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timezone

import pytest
from ageneval.task.orchestrator.concurrency import PermitBroker, PermitLedger, PermitPool
from ageneval.task.orchestrator.schema import TrialResult
from ageneval.task.orchestrator.state import RunDirectory, atomic_write_json, read_json


def test_run_directory_is_atomic_and_resume_config_is_immutable(tmp_path) -> None:
    run = RunDirectory(tmp_path / "run")
    run.initialize(config={"name": "one"}, lock={"campaign_id": "campaign-1"})
    assert run.verify_config({"name": "one"})["campaign_id"] == "campaign-1"
    with pytest.raises(ValueError, match="does not match"):
        run.verify_config({"name": "changed"})

    now = datetime.now(timezone.utc)
    result = TrialResult(
        trial_id="trial-1",
        cell_id="cell-1",
        task_id="task-1",
        repetition=1,
        attempt=1,
        status="completed",
        uploaded=True,
        started_at=now,
        ended_at=now,
    )
    run.write_trial_result(result)
    assert run.load_trial_result("trial-1") == result
    assert read_json(run.trial_dir("trial-1") / "attempts" / "1" / "result.json")


def test_corrupt_trial_result_is_quarantined_for_rerun(tmp_path) -> None:
    run = RunDirectory(tmp_path / "run")
    path = run.trial_dir("trial-1") / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="quarantined"):
        assert run.load_trial_result("trial-1") is None
    assert not path.exists()
    assert len(list(path.parent.glob("result.corrupt-*.json"))) == 1


def test_retry_attempt_does_not_become_terminal_result(tmp_path) -> None:
    run = RunDirectory(tmp_path / "run")
    now = datetime.now(timezone.utc)
    attempt = TrialResult(
        trial_id="trial-1",
        cell_id="cell-1",
        task_id="task-1",
        repetition=1,
        attempt=1,
        status="failed",
        retryable=True,
        started_at=now,
        ended_at=now,
    )
    run.write_trial_attempt(attempt)
    assert run.load_trial_result("trial-1") is None
    assert read_json(run.trial_dir("trial-1") / "attempts/1/result.json")


def test_single_controller_lock(tmp_path) -> None:
    run = RunDirectory(tmp_path / "run")
    with run.controller_lock():
        with pytest.raises(RuntimeError, match="already has a controller"):
            with run.controller_lock():
                pass
    with run.controller_lock():
        pass


def test_stale_controller_lock_is_recovered(tmp_path) -> None:
    run = RunDirectory(tmp_path / "run")
    run.root.mkdir(parents=True)
    atomic_write_json(
        run.controller_lock_path,
        {"pid": 2_000_000_000, "host": socket.gethostname()},
    )
    with run.controller_lock():
        assert run.controller_lock_path.exists()
    assert not run.controller_lock_path.exists()


def test_permit_ledger_releases_on_cancellation() -> None:
    async def scenario() -> None:
        pool = PermitPool(1)
        entered = asyncio.Event()

        async def holder() -> None:
            ledger = PermitLedger()
            try:
                await ledger.acquire(pool)
                entered.set()
                await asyncio.Event().wait()
            finally:
                ledger.release_all()

        task = asyncio.create_task(holder())
        await entered.wait()
        assert pool.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.active == 0

    asyncio.run(scenario())


def test_permit_pool_never_exceeds_limit() -> None:
    async def scenario() -> None:
        pool = PermitPool(3)

        async def work() -> None:
            async with pool.permit():
                await asyncio.sleep(0.01)

        await asyncio.gather(*(work() for _ in range(20)))
        assert pool.high_water == 3
        assert pool.active == 0

    asyncio.run(scenario())


def test_controller_permit_broker_releases_crashed_cell_atomically() -> None:
    async def scenario() -> None:
        broker = PermitBroker({"global": 2, "model:shared": 1})
        await broker.acquire("cell-a:trial-1", "global")
        await broker.acquire("cell-a:trial-1", "model:shared")
        await broker.acquire("cell-b:trial-2", "global")

        broker.release_prefix("cell-a:")

        assert broker.held("cell-a:trial-1") == ()
        assert broker.held("cell-b:trial-2") == ("global",)
        snapshot = broker.snapshot()
        assert snapshot["global"]["active"] == 1
        assert snapshot["model:shared"]["active"] == 0
        broker.release_all("cell-b:trial-2")

    asyncio.run(scenario())


def test_controller_permit_broker_enforces_shared_model_limit() -> None:
    async def scenario() -> None:
        broker = PermitBroker({"model:shared": 2})
        active = 0
        high_water = 0
        lock = asyncio.Lock()

        async def work(index: int) -> None:
            nonlocal active, high_water
            owner = f"cell-{index}:trial-{index}"
            await broker.acquire(owner, "model:shared")
            async with lock:
                active += 1
                high_water = max(high_water, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            broker.release(owner, "model:shared")

        await asyncio.gather(*(work(index) for index in range(12)))
        assert high_water == 2
        assert broker.snapshot()["model:shared"]["active"] == 0

    asyncio.run(scenario())


def test_independent_model_pool_can_use_the_entire_global_model_limit() -> None:
    async def scenario() -> None:
        broker = PermitBroker(
            {
                "model:glm": 32,
                "model:gpt": 32,
                "model:total": 32,
            }
        )
        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def work(index: int) -> None:
            nonlocal entered
            owner = f"cell-glm:trial-{index}"
            await broker.acquire_many(owner, ("model:glm", "model:total"))
            entered += 1
            if entered == 32:
                all_entered.set()
            await release.wait()
            broker.release_many(owner, ("model:total", "model:glm"))

        tasks = [asyncio.create_task(work(index)) for index in range(32)]
        await asyncio.wait_for(all_entered.wait(), timeout=1)
        snapshot = broker.snapshot()
        assert snapshot["model:glm"]["active"] == 32
        assert snapshot["model:gpt"]["active"] == 0
        assert snapshot["model:total"]["active"] == 32
        assert snapshot["model:total"]["high_water"] == 32
        release.set()
        await asyncio.gather(*tasks)
        assert all(item["active"] == 0 for item in broker.snapshot().values())

    asyncio.run(scenario())


def test_acquire_many_rolls_back_partial_batch_when_cancelled() -> None:
    async def scenario() -> None:
        broker = PermitBroker({"model:gpt": 1, "model:total": 1})
        await broker.acquire("holder", "model:total")

        blocked = asyncio.create_task(
            broker.acquire_many("candidate", ("model:gpt", "model:total"))
        )
        await asyncio.sleep(0)
        assert broker.snapshot()["model:gpt"]["active"] == 1
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        assert broker.held("candidate") == ()
        assert broker.snapshot()["model:gpt"]["active"] == 0
        broker.release_all("holder")

    asyncio.run(scenario())
