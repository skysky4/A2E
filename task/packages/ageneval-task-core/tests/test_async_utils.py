from __future__ import annotations

import asyncio
import threading
import time

import pytest
from ageneval.task.core.async_utils import run_sync_in_daemon_thread


def test_run_sync_in_daemon_thread_returns_value() -> None:
    async def scenario() -> tuple[int, bool]:
        def blocking() -> tuple[int, bool]:
            return 42, threading.current_thread().daemon

        return await run_sync_in_daemon_thread(blocking)

    assert asyncio.run(scenario()) == (42, True)


def test_run_sync_in_daemon_thread_propagates_exception() -> None:
    async def scenario() -> None:
        def blocking() -> None:
            raise ValueError("boom")

        await run_sync_in_daemon_thread(blocking)

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(scenario())


def test_cancelled_worker_does_not_delay_asyncio_shutdown() -> None:
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, bool] = {}

    def blocking() -> None:
        observed["daemon"] = threading.current_thread().daemon
        started.set()
        release.wait(10)

    async def scenario() -> None:
        task = asyncio.create_task(run_sync_in_daemon_thread(blocking))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    before = time.monotonic()
    try:
        asyncio.run(scenario())
        assert time.monotonic() - before < 1
        assert observed == {"daemon": True}
    finally:
        release.set()
