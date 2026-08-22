"""Small asyncio helpers shared by framework adapters."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_sync_in_daemon_thread(
    func: Callable[..., T],
    /,
    *args: Any,
    thread_name: str | None = None,
    **kwargs: Any,
) -> T:
    """Run blocking code without registering it in asyncio's default executor.

    ``asyncio.to_thread`` uses the event loop's default ``ThreadPoolExecutor``.
    Cancelling the awaiting task does not stop that worker, and ``asyncio.run``
    waits for default-executor workers during shutdown. Some agent SDK calls can
    therefore keep an otherwise completed experiment alive indefinitely.

    Frameworks without a native async API are instead isolated in a daemon
    thread. Cancellation still cannot interrupt third-party synchronous code,
    but the abandoned worker no longer blocks loop or interpreter shutdown.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def set_result(value: T) -> None:
        if not future.done():
            future.set_result(value)

    def set_exception(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def worker() -> None:
        try:
            value = func(*args, **kwargs)
        except BaseException as exc:
            try:
                loop.call_soon_threadsafe(set_exception, exc)
            except RuntimeError:
                # The caller was cancelled and its event loop has already
                # closed. There is no remaining consumer for this outcome.
                pass
        else:
            try:
                loop.call_soon_threadsafe(set_result, value)
            except RuntimeError:
                pass

    thread = threading.Thread(
        target=worker,
        name=thread_name or f"a2e-{getattr(func, '__name__', 'blocking-call')}",
        daemon=True,
    )
    thread.start()
    return await future
