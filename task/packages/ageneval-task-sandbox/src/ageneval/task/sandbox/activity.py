"""Process-local sandbox activity reporting.

Campaign Trials run in dedicated processes, so a single process-wide callback
is sufficient and, unlike a ContextVar, also covers SDK-created tool threads.
The sandbox package stays independent of the orchestrator by exposing only a
small callback API.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

ActivityCallback = Callable[[str, str], None]

_callback: ActivityCallback | None = None
_lock = threading.Lock()


def set_activity_callback(callback: ActivityCallback | None) -> None:
    global _callback
    with _lock:
        _callback = callback


def emit_activity(kind: str, state: str) -> None:
    with _lock:
        callback = _callback
    if callback is not None:
        callback(kind, state)


__all__ = ["emit_activity", "set_activity_callback"]
