#!/usr/bin/env python3
"""Private synchronous worker used by the Trial process concurrency probe."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone


def send(message: dict) -> None:
    os.write(
        sys.stdout.fileno(),
        json.dumps(message, separators=(",", ":")).encode() + b"\n",
    )


def lifecycle(event: str, attempt: int) -> None:
    request_id = uuid.uuid4().hex
    send(
        {
            "type": "lifecycle_request",
            "request_id": request_id,
            "event": event,
            "attempt": attempt,
        }
    )
    response = json.loads(sys.stdin.buffer.readline())
    if response.get("request_id") != request_id or not response.get("ok"):
        raise RuntimeError(response.get("error") or "probe lifecycle rejected")


def write_result(result: dict) -> None:
    path = os.environ["A2E_TRIAL_RESULT_PATH"]
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".process-result.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with open(path, "rb") as stream:
        raw = stream.read()
    send(
        {
            "type": "result_ready",
            "path": os.path.basename(path),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def main() -> int:
    payload = json.loads(sys.stdin.buffer.readline())
    attempt = int(payload["attempt"])
    lifecycle("AGENT_START", attempt)
    send({"type": "activity", "name": "probe:blocking", "state": "start"})
    deadline = time.monotonic() + float(payload["probe_timeout"])
    release = payload["release_path"]
    while not os.path.exists(release):
        if time.monotonic() >= deadline:
            raise TimeoutError("probe barrier was not released")
        time.sleep(0.005)
    time.sleep(float(payload["blocking_seconds"]))
    send({"type": "activity", "name": "probe:blocking", "state": "end"})
    lifecycle("AGENT_END", attempt)
    now = datetime.now(timezone.utc).isoformat()
    trial = payload["trial"]
    write_result(
        {
            "trial_id": trial["trial_id"],
            "cell_id": payload["cell"]["cell_id"],
            "task_id": trial["task_id"],
            "repetition": trial["repetition"],
            "attempt": attempt,
            "status": "local_complete",
            "output": {"blob": "x" * int(payload.get("result_blob_bytes", 0))},
            "started_at": now,
            "ended_at": now,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
