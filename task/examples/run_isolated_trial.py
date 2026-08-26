#!/usr/bin/env python3
"""Private duplex JSON-lines worker for one Campaign Trial attempt."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.core import setup_instrumentation
from ageneval.task.orchestrator.executor import _execute_attempt
from ageneval.task.orchestrator.schema import LifecycleEvent, TrialResult
from ageneval.task.orchestrator.state import atomic_write_json
from ageneval.task.runners import framework_for_agent
from ageneval.task.sandbox import set_activity_callback


def _read_initial_message(fd: int) -> tuple[bytes, bytes]:
    """Read one newline-delimited message without buffered-IO shutdown locks."""
    buffered = b""
    while b"\n" not in buffered:
        chunk = os.read(fd, 65_536)
        if not chunk:
            return buffered, b""
        buffered += chunk
    line, remaining = buffered.split(b"\n", 1)
    return line, remaining


class Protocol:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        output_fd: int,
        initial_input: bytes = b"",
    ) -> None:
        self.loop = loop
        self.output_fd = output_fd
        self._initial_input = initial_input
        self._write_lock = threading.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._closing = threading.Event()
        self._reader = threading.Thread(
            target=self._read_responses,
            name="a2e-trial-parent-monitor",
            daemon=True,
        )

    def start(self) -> None:
        self._reader.start()

    def close(self) -> None:
        self._closing.set()

    def send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        with self._write_lock:
            os.write(self.output_fd, data)

    async def lifecycle(self, event: LifecycleEvent, attempt: int | None) -> None:
        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = self.loop.create_future()
        self._pending[request_id] = future
        self.send(
            {
                "type": "lifecycle_request",
                "request_id": request_id,
                "event": event.value,
                "attempt": attempt,
            }
        )
        try:
            response = await future
        finally:
            self._pending.pop(request_id, None)
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "lifecycle request was rejected")

    def activity(self, name: str, state: str) -> None:
        self.send({"type": "activity", "name": name, "state": state})

    def _read_responses(self) -> None:
        buffered = self._initial_input
        while True:
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                if not line:
                    continue
                if not self._handle_response_line(line):
                    return
            chunk = os.read(sys.stdin.fileno(), 65_536)
            if not chunk:
                if not self._closing.is_set():
                    # Parent disappeared. SIGTERM is handled in the main thread
                    # and unwinds sandbox/provider finally blocks.
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            buffered += chunk

    def _handle_response_line(self, line: bytes) -> bool:
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            os.kill(os.getpid(), signal.SIGTERM)
            return False
        if not isinstance(message, dict) or message.get("type") != "lifecycle_response":
            return True
        request_id = message.get("request_id")
        future = self._pending.get(request_id)
        if future is not None:
            self.loop.call_soon_threadsafe(self._resolve, future, message)
        return True

    @staticmethod
    def _resolve(
        future: asyncio.Future[dict[str, Any]], message: dict[str, Any]
    ) -> None:
        if not future.done():
            future.set_result(message)


def _handle_sigterm(_signum: int, _frame: object) -> None:
    raise SystemExit(143)


async def _run(payload: dict[str, Any], protocol: Protocol) -> TrialResult:
    from opentelemetry.context import attach, detach
    from opentelemetry.propagate import extract

    profile = ModelProfile.model_validate(payload["profile"])
    api_key = os.environ.get("A2E_TRIAL_MODEL_API_KEY")
    if not api_key:
        raise ValueError("A2E_TRIAL_MODEL_API_KEY is missing")
    resolved = ResolvedModel(
        profile=profile,
        base_url=payload.get("base_url"),
        api_key=api_key,
    )
    provider = setup_instrumentation(
        project_name=payload["project_name"],
        endpoint=payload.get("otel_endpoint"),
        framework=framework_for_agent(payload["cell"]["harness"]),
        batch=True,
        extra_resource_attributes=(
            ("a2e.campaign_id", payload["cell"]["campaign_id"]),
            ("a2e.cell_id", payload["cell"]["cell_id"]),
            ("a2e.trial_id", payload["trial"]["trial_id"]),
            ("a2e.attempt", payload["attempt"]),
        ),
    )
    set_activity_callback(protocol.activity)
    context_token = attach(extract(payload.get("trace_context") or {}))
    agent_kwargs_override = payload.get("agent_kwargs")
    if isinstance(agent_kwargs_override, dict):
        agent_kwargs_override = dict(agent_kwargs_override)
        legacy_key = os.environ.get("A2E_TRIAL_LEGACY_API_KEY")
        legacy_base = os.environ.get("A2E_TRIAL_LEGACY_API_BASE")
        if legacy_key:
            agent_kwargs_override["api_key"] = legacy_key
        if legacy_base:
            agent_kwargs_override["api_base"] = legacy_base
    try:
        return await _execute_attempt(
            cell=payload["cell"],
            trial=payload["trial"],
            task_payload=payload["task"],
            benchmark=payload["benchmark"],
            resolved_model=resolved,
            provider=provider,
            lifecycle=protocol.lifecycle,
            attempt=payload["attempt"],
            timeout_seconds=payload.get("timeout_seconds"),
            agent_kwargs_override=agent_kwargs_override,
        )
    finally:
        detach(context_token)
        set_activity_callback(None)
        provider.force_flush(timeout_millis=10_000)
        provider.shutdown()


def _failure(payload: dict[str, Any], exc: BaseException) -> TrialResult:
    now = datetime.now(timezone.utc)
    trial = payload["trial"]
    return TrialResult(
        trial_id=trial["trial_id"],
        cell_id=payload["cell"]["cell_id"],
        task_id=trial["task_id"],
        repetition=trial["repetition"],
        attempt=payload["attempt"],
        status="cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
        error=str(exc)[:2000] or type(exc).__name__,
        error_type=type(exc).__name__,
        started_at=now,
        ended_at=now,
    )


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    first_line, remaining_input = _read_initial_message(sys.stdin.fileno())
    if not first_line:
        return 2
    payload = json.loads(first_line)
    # Keep a private duplicate of the parent protocol pipe, then redirect fd 1
    # itself to stderr. This also catches native SDKs and descendant processes
    # that bypass Python's sys.stdout object.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    async def execute() -> TrialResult:
        protocol = Protocol(
            asyncio.get_running_loop(), protocol_fd, initial_input=remaining_input
        )
        protocol.start()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                return await _run(payload, protocol)
        finally:
            protocol.close()

    try:
        result = asyncio.run(execute())
    except BaseException as exc:
        result = _failure(payload, exc)
    result_path_value = os.environ.get("A2E_TRIAL_RESULT_PATH")
    if not result_path_value:
        raise RuntimeError("A2E_TRIAL_RESULT_PATH is missing")
    result_path = Path(result_path_value).resolve()
    atomic_write_json(result_path, result.model_dump(mode="json"))
    raw = result_path.read_bytes()
    # os.write bypasses redirect_stdout and preserves the protocol channel.
    # The notification stays tiny regardless of agent output or CTRF size.
    data = json.dumps(
        {
            "type": "result_ready",
            "path": result_path.name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        separators=(",", ":"),
    ).encode() + b"\n"
    os.write(protocol_fd, data)
    os.close(protocol_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
