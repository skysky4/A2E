"""Async supervisor for one isolated Trial attempt process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ageneval.model.gateway import ResolvedModel

from .schema import LifecycleEvent, TrialResult

LifecycleHook = Callable[[LifecycleEvent, int | None], Awaitable[None]]
ActivityHook = Callable[[str, str], None]
ProcessHook = Callable[[str, int], None]


class TrialProcessError(RuntimeError):
    """The Trial subprocess violated its protocol or exited unexpectedly."""


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> bool:
    """Terminate a Trial process group; return whether SIGKILL was required."""
    if process.returncode is not None:
        return False
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            process.terminate()
    except ProcessLookupError:
        return False
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return False
    except asyncio.TimeoutError:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        return True


class TrialProcessRunner:
    """Run one attempt with stdout reserved for small JSON-lines control IPC.

    The potentially large :class:`TrialResult` is transferred through an
    atomically-written file.  Stdout carries only lifecycle messages and a
    bounded ``result_ready`` notification, so agent/tool output size cannot
    exceed ``asyncio.StreamReader``'s line limit.
    """

    def __init__(
        self,
        *,
        script: Path,
        cancellation_grace_seconds: float,
        lifecycle: LifecycleHook,
        activity: ActivityHook | None = None,
        process_event: ProcessHook | None = None,
    ) -> None:
        self.script = script
        self.cancellation_grace_seconds = cancellation_grace_seconds
        self.lifecycle = lifecycle
        self.activity = activity
        self.process_event = process_event

    async def run(
        self,
        *,
        payload: dict[str, Any],
        resolved_model: ResolvedModel,
        python: str,
        pythonpath: str,
        stderr_path: Path,
        result_path: Path,
        timeout_seconds: float | None,
        extra_env: dict[str, str] | None = None,
    ) -> TrialResult:
        env = dict(os.environ)
        env["A2E_TRIAL_MODEL_API_KEY"] = resolved_model.api_key.get_secret_value()
        cell = payload["cell"]
        trial = payload["trial"]
        env.update(
            {
                "A2E_CAMPAIGN_ID": str(cell["campaign_id"]),
                "A2E_CELL_ID": str(cell["cell_id"]),
                "A2E_TRIAL_ID": str(trial["trial_id"]),
                "A2E_ATTEMPT": str(payload["attempt"]),
                "A2E_TRIAL_RESULT_PATH": str(result_path.resolve()),
                "A2E_TRIAL_ATTEMPT_DIR": str(result_path.parent.resolve()),
            }
        )
        if extra_env:
            env.update(extra_env)
        if pythonpath:
            env["PYTHONPATH"] = (
                f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else pythonpath
            )
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        # Never accept a stale file from an interrupted launch.  A result is
        # valid only after this process emits a matching result_ready digest.
        result_path.unlink(missing_ok=True)

        def load_result(message: dict[str, Any]) -> TrialResult:
            relative_path = message.get("path")
            if relative_path != result_path.name:
                raise TrialProcessError(
                    f"Trial result_ready path mismatch: {relative_path!r}"
                )
            expected_size = message.get("size_bytes")
            expected_digest = message.get("sha256")
            if not isinstance(expected_size, int) or expected_size < 0:
                raise TrialProcessError("Trial result_ready has invalid size_bytes")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise TrialProcessError("Trial result_ready has invalid sha256")
            try:
                raw = result_path.read_bytes()
            except OSError as exc:
                raise TrialProcessError(
                    f"Trial announced a missing result file: {result_path}"
                ) from exc
            if len(raw) != expected_size:
                raise TrialProcessError(
                    "Trial result file size does not match result_ready notification"
                )
            actual_digest = hashlib.sha256(raw).hexdigest()
            if actual_digest != expected_digest:
                raise TrialProcessError(
                    "Trial result file digest does not match result_ready notification"
                )
            try:
                result = TrialResult.model_validate_json(raw)
            except (ValueError, TypeError) as exc:
                raise TrialProcessError(
                    "Trial result file is not a valid TrialResult"
                ) from exc
            expected_identity = (
                str(trial["trial_id"]),
                str(cell["cell_id"]),
                str(trial["task_id"]),
                int(trial["repetition"]),
                int(payload["attempt"]),
            )
            actual_identity = (
                result.trial_id,
                result.cell_id,
                result.task_id,
                result.repetition,
                result.attempt,
            )
            if actual_identity != expected_identity:
                raise TrialProcessError("Trial result identity does not match launch payload")
            return result

        with stderr_path.open("ab", buffering=0) as error_stream:
            process = await asyncio.create_subprocess_exec(
                python,
                str(self.script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=error_stream,
                env=env,
                start_new_session=os.name == "posix",
            )
            if self.process_event is not None:
                self.process_event("start", process.pid)
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n"
            )
            await process.stdin.drain()

            async def exchange() -> TrialResult:
                result: TrialResult | None = None
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    try:
                        message = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise TrialProcessError(
                            "Trial stdout contained non-protocol output; see attempt stderr.log"
                        ) from exc
                    if not isinstance(message, dict):
                        raise TrialProcessError("Trial protocol message must be an object")
                    kind = message.get("type")
                    if kind == "lifecycle_request":
                        request_id = message.get("request_id")
                        if not isinstance(request_id, str):
                            raise TrialProcessError("lifecycle request is missing request_id")
                        response: dict[str, Any] = {
                            "type": "lifecycle_response",
                            "request_id": request_id,
                        }
                        try:
                            await self.lifecycle(
                                LifecycleEvent(message["event"]), message.get("attempt")
                            )
                            response["ok"] = True
                        except BaseException as exc:
                            response.update({"ok": False, "error": str(exc)[:1000]})
                            process.stdin.write(
                                json.dumps(response, separators=(",", ":")).encode() + b"\n"
                            )
                            await process.stdin.drain()
                            raise
                        process.stdin.write(
                            json.dumps(response, separators=(",", ":")).encode() + b"\n"
                        )
                        await process.stdin.drain()
                    elif kind == "activity":
                        if self.activity is not None:
                            self.activity(
                                str(message.get("name") or "unknown"),
                                str(message.get("state")),
                            )
                    elif kind == "result_ready":
                        if result is not None:
                            raise TrialProcessError("Trial emitted more than one result")
                        result = load_result(message)
                    elif kind == "result":
                        raise TrialProcessError(
                            "inline Trial results are no longer supported; use result_ready"
                        )
                    else:
                        raise TrialProcessError(f"unknown Trial protocol message: {kind!r}")
                returncode = await process.wait()
                if returncode != 0:
                    raise TrialProcessError(f"Trial process exited with code {returncode}")
                if result is None:
                    raise TrialProcessError("Trial process exited without a result")
                return result

            killed = False
            try:
                if timeout_seconds is None:
                    return await exchange()
                return await asyncio.wait_for(exchange(), timeout=timeout_seconds)
            except BaseException:
                killed = await _terminate_process(
                    process, grace_seconds=self.cancellation_grace_seconds
                )
                raise
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    killed = (
                        await _terminate_process(
                            process, grace_seconds=self.cancellation_grace_seconds
                        )
                        or killed
                    )
                if self.process_event is not None:
                    self.process_event("killed" if killed else "end", process.pid)


__all__ = ["TrialProcessError", "TrialProcessRunner"]
