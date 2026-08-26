"""Atomic Harbor-style run directory and single-controller lock."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import TrialResult


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


class RunDirectory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.json"
        self.lock_path = self.root / "lock.json"
        self.result_path = self.root / "result.json"
        self.controller_lock_path = self.root / ".controller.lock"

    def initialize(self, *, config: dict[str, Any], lock: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists() or self.lock_path.exists():
            raise FileExistsError(f"run directory already initialized: {self.root}")
        atomic_write_json(self.config_path, config)
        atomic_write_json(self.lock_path, lock)
        atomic_write_json(
            self.result_path,
            {
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "counts": {},
            },
        )

    def verify_config(self, normalized_config: dict[str, Any]) -> dict[str, Any]:
        existing = read_json(self.config_path)
        if existing != normalized_config:
            raise ValueError(
                f"resume config does not match immutable config.json in {self.root}"
            )
        lock = read_json(self.lock_path)
        if not isinstance(lock, dict):
            raise ValueError("lock.json must contain an object")
        return lock

    def update_lock(self, update: dict[str, Any]) -> None:
        current = read_json(self.lock_path)
        current.update(update)
        atomic_write_json(self.lock_path, current)

    def trial_dir(self, trial_id: str) -> Path:
        return self.root / "trials" / trial_id

    def write_trial_spec(self, trial_id: str, spec: dict[str, Any], lock: dict[str, Any]) -> None:
        directory = self.trial_dir(trial_id)
        for child in ("agent", "verifier", "artifacts", "attempts"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        if not (directory / "config.json").exists():
            atomic_write_json(directory / "config.json", spec)
        if not (directory / "lock.json").exists():
            atomic_write_json(directory / "lock.json", lock)

    def write_trial_result(self, result: TrialResult) -> None:
        directory = self.trial_dir(result.trial_id)
        payload = result.model_dump(mode="json")
        atomic_write_json(directory / "attempts" / str(result.attempt) / "result.json", payload)
        atomic_write_json(directory / "result.json", payload)

    def write_trial_attempt(self, result: TrialResult) -> None:
        """Persist a retry attempt without publishing it as the Trial terminal state."""
        directory = self.trial_dir(result.trial_id)
        payload = result.model_dump(mode="json")
        atomic_write_json(directory / "attempts" / str(result.attempt) / "result.json", payload)

    def load_trial_result(self, trial_id: str) -> TrialResult | None:
        path = self.trial_dir(trial_id) / "result.json"
        if not path.exists():
            return None
        try:
            return TrialResult.model_validate(read_json(path))
        except (ValueError, TypeError) as exc:
            quarantine = path.with_name(f"result.corrupt-{time.time_ns()}.json")
            os.replace(path, quarantine)
            warnings.warn(
                f"quarantined corrupt trial result {path} as {quarantine.name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def summarize(
        self,
        trial_ids: list[str],
        *,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial_id in trial_ids:
            result = self.load_trial_result(trial_id)
            key = result.status if result else "pending"
            counts[key] = counts.get(key, 0) + 1
        summary = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            **(extra or {}),
        }
        atomic_write_json(self.result_path, summary)
        return summary

    def summarize_cell(self, cell_id: str, trial_ids: list[str]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial_id in trial_ids:
            result = self.load_trial_result(trial_id)
            key = result.status if result else "pending"
            counts[key] = counts.get(key, 0) + 1
        summary = {
            "cell_id": cell_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
        }
        atomic_write_json(self.root / "cells" / cell_id / "result.json", summary)
        return summary

    @contextmanager
    def controller_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        hostname = socket.gethostname()
        payload = json.dumps({"pid": os.getpid(), "host": hostname})
        for attempt in range(2):
            try:
                fd = os.open(
                    self.controller_lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                break
            except FileExistsError as exc:
                try:
                    owner = read_json(self.controller_lock_path)
                except ValueError:
                    owner = {}
                owner_pid = owner.get("pid") if isinstance(owner, dict) else None
                owner_host = owner.get("host") if isinstance(owner, dict) else None
                stale = owner_host == hostname and isinstance(owner_pid, int)
                if stale:
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        stale = False
                    else:
                        stale = False
                if stale and attempt == 0:
                    self.controller_lock_path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"campaign already has a controller: {owner}") from exc
        else:  # pragma: no cover - loop either opens or raises
            raise RuntimeError("failed to acquire campaign controller lock")
        try:
            os.write(fd, payload.encode())
            os.close(fd)
            yield
        finally:
            try:
                self.controller_lock_path.unlink()
            except FileNotFoundError:
                pass
