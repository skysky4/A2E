"""GDPval computer-use sandbox: official E2B, else an isolated workspace.

Official GDPval sampling runs inside a container (E2B / production Docker)
with the task's reference files on disk. This module:

* prefers E2B when ``E2B_API_KEY`` is set and the SDK imports;
* otherwise uses a per-task isolated directory and copies every reference
  file into it so attachments are actually present (never a text placeholder).

Tool executors talk to this object, not the host prompt.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SANDBOX_ROOT = "/home/user"


class GDPSandbox:
    """One task workspace. ``backend`` is ``e2b`` or ``local``."""

    def __init__(self, workspace: str | Path, *, backend: str = "local", handle: Any = None):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.handle = handle

    def put_file(self, name: str, src: str | Path) -> str:
        dest_name = os.path.basename(name)
        src_path = Path(src)
        if self.backend == "e2b" and self.handle is not None:
            data = src_path.read_bytes()
            remote = f"{_SANDBOX_ROOT}/{dest_name}"
            _e2b_write(self.handle, remote, data)
            local = self.workspace / dest_name
            if src_path.resolve() != local.resolve():
                shutil.copy2(src_path, local)
            return remote
        dest = self.workspace / dest_name
        if src_path.resolve() != dest.resolve():
            shutil.copy2(src_path, dest)
        return str(dest)

    def write_bytes(self, name: str, data: bytes) -> str:
        dest_name = os.path.basename(name)
        (self.workspace / dest_name).write_bytes(data)
        if self.backend == "e2b" and self.handle is not None:
            remote = f"{_SANDBOX_ROOT}/{dest_name}"
            _e2b_write(self.handle, remote, data)
            return remote
        return str(self.workspace / dest_name)

    def read_bytes(self, name: str) -> bytes:
        dest_name = os.path.basename(name)
        local = self.workspace / dest_name
        if local.is_file():
            return local.read_bytes()
        if self.backend == "e2b" and self.handle is not None:
            data = _e2b_read(self.handle, f"{_SANDBOX_ROOT}/{dest_name}")
            local.write_bytes(data)
            return data
        raise FileNotFoundError(dest_name)

    def list_files(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.workspace.iterdir()):
            if path.is_file():
                items.append({"name": path.name, "path": str(path), "bytes": path.stat().st_size})
        return items

    def exec_python(self, code: str, *, timeout: int = 30) -> dict[str, Any]:
        if self.backend == "e2b" and self.handle is not None:
            try:
                result = _e2b_run(self.handle, ["python", "-c", code], timeout=timeout)
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("E2B python failed (%s); using local workspace", exc)
        return _local_python(code, cwd=self.workspace, timeout=timeout)

    def exec_shell(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        if self.backend == "e2b" and self.handle is not None:
            try:
                return _e2b_run(self.handle, ["bash", "-lc", command], timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("E2B shell failed (%s); using local workspace", exc)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "stdout": "", "stderr": f"timeout after {timeout}s"}
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:20000],
            "stderr": (proc.stderr or "")[:4000],
        }

    def cleanup(self) -> None:
        if self.handle is not None:
            try:
                kill = getattr(self.handle, "kill", None) or getattr(self.handle, "close", None)
                if kill:
                    kill()
            except Exception:  # noqa: BLE001
                pass
            self.handle = None


def open_gdp_sandbox(workspace: str | Path | None = None) -> GDPSandbox:
    """Start official E2B when configured; otherwise an isolated local workspace."""
    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="a2e-gdpval-"))
    ws.mkdir(parents=True, exist_ok=True)
    handle = _try_e2b()
    if handle is not None:
        logger.info("GDPval sandbox backend=e2b workspace=%s", ws)
        return GDPSandbox(ws, backend="e2b", handle=handle)
    logger.info("GDPval sandbox backend=local workspace=%s", ws)
    return GDPSandbox(ws, backend="local")


def sandbox_from_state(state: dict[str, Any]) -> GDPSandbox:
    existing = state.get("_gdp_sandbox")
    if isinstance(existing, GDPSandbox):
        return existing
    ws = state.get("workspace") or tempfile.mkdtemp(prefix="a2e-gdpval-")
    box = GDPSandbox(ws, backend=str(state.get("sandbox_backend") or "local"))
    state["_gdp_sandbox"] = box
    state["workspace"] = str(box.workspace)
    state["sandbox_backend"] = box.backend
    return box


def _try_e2b() -> Any:
    key = (os.environ.get("E2B_API_KEY") or "").strip()
    if not key:
        return None
    timeout = int(os.environ.get("A2E_GDPVAL_E2B_TIMEOUT", "3600"))
    try:
        from e2b import Sandbox  # type: ignore

        create = getattr(Sandbox, "create", None)
        if create:
            return create(api_key=key, timeout=timeout)
        return Sandbox(api_key=key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("E2B SDK unavailable (%s)", str(exc)[:160])
        return None


def _e2b_write(handle: Any, path: str, data: bytes) -> None:
    files = getattr(handle, "files", None)
    if files is not None and hasattr(files, "write"):
        files.write(path, data)
        return
    filesystem = getattr(handle, "filesystem", None)
    if filesystem is not None and hasattr(filesystem, "write"):
        filesystem.write(path, data)
        return
    raise RuntimeError("E2B handle has no files.write")


def _e2b_read(handle: Any, path: str) -> bytes:
    files = getattr(handle, "files", None)
    if files is not None and hasattr(files, "read"):
        raw = files.read(path)
        return raw if isinstance(raw, bytes) else str(raw).encode("utf-8", "replace")
    filesystem = getattr(handle, "filesystem", None)
    if filesystem is not None and hasattr(filesystem, "read"):
        raw = filesystem.read(path)
        return raw if isinstance(raw, bytes) else str(raw).encode("utf-8", "replace")
    raise FileNotFoundError(path)


def _e2b_run(handle: Any, argv: list[str], *, timeout: int) -> dict[str, Any]:
    commands = getattr(handle, "commands", None)
    if commands is not None and hasattr(commands, "run"):
        result = commands.run(argv, timeout=timeout)
        return {
            "returncode": int(getattr(result, "exit_code", 0) or 0),
            "stdout": str(getattr(result, "stdout", "") or "")[:20000],
            "stderr": str(getattr(result, "stderr", "") or "")[:4000],
        }
    process = getattr(handle, "process", None)
    if process is not None and hasattr(process, "start"):
        proc = process.start(" ".join(argv))
        wait = getattr(proc, "wait", None)
        if wait:
            wait(timeout=timeout)
        return {
            "returncode": int(getattr(proc, "exit_code", 0) or 0),
            "stdout": str(getattr(proc, "stdout", "") or "")[:20000],
            "stderr": str(getattr(proc, "stderr", "") or "")[:4000],
        }
    raise RuntimeError("E2B handle has no commands.run")


def _local_python(code: str, *, cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": f"code_exec timed out after {timeout}s"}
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[:20000],
        "stderr": (proc.stderr or "")[:4000],
    }
