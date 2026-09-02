"""Grade a Terminal-Bench 2.1 task inside the live sandbox.

``score_terminal_bench_2_1(task, sandbox, model_patch)`` is called by
``SandboxScoringRunner`` while the container is still alive, reproducing the
official TB2 verifier flow:

    1. after the agent exits, copy trusted, pinned ``uv``/``uvx`` binaries from
       the host into the still-live container;
    2. copy the held-out ``tests/`` dir into the container at ``/tests/`` (the
       agent never saw these — they arrive only at grading time), replacing the
       test launcher's online uv installer with a local availability check;
    3. ``mkdir -p /logs/verifier`` (where TB2's ``test.sh`` writes its outputs);
    4. run ``bash /tests/test.sh`` in the task working dir. ``uvx`` resolves
       pytest from the pre-warmed cache volumes and writes CTRF plus reward;
    5. read ``reward.txt`` → ``resolved``; parse ``ctrf.json`` for test counts.

Prepare the pinned binaries and dependency caches before a benchmark with
``python scripts/prewarm_tb21_verifier_cache.py``. Override the trusted host
binary directory with ``A2E_TB21_UV_BIN_DIR`` when needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.grading import GradeReport, GraderSpec

logger = logging.getLogger(__name__)

_DEFAULT_DOCKER_GW = "172.17.0.1"  # default docker bridge gateway (host from container)
_UV_VERSION = "0.9.5"
_UV_CONTAINER_DIR = "/opt/a2e-verifier/bin"
_DEFAULT_CONTAINER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_UV_INSTALL_RE = re.compile(
    r"(?m)^curl -LsSf https://astral\.sh/uv/0\.9\.5/install\.sh \| sh[ \t]*\n"
    r"(?:[ \t]*\n)*source \$HOME/\.local/bin/env[ \t]*$"
)
_UV_LOCAL_CHECK = """# uv/uvx are injected by the A2E grader after the agent exits.
if ! command -v uvx >/dev/null 2>&1; then
  echo "Verifier bootstrap failed: injected uvx is unavailable" >&2
  exit 86
fi"""


def _tests_dir_for(task: TaskInput) -> Path:
    """Resolve the vendored ``tests/`` dir for this task (package data)."""
    name = str(task.metadata.get("tb_task") or task.task_id)
    return Path(__file__).resolve().parent / "vendor" / "tasks" / name / "tests"


def _container_proxy_env() -> dict[str, str]:
    """Proxy env to inject so the in-container verifier can reach the internet.

    Priority: explicit ``A2E_TB2_SCORE_PROXY`` → host proxy with
    localhost/127.0.0.1 rewritten to the docker bridge gateway → none.
    """
    override = os.environ.get("A2E_TB2_SCORE_PROXY")
    host = override or os.environ.get("https_proxy") or os.environ.get("http_proxy") \
        or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if not host:
        return {}
    gw = os.environ.get("A2E_TB2_DOCKER_GW", _DEFAULT_DOCKER_GW)
    fixed = host.replace("127.0.0.1", gw).replace("localhost", gw)
    return {
        "http_proxy": fixed, "https_proxy": fixed,
        "HTTP_PROXY": fixed, "HTTPS_PROXY": fixed,
    }


def _repo_root() -> Path:
    """Find the checkout root without assuming a fixed editable-install depth."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "prewarm_tb21_verifier_cache.py").is_file():
            return parent
    raise FileNotFoundError("could not locate the A2E repository root")


def _trusted_uv_dir() -> Path:
    configured = os.environ.get("A2E_TB21_UV_BIN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return _repo_root() / ".a2e-cache" / "tb21-verifier" / f"uv-{_UV_VERSION}"


def _checked_exec(sandbox, cmd: list[str], **kwargs: Any):
    result = sandbox.exec(cmd, **kwargs)
    if result.returncode != 0:
        rendered = " ".join(cmd)
        raise RuntimeError(
            f"container command failed ({result.returncode}): {rendered}: "
            f"{(result.stderr or result.stdout or '')[-1000:]}"
        )
    return result


def _inject_uv(sandbox) -> tuple[dict[str, str], str]:
    """Copy pinned uv tools into the container only after the agent has exited."""
    source_dir = _trusted_uv_dir()
    sources = {name: source_dir / name for name in ("uv", "uvx")}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "trusted Terminal-Bench verifier binaries are missing: "
            f"{missing}; run scripts/prewarm_tb21_verifier_cache.py first"
        )

    _checked_exec(sandbox, ["mkdir", "-p", _UV_CONTAINER_DIR])
    for name, source in sources.items():
        sandbox.write_file(f"{_UV_CONTAINER_DIR}/{name}", source.read_bytes())
    _checked_exec(
        sandbox,
        [
            "chmod",
            "0555",
            f"{_UV_CONTAINER_DIR}/uv",
            f"{_UV_CONTAINER_DIR}/uvx",
        ],
    )

    path_result = _checked_exec(sandbox, ["sh", "-c", 'printf "%s" "$PATH"'])
    container_path = path_result.stdout.strip() or _DEFAULT_CONTAINER_PATH
    verifier_path = f"{_UV_CONTAINER_DIR}:{container_path}"
    version_result = _checked_exec(
        sandbox,
        [f"{_UV_CONTAINER_DIR}/uvx", "--version"],
        env={"PATH": verifier_path},
    )
    version = version_result.stdout.strip()
    expected = f"uvx {_UV_VERSION}"
    if not version.startswith(expected):
        raise RuntimeError(f"expected {expected}, got {version!r}")
    return {"PATH": verifier_path, "UV_OFFLINE": "1"}, version


def _test_contents(path: Path, rel: str) -> tuple[bytes, bool]:
    contents = path.read_bytes()
    if rel != "test.sh":
        return contents, False
    text = contents.decode("utf-8")
    rewritten, count = _UV_INSTALL_RE.subn(_UV_LOCAL_CHECK, text)
    if "https://astral.sh/uv/0.9.5/install.sh" in text and count != 1:
        raise ValueError(f"could not safely rewrite uv bootstrap in {path}")
    return rewritten.encode("utf-8"), count == 1


def _copy_tests_into_container(tests_dir: Path, sandbox) -> tuple[int, bool]:
    """Write held-out tests and disable their redundant online uv bootstrap."""
    count = 0
    bootstrap_rewritten = False
    for path in sorted(tests_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(tests_dir).as_posix()
        contents, rewritten = _test_contents(path, rel)
        sandbox.write_file(f"/tests/{rel}", contents)
        bootstrap_rewritten |= rewritten
        count += 1
    return count, bootstrap_rewritten


def _persist_ctrf_artifact(raw: bytes) -> dict[str, Any] | None:
    """Atomically preserve the exact verifier bytes in the Trial attempt.

    The orchestrator supplies ``A2E_TRIAL_ATTEMPT_DIR`` to isolated Trial
    processes.  Keeping this helper environment-driven avoids coupling the
    dataset package to Campaign classes and still lets direct grader callers
    run without an artifact directory.
    """
    attempt_dir = os.environ.get("A2E_TRIAL_ATTEMPT_DIR")
    if not attempt_dir:
        return None
    target = Path(attempt_dir).resolve() / "verifier" / "ctrf.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".ctrf.json.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "path": "verifier/ctrf.json",
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _parse_ctrf(sandbox) -> tuple[dict[str, Any], str | None]:
    """Preserve the raw CTRF, then parse its full payload and summary."""
    try:
        raw_value = sandbox.read_file("/logs/verifier/ctrf.json", text=False)
        raw = raw_value.encode() if isinstance(raw_value, str) else raw_value
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"[-1000:]

    artifact: dict[str, Any] = {}
    try:
        artifact_metadata = _persist_ctrf_artifact(raw)
        if artifact_metadata is not None:
            artifact["tb_ctrf_artifact"] = artifact_metadata
    except Exception as exc:
        artifact["tb_ctrf_artifact_error"] = f"{type(exc).__name__}: {exc}"[-1000:]
        return artifact, artifact["tb_ctrf_artifact_error"]

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("CTRF root must be an object")
        summary = (parsed.get("results") or {}).get("summary") or {}
        counts = {
            "tb_tests_total": summary.get("tests"),
            "tb_tests_passed": summary.get("passed"),
            "tb_tests_failed": summary.get("failed"),
        }
        if any(not isinstance(value, int) for value in counts.values()):
            raise ValueError(f"CTRF summary has invalid counts: {counts}")
        return ({**artifact, "tb_ctrf": parsed, **counts}, None)
    except Exception as exc:
        return artifact, f"{type(exc).__name__}: {exc}"[-1000:]


def score_terminal_bench_2_1(task: TaskInput, sandbox, model_patch: str) -> dict[str, Any]:
    """Grade the agent's work via the official held-out tests. Returns at least
    ``{"resolved": bool}`` (plus reward + per-test counts when available)."""
    tests_dir = _tests_dir_for(task)
    if not (tests_dir / "test.sh").exists():
        return {"resolved": False, "status": "no_tests"}

    workdir = str(task.metadata.get("tb_workdir") or "/app")
    timeout = int(float(task.metadata.get("verifier_timeout_sec") or 900.0))
    verifier_env = {
        str(key): str(value)
        for key, value in (task.metadata.get("verifier_env") or {}).items()
    }

    phase = "inject_uv"
    try:
        uv_env, uv_version = _inject_uv(sandbox)
        phase = "copy_tests"
        n_files, bootstrap_rewritten = _copy_tests_into_container(tests_dir, sandbox)
        phase = "prepare_logs"
        _checked_exec(sandbox, ["mkdir", "-p", "/logs/verifier"])
        # A fresh sandbox should not contain these, but clearing them makes the
        # reward/report provenance explicit and prevents stale or agent-created
        # artifacts from being mistaken for this verifier invocation's output.
        _checked_exec(
            sandbox,
            [
                "rm",
                "-f",
                "/logs/verifier/reward.txt",
                "/logs/verifier/ctrf.json",
            ]
        )
        phase = "run_tests"
        res = sandbox.exec(
            ["bash", "/tests/test.sh"],
            cwd=workdir,
            env={**_container_proxy_env(), **verifier_env, **uv_env},
            timeout=timeout,
        )
    except Exception as exc:
        logger.exception("terminal-bench-2.1 verifier crashed on %s", task.task_id)
        return {
            "resolved": False,
            "status": "verifier_error",
            "tb_verifier_phase": phase,
            "score_error": str(exc)[:1000],
        }

    reward_raw = ""
    reward_error = None
    try:
        reward_raw = str(sandbox.read_file("/logs/verifier/reward.txt", text=True)).strip()
    except Exception as exc:
        reward_error = f"{type(exc).__name__}: {exc}"[-1000:]

    ctrf, ctrf_error = _parse_ctrf(sandbox)

    has_report = ctrf_error is None
    resolved = bool(
        reward_raw == "1"
        and has_report
        and ctrf["tb_tests_total"] > 0
        and ctrf["tb_tests_failed"] == 0
    )
    report: dict[str, Any] = {
        "resolved": resolved,
        "tb_reward": reward_raw or None,
        "status": "graded" if reward_raw and has_report else "verifier_error",
        "tb_verifier_files": n_files,
        "tb_verifier_exit": res.returncode,
        "tb_verifier_phase": "complete" if has_report else "parse_report",
        "tb_uv_injected": True,
        "tb_uv_version": uv_version,
        "tb_bootstrap_rewritten": bootstrap_rewritten,
        # Always retain both streams.  A reward of 0 without CTRF often means
        # uv/pytest failed before collection, and test.sh may still exit 0
        # because its final command writes reward.txt.
        "tb_verifier_stdout_tail": (res.stdout or "")[-8000:],
        "tb_verifier_stderr_tail": (res.stderr or "")[-8000:],
        "tb_reward_read_error": reward_error,
        "tb_ctrf_error": ctrf_error,
    }
    report.update(ctrf)
    logger.info(
        "terminal-bench-2.1 score %s: resolved=%s reward=%r",
        task.task_id,
        resolved,
        reward_raw,
    )
    return report


def _reward_number(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def grade_terminal_bench_2_1_output(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GradeReport:
    """Build canonical metrics from the enhanced verifier's existing report."""
    del expected, input, metadata
    resolved = bool(output.get("resolved"))
    score = float(resolved)
    reward = _reward_number(output.get("tb_reward"))
    return GradeReport(
        score=score,
        passed=resolved,
        metrics={"tb_resolved": score, "tb_reward": reward},
        metadata={
            key: output.get(key)
            for key in (
                "status",
                "tb_tests_total",
                "tb_tests_passed",
                "tb_tests_failed",
                "tb_verifier_files",
                "tb_verifier_exit",
                "tb_verifier_phase",
                "tb_uv_injected",
                "tb_uv_version",
                "tb_bootstrap_rewritten",
                "tb_ctrf_artifact",
            )
        },
        error=output.get("score_error") or output.get("tb_ctrf_error"),
        explanation="Terminal-Bench 2.1 reward from its live held-out verifier.",
        official=True,
        source="harbor-framework/terminal-bench-2-1",
        version="2.1",
    )


GRADER = GraderSpec(
    id="tb_resolved",
    grade=score_terminal_bench_2_1,
    summarize=grade_terminal_bench_2_1_output,
    mode="inline",
    official=True,
    source="harbor-framework/terminal-bench-2-1",
    version="2.1",
)
post_platform_grade = grade_terminal_bench_2_1_output

__all__ = [
    "GRADER",
    "grade_terminal_bench_2_1_output",
    "post_platform_grade",
    "score_terminal_bench_2_1",
]
