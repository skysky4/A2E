from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ageneval.task.core.dataset import TaskInput
from ageneval.task.datasets.terminal_bench_2_1.grader import (
    _test_contents,
    score_terminal_bench_2_1,
)


class _Sandbox:
    def __init__(self, *, reward: str, ctrf: dict | None) -> None:
        self.reward = reward
        self.ctrf = ctrf
        self.writes: list[tuple[str, bytes]] = []

    def write_file(self, path: str, contents: str | bytes) -> None:
        data = contents.encode() if isinstance(contents, str) else contents
        self.writes.append((path, data))

    def exec(self, cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        if cmd[-2:] == ["uvx", "--version"] or cmd[-1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="uvx 0.9.5\n", stderr="")
        if cmd == ["sh", "-c", 'printf "%s" "$PATH"']:
            return SimpleNamespace(returncode=0, stdout="/usr/bin:/bin", stderr="")
        return SimpleNamespace(returncode=0, stdout="pytest output", stderr="")

    def read_file(self, path: str, text: bool = True) -> str:
        assert text
        if path.endswith("reward.txt"):
            return self.reward
        if path.endswith("ctrf.json") and self.ctrf is not None:
            return json.dumps(self.ctrf)
        raise FileNotFoundError(path)


def _task() -> TaskInput:
    return TaskInput(
        task_id="circuit-fibsqrt",
        instruction="",
        metadata={
            "tb_task": "circuit-fibsqrt",
            "tb_workdir": "/app",
            "verifier_timeout_sec": 10,
        },
    )


def _trusted_tools(tmp_path: Path, monkeypatch) -> None:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    (tool_dir / "uv").write_bytes(b"trusted uv")
    (tool_dir / "uvx").write_bytes(b"trusted uvx")
    monkeypatch.setenv("A2E_TB21_UV_BIN_DIR", str(tool_dir))


def test_test_launcher_uses_injected_uv(tmp_path: Path) -> None:
    launcher = tmp_path / "test.sh"
    launcher.write_text(
        "#!/bin/bash\n"
        "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n"
        "source $HOME/.local/bin/env\n"
        "uvx pytest\n",
        encoding="utf-8",
    )

    contents, rewritten = _test_contents(launcher, "test.sh")

    assert rewritten is True
    assert b"command -v uvx" in contents
    assert b"https://astral.sh" not in contents


def test_reward_without_ctrf_is_verifier_error(tmp_path: Path, monkeypatch) -> None:
    _trusted_tools(tmp_path, monkeypatch)
    sandbox = _Sandbox(reward="0", ctrf=None)

    result = score_terminal_bench_2_1(_task(), sandbox, "")

    written_paths = [path for path, _ in sandbox.writes]
    assert written_paths[:2] == [
        "/opt/a2e-verifier/bin/uv",
        "/opt/a2e-verifier/bin/uvx",
    ]
    copied_launcher = dict(sandbox.writes)["/tests/test.sh"]
    assert b"https://astral.sh" not in copied_launcher
    assert result["resolved"] is False
    assert result["status"] == "verifier_error"
    assert result["tb_verifier_phase"] == "parse_report"
    assert result["tb_uv_injected"] is True


def test_reward_and_ctrf_must_both_confirm_success(tmp_path: Path, monkeypatch) -> None:
    _trusted_tools(tmp_path, monkeypatch)
    sandbox = _Sandbox(
        reward="1",
        ctrf={
            "results": {
                "summary": {"tests": 3, "passed": 3, "failed": 0},
            }
        },
    )

    result = score_terminal_bench_2_1(_task(), sandbox, "")

    assert result["resolved"] is True
    assert result["status"] == "graded"
    assert result["tb_tests_total"] == 3
    assert result["tb_tests_passed"] == 3
    assert result["tb_tests_failed"] == 0
