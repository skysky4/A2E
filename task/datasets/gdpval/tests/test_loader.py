from __future__ import annotations

from pathlib import Path

from ageneval.task.datasets.gdpval.loader import _attachments_dir


def test_attachment_cache_defaults_to_current_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("A2E_GDPVAL_FILES_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert _attachments_dir() == tmp_path / "a2e" / "gdpval-files"


def test_attachment_cache_accepts_explicit_override(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "gdpval"
    monkeypatch.setenv("A2E_GDPVAL_FILES_DIR", str(configured))

    assert _attachments_dir() == configured
