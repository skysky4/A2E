from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_tb21_campaign.py"
_SPEC = importlib.util.spec_from_file_location("validate_tb21_campaign", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ValidationOptions = _MODULE.ValidationOptions
validate_campaign = _MODULE.validate_campaign


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[ValidationOptions, dict[str, Any]]:
    campaign = tmp_path / "campaign"
    cells = [
        {"cell_id": "cell-glm", "model": "glm-5.3"},
        {"cell_id": "cell-gpt", "model": "gpt-5.6-sol"},
    ]
    trials: list[dict[str, Any]] = []
    for cell in cells:
        for index in range(2):
            trial_id = f"trial-{cell['cell_id']}-{index}"
            trials.append({"trial_id": trial_id, "cell_id": cell["cell_id"]})
            ctrf = {
                "results": {
                    "summary": {
                        "tests": 1,
                        "passed": int(index == 0),
                        "failed": int(index != 0),
                    },
                    "tests": [
                        {
                            "name": "test",
                            "status": "passed" if index == 0 else "failed",
                        }
                    ],
                }
            }
            raw_ctrf = json.dumps(ctrf, separators=(",", ":")).encode()
            artifact_path = (
                campaign
                / "trials"
                / trial_id
                / "attempts"
                / "1"
                / "verifier"
                / "ctrf.json"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(raw_ctrf)
            _write_json(
                campaign / "trials" / trial_id / "result.json",
                {
                    "attempt": 1,
                    "status": "completed",
                    "uploaded": True,
                    "output": {
                        "resolved": index == 0,
                        "tb_reward": float(index == 0),
                        "tb_ctrf": ctrf,
                        "tb_ctrf_artifact": {
                            "path": "verifier/ctrf.json",
                            "size_bytes": len(raw_ctrf),
                            "sha256": hashlib.sha256(raw_ctrf).hexdigest(),
                        },
                    },
                    "grades": [{"name": "terminal-bench", "error": None}],
                },
            )
    lock = {
        "campaign_id": "campaign-test",
        "cells": cells,
        "trials": trials,
        "server": {
            "cells": {
                "cell-glm": {"experiment_id": 1},
                "cell-gpt": {"experiment_id": 2},
            }
        },
    }
    result = {
        "status": "completed",
        "counts": {"completed": 4},
        "concurrency": {
            "global": {"limit": 4, "active": 0, "high_water": 4},
            "sandbox": {"limit": 2, "active": 0, "high_water": 2},
            "model:total": {"limit": 2, "active": 0, "high_water": 2},
            "model:zai-glm": {"limit": 1, "active": 0, "high_water": 1},
            "model:gpt-5.6-sol": {"limit": 1, "active": 0, "high_water": 1},
            "grader": {"limit": 1, "active": 0, "high_water": 1},
            "upload": {"limit": 1, "active": 0, "high_water": 1},
        },
        "runtime": {
            "trial_processes": {
                "active": 0,
                "high_water": 4,
                "started": 4,
                "completed": 4,
            },
            "activities": {
                "docker:exec": {"active": 0, "high_water": 2, "started": 8}
            },
        },
        "gateways": {
            "glm-5.3": {
                "inflight_requests": 0,
                "inflight_high_water": 1,
            },
            "gpt-5.6-sol": {
                "inflight_requests": 0,
                "inflight_high_water": 1,
            },
        },
    }
    _write_json(campaign / "lock.json", lock)
    _write_json(campaign / "result.json", result)
    database = tmp_path / "a2e.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE experiment_runs (id INTEGER PRIMARY KEY, experiment_id INTEGER);
        CREATE TABLE experiment_run_annotations (
            id INTEGER PRIMARY KEY,
            experiment_run_id INTEGER,
            name TEXT
        );
        """
    )
    for run_id in range(1, 5):
        experiment_id = 1 if run_id <= 2 else 2
        connection.execute(
            "INSERT INTO experiment_runs (id, experiment_id) VALUES (?, ?)",
            (run_id, experiment_id),
        )
        connection.execute(
            "INSERT INTO experiment_run_annotations "
            "(experiment_run_id, name) VALUES (?, 'terminal-bench')",
            (run_id,),
        )
    connection.commit()
    connection.close()
    options = ValidationOptions(
        campaign_dir=campaign,
        database=database,
        expected_tasks_per_model=2,
        expected_models=("glm-5.3", "gpt-5.6-sol"),
        expected_limits={
            "global": 4,
            "sandbox": 2,
            "model:total": 2,
            "model:zai-glm": 1,
            "model:gpt-5.6-sol": 1,
            "grader": 1,
            "upload": 1,
        },
        require_saturated=frozenset(
            {
                "global",
                "sandbox",
                "model:total",
                "model:zai-glm",
                "model:gpt-5.6-sol",
            }
        ),
        require_exercised=frozenset({"grader", "upload"}),
    )
    return options, result


def test_valid_campaign_passes_and_summarizes_models(tmp_path: Path) -> None:
    options, _ = _fixture(tmp_path)
    report = validate_campaign(options)
    assert report["status"] == "passed"
    assert report["database_counts"] == {
        "experiment_runs": 4,
        "terminal_bench_evaluations": 4,
    }
    assert report["models"]["glm-5.3"]["resolved"] == 1
    assert report["models"]["gpt-5.6-sol"]["average_reward"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("concurrency", "global", "high_water", 5), "exceeds limit"),
        (("concurrency", "sandbox", "active", 1), "active=1"),
        (("concurrency", "model:zai-glm", "high_water", 0), "expected saturation"),
        (("counts", "completed", None, 3), "completed Trials"),
    ],
)
def test_invalid_campaign_accounting_fails(
    tmp_path: Path,
    mutation: tuple[str, str, str | None, int],
    message: str,
) -> None:
    options, result = _fixture(tmp_path)
    first, second, third, value = mutation
    if third is None:
        result[first][second] = value
    else:
        result[first][second][third] = value
    _write_json(options.campaign_dir / "result.json", result)
    report = validate_campaign(options)
    assert report["status"] == "failed"
    assert any(message in error for error in report["errors"])


def test_missing_evaluation_fails(tmp_path: Path) -> None:
    options, _ = _fixture(tmp_path)
    connection = sqlite3.connect(options.database)
    connection.execute("DELETE FROM experiment_run_annotations WHERE id = 1")
    connection.commit()
    connection.close()
    report = validate_campaign(options)
    assert report["status"] == "failed"
    assert any("evaluations" in error for error in report["errors"])


def test_missing_trial_result_fails(tmp_path: Path) -> None:
    options, _ = _fixture(tmp_path)
    missing = options.campaign_dir / "trials" / "trial-cell-glm-0" / "result.json"
    missing.unlink()
    report = validate_campaign(options)
    assert report["status"] == "failed"
    assert any("Trial results" in error for error in report["errors"])


def test_missing_database_run_fails(tmp_path: Path) -> None:
    options, _ = _fixture(tmp_path)
    connection = sqlite3.connect(options.database)
    connection.execute("DELETE FROM experiment_runs WHERE id = 1")
    connection.commit()
    connection.close()
    report = validate_campaign(options)
    assert report["status"] == "failed"
    assert any("ExperimentRun rows" in error for error in report["errors"])


def test_corrupt_ctrf_artifact_fails(tmp_path: Path) -> None:
    options, _ = _fixture(tmp_path)
    artifact = (
        options.campaign_dir
        / "trials"
        / "trial-cell-glm-0"
        / "attempts"
        / "1"
        / "verifier"
        / "ctrf.json"
    )
    artifact.write_text("{}", encoding="utf-8")
    report = validate_campaign(options)
    assert report["status"] == "failed"
    assert any("artifact" in error for error in report["errors"])
