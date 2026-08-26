from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tb21_missing_ctrf_rerun.py"
_SPEC = importlib.util.spec_from_file_location("tb21_missing_ctrf_rerun", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_pairs_are_selected_per_model_without_cross_product(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    cells = [
        {"cell_id": "cell-glm", "model": "glm-5.3"},
        {"cell_id": "cell-gpt", "model": "gpt-5.6-sol"},
    ]
    trials = []
    cases = [
        ("trial-glm-valid", "cell-glm", "task-a", True),
        ("trial-glm-missing", "cell-glm", "task-b", False),
        ("trial-gpt-valid", "cell-gpt", "task-c", True),
        ("trial-gpt-missing", "cell-gpt", "task-d", False),
    ]
    for trial_id, cell_id, task_id, valid in cases:
        trials.append({"trial_id": trial_id, "cell_id": cell_id, "task_id": task_id})
        output = (
            {"tb_tests_total": 2, "tb_tests_passed": 1, "tb_tests_failed": 1}
            if valid
            else {"tb_ctrf_error": "truncated"}
        )
        _write_json(
            campaign / "trials" / trial_id / "result.json",
            {"status": "completed" if valid else "failed", "output": output},
        )
    _write_json(campaign / "lock.json", {"cells": cells, "trials": trials})

    selected = _MODULE.missing_pairs(campaign)

    assert [row["task_id"] for row in selected["glm-5.3"]] == ["task-b"]
    assert [row["task_id"] for row in selected["gpt-5.6-sol"]] == ["task-d"]


def test_validate_checks_raw_ctrf_and_complete_database_payload(tmp_path: Path) -> None:
    output_root = tmp_path / "rerun"
    runs_dir = output_root / "campaigns" / "glm-5-3"
    campaign = runs_dir / "campaign-test"
    trial_id = "trial-test"
    ctrf = {
        "results": {
            "summary": {"tests": 1, "passed": 1, "failed": 0},
            "tests": [{"name": "test", "status": "passed"}],
        }
    }
    raw = json.dumps(ctrf, separators=(",", ":")).encode()
    artifact = campaign / "trials" / trial_id / "attempts" / "1" / "verifier" / "ctrf.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(raw)
    task_output = {
        "tb_tests_total": 1,
        "tb_tests_passed": 1,
        "tb_tests_failed": 0,
        "tb_ctrf": ctrf,
        "tb_ctrf_artifact": {
            "path": "verifier/ctrf.json",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    _write_json(
        campaign / "trials" / trial_id / "result.json",
        {
            "attempt": 1,
            "status": "completed",
            "uploaded": True,
            "experiment_run_id": base64.b64encode(b"ExperimentRun:1").decode(),
            "output": task_output,
            "grades": [{"name": "terminal-bench", "error": None}],
        },
    )
    experiment_id = base64.b64encode(b"Experiment:7").decode()
    _write_json(
        campaign / "lock.json",
        {
            "trials": [{"trial_id": trial_id, "task_id": "task-a"}],
            "server": {"cells": {"cell-test": {"experiment_id": experiment_id}}},
        },
    )
    _write_json(
        campaign / "result.json",
        {"status": "completed", "counts": {"completed": 1}},
    )
    _write_json(
        output_root / "rerun-plan.json",
        {
            "total_trials": 1,
            "models": [
                {
                    "model": "glm-5.3",
                    "task_count": 1,
                    "task_ids": ["task-a"],
                    "runs_dir": str(runs_dir),
                }
            ],
        },
    )
    connection = sqlite3.connect(output_root / "a2e.db")
    connection.executescript(
        """
        CREATE TABLE experiment_runs (
            id INTEGER PRIMARY KEY,
            experiment_id INTEGER,
            output TEXT,
            error TEXT
        );
        CREATE TABLE experiment_run_annotations (
            id INTEGER PRIMARY KEY,
            experiment_run_id INTEGER,
            name TEXT,
            error TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO experiment_runs (id, experiment_id, output) VALUES (1, 7, ?)",
        (json.dumps({"task_output": task_output}),),
    )
    connection.execute(
        "INSERT INTO experiment_run_annotations "
        "(experiment_run_id, name) VALUES (1, 'terminal-bench')"
    )
    connection.commit()
    connection.close()

    status = _MODULE.validate(argparse.Namespace(output=output_root))

    assert status == 0
    report = json.loads((output_root / "ctrf-rerun-report.json").read_text())
    assert report["status"] == "passed"
    assert report["database_complete_ctrf"] == 1
    assert (output_root / "CTRF_COMPLETE").is_file()

    targets = output_root / "glm-api-reruns" / "test" / "targets.json"
    _write_json(
        targets,
        {
            "campaign": str(campaign),
            "targets": [
                {"trial_id": trial_id, "task_id": "task-a", "attempt_before": 0}
            ],
        },
    )
    rerun_status = _MODULE.validate_glm_api_rerun(
        argparse.Namespace(output=output_root, targets=targets)
    )
    assert rerun_status == 0
    assert (targets.parent / "COMPLETE").is_file()


def test_select_glm_api_failures_refuses_mixed_failure_types(tmp_path: Path) -> None:
    output = tmp_path / "rerun"
    runs_dir = output / "campaigns" / "glm-5-3"
    campaign = runs_dir / "campaign-test"
    trials = [
        {"trial_id": "trial-api", "task_id": "api-task"},
        {"trial_id": "trial-timeout", "task_id": "timeout-task"},
    ]
    _write_json(campaign / "lock.json", {"trials": trials})
    _write_json(
        campaign / "trials" / "trial-api" / "result.json",
        {
            "attempt": 1,
            "status": "failed",
            "error": "503 No available accounts: no available accounts",
        },
    )
    _write_json(
        campaign / "trials" / "trial-timeout" / "result.json",
        {"attempt": 1, "status": "failed", "error": "agent timed out after 900s"},
    )
    _write_json(
        output / "rerun-plan.json",
        {
            "models": [
                {
                    "model": "glm-5.3",
                    "runs_dir": str(runs_dir),
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="other failed Trials"):
        _MODULE.select_glm_api_failures(
            argparse.Namespace(
                output=output,
                targets=output / "glm-api-reruns" / "targets.json",
            )
        )
