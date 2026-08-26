#!/usr/bin/env python3
"""Prepare and validate exact model/task reruns for missing TB2.1 CTRF."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

EXPECTED_MODELS = ("glm-5.3", "gpt-5.6-sol")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing generated file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_immutable(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def campaign_directory(source: Path) -> Path:
    if (source / "lock.json").is_file() and (source / "result.json").is_file():
        return source
    candidates = sorted((source / "campaigns").glob("campaign-*"))
    candidates = [path for path in candidates if (path / "lock.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one source Campaign under {source}, found {len(candidates)}"
        )
    return candidates[0]


def valid_ctrf_summary(output: dict[str, Any]) -> bool:
    total = output.get("tb_tests_total")
    passed = output.get("tb_tests_passed")
    failed = output.get("tb_tests_failed")
    return (
        output.get("tb_ctrf_error") is None
        and isinstance(total, int)
        and total > 0
        and isinstance(passed, int)
        and isinstance(failed, int)
        and passed + failed == total
    )


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def missing_pairs(source_campaign: Path) -> dict[str, list[dict[str, str]]]:
    lock = read_json(source_campaign / "lock.json")
    cells = {
        str(cell["cell_id"]): str(cell["model"])
        for cell in lock.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_id") and cell.get("model")
    }
    unknown = set(cells.values()) - set(EXPECTED_MODELS)
    if unknown:
        raise RuntimeError(f"unexpected source models: {sorted(unknown)}")
    selected: dict[str, list[dict[str, str]]] = {
        model: [] for model in EXPECTED_MODELS
    }
    for trial in lock.get("trials", []):
        if not isinstance(trial, dict):
            continue
        trial_id = str(trial.get("trial_id") or "")
        cell_id = str(trial.get("cell_id") or "")
        task_id = str(trial.get("task_id") or "")
        if not trial_id or cell_id not in cells or not task_id:
            raise RuntimeError(f"invalid source Trial lock entry: {trial!r}")
        result = read_json(source_campaign / "trials" / trial_id / "result.json")
        output = result.get("output")
        output = output if isinstance(output, dict) else {}
        if valid_ctrf_summary(output):
            continue
        error = str(result.get("error") or output.get("tb_ctrf_error") or "missing CTRF")
        selected[cells[cell_id]].append(
            {
                "task_id": task_id,
                "source_trial_id": trial_id,
                "source_status": str(result.get("status") or "unknown"),
                "reason": error[:500],
            }
        )
    return {model: rows for model, rows in selected.items() if rows}


def prepare(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    source_campaign = campaign_directory(source)
    pairs = missing_pairs(source_campaign)
    if not pairs:
        raise RuntimeError("source Campaign has no missing or invalid CTRF summaries")

    models: list[dict[str, Any]] = []
    manifest_lines = ["model\ttask_id\tsource_trial_id\tsource_status\treason"]
    for model in EXPECTED_MODELS:
        rows = pairs.get(model, [])
        if not rows:
            continue
        model_slug = slug(model)
        concurrency = min(args.model_concurrency, len(rows))
        config_root = output / "config" / model_slug
        runs_dir = output / "campaigns" / model_slug
        source_profile = repo / "task" / "models" / f"{model}.yaml"
        profile = yaml.safe_load(source_profile.read_text(encoding="utf-8"))
        profile["concurrency"]["max_sessions"] = concurrency
        profile_path = config_root / "models" / f"{model}.yaml"
        write_immutable(
            profile_path,
            yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
        )
        task_ids = [row["task_id"] for row in rows]
        campaign = {
            "schema_version": 1,
            "name": f"tb21-missing-ctrf-{model_slug}",
            "models": [model],
            "benchmarks": [
                {
                    "id": "terminal-bench-2.1",
                    "sample": {"task_ids": task_ids},
                    "graders": [
                        {"id": "terminal-bench", "mode": "inline", "required": True}
                    ],
                }
            ],
            "harnesses": ["claude-sdk"],
            "repetitions": 1,
            "matrix": {"exclude": []},
            "execution": {
                "n_concurrent_trials": concurrency,
                "n_active_cells": 1,
                "n_concurrent_sandboxes": concurrency,
                "n_concurrent_model_sessions": concurrency,
                "n_concurrent_graders": concurrency,
                "n_concurrent_uploads": min(8, concurrency),
                "queue_capacity": concurrency,
                "cancellation_grace_seconds": 30,
                "timeout_seconds": None,
                "retry": {"max_retries": 0},
            },
            "artifacts": {"retain": "all"},
        }
        config_path = config_root / "campaign.yaml"
        write_immutable(
            config_path,
            yaml.safe_dump(campaign, sort_keys=False, allow_unicode=True),
        )
        models.append(
            {
                "model": model,
                "slug": model_slug,
                "task_count": len(task_ids),
                "task_ids": task_ids,
                "concurrency": concurrency,
                "config": str(config_path),
                "models_dir": str(profile_path.parent),
                "runs_dir": str(runs_dir),
                "log": str(output / f"campaign-{model_slug}.log"),
            }
        )
        for row in rows:
            manifest_lines.append(
                "\t".join(
                    (
                        model,
                        row["task_id"],
                        row["source_trial_id"],
                        row["source_status"],
                        row["reason"].replace("\t", " ").replace("\n", " "),
                    )
                )
            )

    plan = {
        "schema_version": 1,
        "source_root": str(source),
        "source_campaign": str(source_campaign),
        "output_root": str(output),
        "total_trials": sum(item["task_count"] for item in models),
        "models": models,
    }
    write_json(output / "rerun-plan.json", plan)
    write_immutable(output / "source-manifest.tsv", "\n".join(manifest_lines) + "\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def decode_global_id(value: Any, expected_kind: str) -> int:
    if isinstance(value, int):
        return value
    raw = str(value)
    if raw.isdigit():
        return int(raw)
    padding = "=" * (-len(raw) % 4)
    try:
        decoded = base64.b64decode(raw + padding).decode("utf-8")
        kind, identifier = decoded.rsplit(":", 1)
        if kind == expected_kind:
            return int(identifier)
    except (ValueError, UnicodeDecodeError):
        pass
    raise ValueError(f"cannot decode {expected_kind} id: {value!r}")


def decode_relay_id(value: Any) -> int:
    return decode_global_id(value, "Experiment")


def model_campaign(output: Path, model_name: str) -> tuple[dict[str, Any], Path]:
    plan = read_json(output / "rerun-plan.json")
    matches = [model for model in plan.get("models", []) if model.get("model") == model_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {model_name} entry in rerun plan")
    model = matches[0]
    campaigns = sorted(Path(model["runs_dir"]).glob("campaign-*"))
    if len(campaigns) != 1:
        raise RuntimeError(
            f"{model_name}: expected one Campaign, found {len(campaigns)}"
        )
    return model, campaigns[0]


def select_glm_api_failures(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    _model, campaign = model_campaign(output, "glm-5.3")
    lock = read_json(campaign / "lock.json")
    targets: list[dict[str, Any]] = []
    other_failures: list[str] = []
    for trial in lock.get("trials", []):
        if not isinstance(trial, dict):
            continue
        trial_id = str(trial.get("trial_id") or "")
        result = read_json(campaign / "trials" / trial_id / "result.json")
        if result.get("status") != "failed":
            continue
        error = str(result.get("error") or "")
        if "503" not in error or "No available accounts" not in error:
            other_failures.append(f"{trial_id}: {error[:300]}")
            continue
        attempt = result.get("attempt")
        if not isinstance(attempt, int) or attempt <= 0:
            raise RuntimeError(f"{trial_id} has invalid attempt: {attempt!r}")
        targets.append(
            {
                "trial_id": trial_id,
                "task_id": str(trial.get("task_id") or result.get("task_id") or ""),
                "attempt_before": attempt,
                "error_before": error,
            }
        )
    if other_failures:
        raise RuntimeError(
            "GLM Campaign contains other failed Trials; --rerun-failed would also rerun "
            "them:\n" + "\n".join(other_failures)
        )
    payload = {
        "schema_version": 1,
        "output_root": str(output),
        "campaign": str(campaign),
        "failure_type": "upstream_503_no_available_accounts",
        "targets": targets,
    }
    write_json(args.targets.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def validate_glm_api_rerun(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    target_file = args.targets.resolve()
    target_plan = read_json(target_file)
    targets = target_plan.get("targets")
    targets = targets if isinstance(targets, list) else []
    errors: list[str] = []
    run_ids: list[int] = []
    campaign = Path(target_plan["campaign"])
    for target in targets:
        if not isinstance(target, dict):
            errors.append(f"invalid target entry: {target!r}")
            continue
        trial_id = str(target.get("trial_id") or "")
        result = read_json(campaign / "trials" / trial_id / "result.json")
        attempt = result.get("attempt")
        if not isinstance(attempt, int) or attempt <= int(target["attempt_before"]):
            errors.append(f"{trial_id} did not append a new attempt")
        validate_local_trial(campaign, {"trial_id": trial_id}, errors)
        if result.get("error") is not None:
            errors.append(f"{trial_id} still has an execution error: {result['error']}")
        run_id = result.get("experiment_run_id")
        if not run_id:
            errors.append(f"{trial_id} has no ExperimentRun id")
            continue
        try:
            run_ids.append(decode_global_id(run_id, "ExperimentRun"))
        except ValueError as exc:
            errors.append(f"{trial_id}: {exc}")

    database = output / "a2e.db"
    db_rows = 0
    db_complete_ctrf = 0
    db_evaluations = 0
    if not database.is_file():
        errors.append(f"rerun database does not exist: {database}")
    elif run_ids:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in run_ids)
            rows = connection.execute(
                f"SELECT id, output, error FROM experiment_runs "
                f"WHERE id IN ({placeholders})",
                run_ids,
            ).fetchall()
            db_rows = len(rows)
            for run_id, raw_output, error in rows:
                if error:
                    errors.append(f"ExperimentRun {run_id} still contains error: {error}")
                try:
                    task_output = json.loads(raw_output or "{}").get("task_output") or {}
                except json.JSONDecodeError:
                    task_output = {}
                if isinstance(task_output.get("tb_ctrf"), dict):
                    db_complete_ctrf += 1
                else:
                    errors.append(f"ExperimentRun {run_id} has no complete CTRF")
            db_evaluations = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM experiment_run_annotations AS annotation "
                    f"WHERE annotation.experiment_run_id IN ({placeholders}) "
                    "AND annotation.name='terminal-bench' AND annotation.error IS NULL",
                    run_ids,
                ).fetchone()[0]
            )
        finally:
            connection.close()

    expected = len(targets)
    if len(run_ids) != expected:
        errors.append(f"expected {expected} ExperimentRun ids, found {len(run_ids)}")
    if db_rows != expected:
        errors.append(f"expected {expected} database rows, found {db_rows}")
    if db_complete_ctrf != expected:
        errors.append(f"expected {expected} database CTRFs, found {db_complete_ctrf}")
    if db_evaluations != expected:
        errors.append(f"expected {expected} evaluations, found {db_evaluations}")

    report = {
        "status": "passed" if not errors else "failed",
        "target_trials": expected,
        "database_runs": db_rows,
        "database_complete_ctrf": db_complete_ctrf,
        "database_evaluations": db_evaluations,
        "errors": errors,
    }
    report_path = target_file.parent / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not errors:
        (target_file.parent / "COMPLETE").touch()
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def validate_local_trial(campaign: Path, trial: dict[str, Any], errors: list[str]) -> None:
    trial_id = str(trial["trial_id"])
    result = read_json(campaign / "trials" / trial_id / "result.json")
    if result.get("status") != "completed" or result.get("uploaded") is not True:
        errors.append(f"{trial_id} is not completed and uploaded")
    output = result.get("output")
    output = output if isinstance(output, dict) else {}
    if not valid_ctrf_summary(output):
        errors.append(f"{trial_id} does not contain a valid CTRF summary")
    ctrf = output.get("tb_ctrf")
    artifact = output.get("tb_ctrf_artifact")
    if not isinstance(ctrf, dict) or not isinstance(artifact, dict):
        errors.append(f"{trial_id} does not contain complete CTRF and artifact metadata")
        return
    if artifact.get("path") != "verifier/ctrf.json":
        errors.append(f"{trial_id} has an invalid CTRF artifact path")
        return
    attempt = result.get("attempt")
    path = (
        campaign
        / "trials"
        / trial_id
        / "attempts"
        / str(attempt)
        / "verifier"
        / "ctrf.json"
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        errors.append(f"{trial_id} cannot read raw CTRF: {exc}")
        return
    if len(raw) != artifact.get("size_bytes"):
        errors.append(f"{trial_id} raw CTRF size mismatch")
    if hashlib.sha256(raw).hexdigest() != artifact.get("sha256"):
        errors.append(f"{trial_id} raw CTRF digest mismatch")
    try:
        persisted = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{trial_id} raw CTRF is invalid JSON: {exc}")
    else:
        if persisted != ctrf:
            errors.append(f"{trial_id} raw CTRF differs from Trial output")
    grades = result.get("grades")
    grades = grades if isinstance(grades, list) else []
    terminal = [
        grade
        for grade in grades
        if isinstance(grade, dict) and grade.get("name") == "terminal-bench"
    ]
    if len(terminal) != 1 or terminal[0].get("error"):
        errors.append(f"{trial_id} does not contain one successful terminal-bench grade")


def validate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    plan = read_json(output / "rerun-plan.json")
    errors: list[str] = []
    experiment_ids: list[int] = []
    local_trials = 0
    for model in plan.get("models", []):
        runs_dir = Path(model["runs_dir"])
        campaigns = sorted(runs_dir.glob("campaign-*"))
        if len(campaigns) != 1:
            errors.append(f"{model['model']}: expected one Campaign, found {len(campaigns)}")
            continue
        campaign = campaigns[0]
        result = read_json(campaign / "result.json")
        expected = int(model["task_count"])
        if result.get("status") != "completed" or result.get("counts") != {
            "completed": expected
        }:
            errors.append(f"{model['model']}: Campaign did not complete {expected} Trials")
        lock = read_json(campaign / "lock.json")
        trials = lock.get("trials") if isinstance(lock.get("trials"), list) else []
        if len(trials) != expected:
            errors.append(f"{model['model']}: lock contains {len(trials)} Trials")
        locked_tasks = {str(trial.get("task_id")) for trial in trials}
        if locked_tasks != set(model["task_ids"]):
            errors.append(f"{model['model']}: locked task IDs differ from rerun plan")
        for trial in trials:
            validate_local_trial(campaign, trial, errors)
            local_trials += 1
        server = lock.get("server")
        server = server if isinstance(server, dict) else {}
        server_cells_by_id = server.get("cells")
        server_cells_by_id = (
            server_cells_by_id if isinstance(server_cells_by_id, dict) else {}
        )
        server_cells = server_cells_by_id.values()
        for cell in server_cells:
            if not isinstance(cell, dict) or not cell.get("experiment_id"):
                errors.append(f"{model['model']}: missing Server Experiment id")
                continue
            try:
                experiment_ids.append(decode_relay_id(cell["experiment_id"]))
            except ValueError as exc:
                errors.append(f"{model['model']}: {exc}")

    database = output / "a2e.db"
    db_runs = 0
    db_evaluations = 0
    db_full_ctrf = 0
    if not database.is_file():
        errors.append(f"rerun database does not exist: {database}")
    elif experiment_ids:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in experiment_ids)
            rows = connection.execute(
                f"SELECT id, output, error FROM experiment_runs "
                f"WHERE experiment_id IN ({placeholders})",
                experiment_ids,
            ).fetchall()
            db_runs = len(rows)
            for _run_id, raw_output, error in rows:
                if error:
                    errors.append(f"database ExperimentRun contains error: {error}")
                try:
                    task_output = json.loads(raw_output or "{}").get("task_output") or {}
                except json.JSONDecodeError:
                    task_output = {}
                if isinstance(task_output.get("tb_ctrf"), dict):
                    db_full_ctrf += 1
            db_evaluations = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM experiment_run_annotations AS annotation "
                    f"JOIN experiment_runs AS run ON run.id=annotation.experiment_run_id "
                    f"WHERE run.experiment_id IN ({placeholders}) "
                    "AND annotation.name='terminal-bench' AND annotation.error IS NULL",
                    experiment_ids,
                ).fetchone()[0]
            )
        finally:
            connection.close()

    expected_total = int(plan["total_trials"])
    if local_trials != expected_total:
        errors.append(f"expected {expected_total} local results, found {local_trials}")
    if db_runs != expected_total:
        errors.append(f"expected {expected_total} ExperimentRuns, found {db_runs}")
    if db_evaluations != expected_total:
        errors.append(f"expected {expected_total} evaluations, found {db_evaluations}")
    if db_full_ctrf != expected_total:
        errors.append(f"expected {expected_total} complete DB CTRFs, found {db_full_ctrf}")

    report = {
        "status": "passed" if not errors else "failed",
        "expected_trials": expected_total,
        "local_trials": local_trials,
        "database_runs": db_runs,
        "database_evaluations": db_evaluations,
        "database_complete_ctrf": db_full_ctrf,
        "errors": errors,
    }
    report_path = output / "ctrf-rerun-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not errors:
        (output / "CTRF_COMPLETE").touch()
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo", type=Path, required=True)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--model-concurrency", type=int, default=16)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    select_parser = subparsers.add_parser("select-glm-api-failures")
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--targets", type=Path, required=True)
    validate_glm_parser = subparsers.add_parser("validate-glm-api-rerun")
    validate_glm_parser.add_argument("--output", type=Path, required=True)
    validate_glm_parser.add_argument("--targets", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "model_concurrency", 1) <= 0:
        parser.error("--model-concurrency must be positive")
    return args


def main() -> int:
    args = parse_args()
    commands = {
        "prepare": prepare,
        "validate": validate,
        "select-glm-api-failures": select_glm_api_failures,
        "validate-glm-api-rerun": validate_glm_api_rerun,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
