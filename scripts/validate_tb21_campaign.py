#!/usr/bin/env python3
"""Validate a completed TB2.1 Campaign and its concurrency accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationOptions:
    campaign_dir: Path
    database: Path
    expected_tasks_per_model: int
    expected_models: tuple[str, ...]
    expected_limits: dict[str, int]
    require_saturated: frozenset[str]
    require_exercised: frozenset[str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _parse_assignment(value: str) -> tuple[str, int]:
    name, separator, raw_limit = value.partition("=")
    if not separator or not name or not raw_limit.isdigit() or int(raw_limit) <= 0:
        raise argparse.ArgumentTypeError("expected RESOURCE=POSITIVE_INTEGER")
    return name, int(raw_limit)


def _validate_ctrf_artifact(
    campaign_dir: Path,
    trial_id: str,
    result: dict[str, Any],
    output: dict[str, Any],
    errors: list[str],
) -> None:
    ctrf = output.get("tb_ctrf")
    artifact = output.get("tb_ctrf_artifact")
    if not isinstance(ctrf, dict):
        errors.append(f"{trial_id} does not contain the complete parsed CTRF")
        return
    if not isinstance(artifact, dict):
        errors.append(f"{trial_id} does not contain CTRF artifact metadata")
        return
    relative_path = artifact.get("path")
    if relative_path != "verifier/ctrf.json":
        errors.append(f"{trial_id} has invalid CTRF artifact path: {relative_path!r}")
        return
    attempt = result.get("attempt")
    if not isinstance(attempt, int) or attempt <= 0:
        errors.append(f"{trial_id} has invalid attempt for CTRF artifact: {attempt!r}")
        return
    path = campaign_dir / "trials" / trial_id / "attempts" / str(attempt) / relative_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        errors.append(f"{trial_id} cannot read CTRF artifact {path}: {exc}")
        return
    if artifact.get("size_bytes") != len(raw):
        errors.append(f"{trial_id} CTRF artifact size does not match metadata")
    digest = hashlib.sha256(raw).hexdigest()
    if artifact.get("sha256") != digest:
        errors.append(f"{trial_id} CTRF artifact digest does not match metadata")
    try:
        persisted = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{trial_id} CTRF artifact is invalid JSON: {exc}")
        return
    if persisted != ctrf:
        errors.append(f"{trial_id} CTRF artifact differs from ExperimentRun output")


def _load_trial_results(
    campaign_dir: Path,
    lock: dict[str, Any],
    cells_by_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for model in {str(cell["model"]) for cell in cells_by_id.values()}:
        models[model] = {
            "trials": 0,
            "completed": 0,
            "uploaded": 0,
            "resolved": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
        }

    for trial in lock.get("trials") or []:
        trial_id = str(trial.get("trial_id") or "")
        cell_id = str(trial.get("cell_id") or "")
        if not trial_id or cell_id not in cells_by_id:
            errors.append(f"invalid locked Trial entry: {trial!r}")
            continue
        path = campaign_dir / "trials" / trial_id / "result.json"
        try:
            result = _read_json(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        results[trial_id] = result
        model = str(cells_by_id[cell_id]["model"])
        summary = models[model]
        summary["trials"] += 1
        if result.get("status") == "completed":
            summary["completed"] += 1
        if result.get("uploaded") is True:
            summary["uploaded"] += 1
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        if output.get("resolved") is True:
            summary["resolved"] += 1
        _validate_ctrf_artifact(campaign_dir, trial_id, result, output, errors)
        reward = output.get("tb_reward")
        if isinstance(reward, int | float) and not isinstance(reward, bool):
            summary["reward_sum"] += float(reward)
            summary["reward_count"] += 1
        grades = result.get("grades") if isinstance(result.get("grades"), list) else []
        terminal_grades = [
            grade
            for grade in grades
            if isinstance(grade, dict) and grade.get("name") == "terminal-bench"
        ]
        if len(terminal_grades) != 1 or terminal_grades[0].get("error"):
            errors.append(
                f"{trial_id} does not contain one successful terminal-bench grade"
            )

    for summary in models.values():
        count = int(summary.pop("reward_count"))
        total = float(summary.pop("reward_sum"))
        summary["average_reward"] = total / count if count else None
    return results, models


def _validate_database(
    database: Path,
    experiment_ids: list[Any],
    expected_total: int,
    errors: list[str],
) -> dict[str, int]:
    counts = {"experiment_runs": 0, "terminal_bench_evaluations": 0}
    if not database.is_file():
        errors.append(f"database does not exist: {database}")
        return counts
    if len(experiment_ids) != 2:
        errors.append(f"expected 2 Server experiments, found {len(experiment_ids)}")
        return counts
    placeholders = ",".join("?" for _ in experiment_ids)
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        try:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                errors.append("SQLite integrity_check failed")
            counts["experiment_runs"] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM experiment_runs WHERE experiment_id IN ({placeholders})",
                    experiment_ids,
                ).fetchone()[0]
            )
            counts["terminal_bench_evaluations"] = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM experiment_run_annotations AS annotation
                    JOIN experiment_runs AS run ON run.id = annotation.experiment_run_id
                    WHERE run.experiment_id IN ({placeholders})
                      AND annotation.name = 'terminal-bench'
                    """,
                    experiment_ids,
                ).fetchone()[0]
            )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        errors.append(f"cannot validate SQLite database: {exc}")
        return counts
    if counts["experiment_runs"] != expected_total:
        errors.append(
            f"expected {expected_total} ExperimentRun rows, found {counts['experiment_runs']}"
        )
    if counts["terminal_bench_evaluations"] != expected_total:
        errors.append(
            "expected "
            f"{expected_total} terminal-bench evaluations, "
            f"found {counts['terminal_bench_evaluations']}"
        )
    return counts


def validate_campaign(options: ValidationOptions) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    result = _read_json(options.campaign_dir / "result.json")
    lock = _read_json(options.campaign_dir / "lock.json")
    expected_total = options.expected_tasks_per_model * len(options.expected_models)

    if result.get("status") != "completed":
        errors.append(
            f"Campaign status is {result.get('status')!r}, expected 'completed'"
        )
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    if counts.get("completed") != expected_total:
        errors.append(
            f"Campaign reports {counts.get('completed', 0)} completed Trials; "
            f"expected {expected_total}"
        )
    unexpected = {
        name: value for name, value in counts.items() if name != "completed" and value
    }
    if unexpected:
        errors.append(f"Campaign contains non-completed Trial states: {unexpected}")

    cells = lock.get("cells") if isinstance(lock.get("cells"), list) else []
    cells_by_id = {
        str(cell.get("cell_id")): cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_id")
    }
    models_in_lock = tuple(
        sorted(str(cell.get("model")) for cell in cells_by_id.values())
    )
    if models_in_lock != tuple(sorted(options.expected_models)):
        errors.append(
            f"expected one Cell for models {sorted(options.expected_models)}, "
            f"found {list(models_in_lock)}"
        )
    trials = lock.get("trials") if isinstance(lock.get("trials"), list) else []
    if len(trials) != expected_total:
        errors.append(f"lock contains {len(trials)} Trials; expected {expected_total}")
    for cell_id, cell in cells_by_id.items():
        cell_trials = sum(1 for trial in trials if trial.get("cell_id") == cell_id)
        if cell_trials != options.expected_tasks_per_model:
            errors.append(
                f"Cell {cell_id} ({cell.get('model')}) contains {cell_trials} Trials; "
                f"expected {options.expected_tasks_per_model}"
            )

    trial_results, model_summaries = _load_trial_results(
        options.campaign_dir, lock, cells_by_id, errors
    )
    if len(trial_results) != expected_total:
        errors.append(
            f"found {len(trial_results)} Trial results; expected {expected_total}"
        )
    for model, summary in model_summaries.items():
        if summary["completed"] != options.expected_tasks_per_model:
            errors.append(
                f"model {model} completed {summary['completed']} Trials; "
                f"expected {options.expected_tasks_per_model}"
            )
        if summary["uploaded"] != options.expected_tasks_per_model:
            errors.append(
                f"model {model} uploaded {summary['uploaded']} Trials; "
                f"expected {options.expected_tasks_per_model}"
            )

    concurrency = (
        result.get("concurrency") if isinstance(result.get("concurrency"), dict) else {}
    )
    concurrency_rows: list[dict[str, Any]] = []
    for resource, expected_limit in options.expected_limits.items():
        stats = concurrency.get(resource)
        if not isinstance(stats, dict):
            errors.append(f"missing concurrency stats for {resource}")
            continue
        limit = stats.get("limit")
        active = stats.get("active")
        high_water = stats.get("high_water")
        row_errors: list[str] = []
        if limit != expected_limit:
            row_errors.append(f"limit={limit}, expected {expected_limit}")
        if active != 0:
            row_errors.append(f"active={active}, expected 0")
        if not isinstance(high_water, int) or not isinstance(limit, int):
            row_errors.append("limit/high_water are not integers")
        elif high_water > limit:
            row_errors.append(f"high_water={high_water} exceeds limit={limit}")
        if resource in options.require_saturated and high_water != limit:
            row_errors.append(
                f"high_water={high_water}, expected saturation at {limit}"
            )
        if resource in options.require_exercised and (
            not isinstance(high_water, int) or high_water <= 0
        ):
            row_errors.append("pool was not exercised")
        errors.extend(f"{resource}: {message}" for message in row_errors)
        concurrency_rows.append(
            {
                "resource": resource,
                "limit": limit,
                "active": active,
                "high_water": high_water,
                "saturated": high_water == limit,
                "status": "failed" if row_errors else "passed",
            }
        )
    unexpected_pools = sorted(set(concurrency) - set(options.expected_limits))
    if unexpected_pools:
        warnings.append(f"unvalidated concurrency pools: {unexpected_pools}")

    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    processes = (
        runtime.get("trial_processes")
        if isinstance(runtime.get("trial_processes"), dict)
        else {}
    )
    if processes.get("active") != 0:
        errors.append(
            f"Trial process active={processes.get('active')}, expected 0"
        )
    process_high_water = processes.get("high_water")
    global_limit = options.expected_limits.get("global")
    if not isinstance(process_high_water, int) or process_high_water <= 1:
        errors.append("Trial process high-water does not prove concurrent execution")
    elif global_limit is not None and process_high_water > global_limit:
        errors.append(
            f"Trial process high-water {process_high_water} exceeds global limit {global_limit}"
        )
    activities = (
        runtime.get("activities")
        if isinstance(runtime.get("activities"), dict)
        else {}
    )
    for name, stats in activities.items():
        if isinstance(stats, dict) and stats.get("active") != 0:
            errors.append(f"runtime activity {name} did not return to zero")
    docker_exec = activities.get("docker:exec")
    if not isinstance(docker_exec, dict) or not docker_exec.get("started"):
        errors.append("runtime did not observe any Docker exec activity")

    gateways = result.get("gateways") if isinstance(result.get("gateways"), dict) else {}
    for model in options.expected_models:
        metrics = gateways.get(model)
        if not isinstance(metrics, dict):
            errors.append(f"missing Gateway metrics for {model}")
            continue
        if metrics.get("inflight_requests") != 0:
            errors.append(f"Gateway {model} still has inflight requests")
        if not isinstance(metrics.get("inflight_high_water"), int):
            errors.append(f"Gateway {model} has no inflight high-water metric")

    server_cells_value = (
        lock.get("server", {}).get("cells", {})
        if isinstance(lock.get("server"), dict)
        else {}
    )
    server_cells = server_cells_value if isinstance(server_cells_value, dict) else {}
    experiment_ids = [
        state.get("experiment_id")
        for state in server_cells.values()
        if isinstance(state, dict) and state.get("experiment_id") is not None
    ]
    database_counts = _validate_database(
        options.database, experiment_ids, expected_total, errors
    )
    return {
        "status": "passed" if not errors else "failed",
        "campaign_id": lock.get("campaign_id"),
        "expected_trials": expected_total,
        "campaign_counts": counts,
        "database_counts": database_counts,
        "concurrency": concurrency_rows,
        "runtime": runtime,
        "gateways": gateways,
        "models": model_summaries,
        "errors": errors,
        "warnings": warnings,
    }


def _write_outputs(report: dict[str, Any], json_path: Path, tsv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["resource\tlimit\thigh_water\tactive\tsaturated\tstatus"]
    for row in report["concurrency"]:
        lines.append(
            "\t".join(
                str(row[key])
                for key in (
                    "resource",
                    "limit",
                    "high_water",
                    "active",
                    "saturated",
                    "status",
                )
            )
        )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_summary(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["model\ttrials\tcompleted\tuploaded\tresolved\taverage_reward"]
    for model, summary in sorted(report["models"].items()):
        values = [
            model,
            summary["trials"],
            summary["completed"],
            summary["uploaded"],
            summary["resolved"],
            "" if summary["average_reward"] is None else summary["average_reward"],
        ]
        lines.append("\t".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-tasks-per-model", type=int, required=True)
    parser.add_argument("--expected-model", action="append", required=True)
    parser.add_argument(
        "--expect-limit", action="append", type=_parse_assignment, required=True
    )
    parser.add_argument("--require-saturated", action="append", default=[])
    parser.add_argument("--require-exercised", action="append", default=[])
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    parser.add_argument("--model-summary-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.expected_tasks_per_model <= 0:
        parser.error("--expected-tasks-per-model must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    options = ValidationOptions(
        campaign_dir=args.campaign_dir.resolve(),
        database=args.database.resolve(),
        expected_tasks_per_model=args.expected_tasks_per_model,
        expected_models=tuple(args.expected_model),
        expected_limits=dict(args.expect_limit),
        require_saturated=frozenset(args.require_saturated),
        require_exercised=frozenset(args.require_exercised),
    )
    try:
        report = validate_campaign(options)
    except (TypeError, ValueError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    _write_outputs(report, args.json_output, args.tsv_output)
    _write_model_summary(report, args.model_summary_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
