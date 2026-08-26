#!/usr/bin/env python3
"""Merge exact TB2.1 Claude SDK reruns into a copied full-run database."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = ("glm-5.3", "gpt-5.6-sol")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _schema_signature(connection: sqlite3.Connection) -> dict[str, Any]:
    """Compare SQLite schemas structurally, ignoring constraint declaration order."""
    signature: dict[str, Any] = {}
    for table in sorted(_table_names(connection)):
        columns = [tuple(row) for row in connection.execute(f"PRAGMA table_info({table})")]
        foreign_keys = sorted(
            tuple(row)[2:]
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        indexes = []
        for index in connection.execute(f"PRAGMA index_list({table})"):
            index_name = str(index[1])
            indexes.append(
                (
                    index_name,
                    int(index[2]),
                    str(index[3]),
                    int(index[4]),
                    tuple(
                        tuple(row)
                        for row in connection.execute(f'PRAGMA index_info("{index_name}")')
                    ),
                )
            )
        signature[table] = (columns, foreign_keys, sorted(indexes))
    return signature


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]


def _insert_row(
    connection: sqlite3.Connection,
    table: str,
    row: sqlite3.Row | dict[str, Any],
    *,
    exclude: set[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> int:
    values = dict(row)
    for name in exclude or set():
        values.pop(name, None)
    values.update(overrides or {})
    columns = list(values)
    placeholders = ",".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [values[column] for column in columns],
    )
    return int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])


def _experiments_by_model(connection: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    matches: dict[str, sqlite3.Row] = {}
    for row in connection.execute("SELECT * FROM experiments ORDER BY id"):
        metadata = _json(row["metadata"])
        if not isinstance(metadata, dict):
            continue
        if metadata.get("benchmark") != "terminal-bench-2.1":
            continue
        if metadata.get("harness") != "claude-sdk":
            continue
        model = str(metadata.get("model_profile") or "")
        if model not in MODELS:
            continue
        if model in matches:
            raise RuntimeError(f"database contains multiple Claude SDK Experiments for {model}")
        matches[model] = row
    if set(matches) != set(MODELS):
        raise RuntimeError(
            f"expected Claude SDK Experiments for {MODELS}, found {sorted(matches)}"
        )
    return matches


def _examples_by_task(
    connection: sqlite3.Connection, experiment: sqlite3.Row
) -> dict[str, int]:
    rows = connection.execute(
        "SELECT revision.dataset_example_id, revision.metadata "
        "FROM dataset_example_revisions AS revision "
        "JOIN dataset_examples AS example "
        "ON example.id=revision.dataset_example_id "
        "WHERE example.dataset_id=? AND revision.dataset_version_id=?",
        (experiment["dataset_id"], experiment["dataset_version_id"]),
    )
    result: dict[str, int] = {}
    for row in rows:
        metadata = _json(row["metadata"])
        task_id = str((metadata or {}).get("task_id") or "")
        if not task_id or task_id in result:
            raise RuntimeError(f"invalid or duplicate task_id in Dataset: {task_id!r}")
        result[task_id] = int(row["dataset_example_id"])
    return result


def _runs_by_task(
    connection: sqlite3.Connection, experiment: sqlite3.Row
) -> dict[tuple[str, int], sqlite3.Row]:
    examples = _examples_by_task(connection, experiment)
    task_by_example = {example: task for task, example in examples.items()}
    result: dict[tuple[str, int], sqlite3.Row] = {}
    for row in connection.execute(
        "SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY id",
        (experiment["id"],),
    ):
        example_id = int(row["dataset_example_id"])
        if example_id not in task_by_example:
            raise RuntimeError(f"Run {row['id']} references an unknown Dataset example")
        key = (task_by_example[example_id], int(row["repetition_number"]))
        if key in result:
            raise RuntimeError(f"duplicate ExperimentRun key: {key}")
        result[key] = row
    return result


def _copy_referenced_trace(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    trace_id: str,
    *,
    project_ids: dict[int, int],
    trace_ids: dict[str, str],
    source_label: str,
) -> str:
    if trace_id in trace_ids:
        return trace_ids[trace_id]
    if destination.execute(
        "SELECT 1 FROM traces WHERE trace_id=?", (trace_id,)
    ).fetchone():
        raise RuntimeError(f"trace_id collision while merging: {trace_id}")
    trace = source.execute(
        "SELECT * FROM traces WHERE trace_id=?", (trace_id,)
    ).fetchone()
    if trace is None:
        raise RuntimeError(f"patch ExperimentRun references missing trace {trace_id}")
    if trace["project_session_rowid"] is not None:
        raise RuntimeError("project-session trace merging is not supported")
    old_project_id = int(trace["project_rowid"])
    if old_project_id not in project_ids:
        project = source.execute(
            "SELECT * FROM projects WHERE id=?", (old_project_id,)
        ).fetchone()
        if project is None:
            raise RuntimeError(f"trace references missing project {old_project_id}")
        project_values = dict(project)
        original_name = str(project_values["name"])
        if destination.execute(
            "SELECT 1 FROM projects WHERE name=?", (original_name,)
        ).fetchone():
            suffix = hashlib.sha256(
                f"{source_label}:{old_project_id}:{original_name}".encode()
            ).hexdigest()[:8]
            project_values["name"] = f"{original_name}-merged-{suffix}"
        project_ids[old_project_id] = _insert_row(
            destination, "projects", project_values, exclude={"id"}
        )
    new_trace_rowid = _insert_row(
        destination,
        "traces",
        trace,
        exclude={"id"},
        overrides={
            "project_rowid": project_ids[old_project_id],
            "project_session_rowid": None,
        },
    )
    for span in source.execute(
        "SELECT * FROM spans WHERE trace_rowid=? ORDER BY id", (trace["id"],)
    ):
        source_span_id = int(span["id"])
        for dependent in ("span_annotations", "document_annotations", "span_costs"):
            if source.execute(
                f"SELECT 1 FROM {dependent} WHERE span_rowid=? LIMIT 1",
                (source_span_id,),
            ).fetchone():
                raise RuntimeError(f"cannot merge non-empty {dependent} for trace {trace_id}")
        _insert_row(
            destination,
            "spans",
            span,
            exclude={"id"},
            overrides={"trace_rowid": new_trace_rowid},
        )
    if source.execute(
        "SELECT 1 FROM trace_annotations WHERE trace_rowid=? LIMIT 1", (trace["id"],)
    ).fetchone():
        raise RuntimeError(f"cannot merge trace annotations for {trace_id}")
    trace_ids[trace_id] = trace_id
    return trace_id


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = _table_names(connection)
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "datasets",
            "dataset_versions",
            "dataset_examples",
            "experiments",
            "experiment_runs",
            "experiment_run_annotations",
            "projects",
            "traces",
            "spans",
        )
        if table in tables
    }


def merge(base_path: Path, patch_path: Path, output_path: Path) -> dict[str, Any]:
    base_path = base_path.resolve()
    patch_path = patch_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"output database already exists: {output_path}")
    if not base_path.is_file() or not patch_path.is_file():
        raise FileNotFoundError("base and patch databases must both exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = _connect_readonly(base_path)
    patch = _connect_readonly(patch_path)
    destination: sqlite3.Connection | None = None
    try:
        if _schema_signature(base) != _schema_signature(patch):
            raise RuntimeError("database schemas differ; refusing an unsafe merge")
        base_version = base.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        patch_version = patch.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        if base_version != patch_version:
            raise RuntimeError(
                f"Alembic versions differ: base={base_version}, patch={patch_version}"
            )
        destination = sqlite3.connect(output_path)
        base.backup(destination)
        destination.row_factory = sqlite3.Row
        destination.execute("PRAGMA foreign_keys=ON")
        before = _counts(destination)

        base_experiments = _experiments_by_model(destination)
        patch_experiments = _experiments_by_model(patch)
        replacements: list[dict[str, Any]] = []
        project_ids: dict[int, int] = {}
        trace_ids: dict[str, str] = {}
        expected_annotation_count = before["experiment_run_annotations"]

        destination.execute("BEGIN IMMEDIATE")
        try:
            for model in MODELS:
                base_runs = _runs_by_task(destination, base_experiments[model])
                patch_runs = _runs_by_task(patch, patch_experiments[model])
                for key, patch_run in patch_runs.items():
                    task_id, repetition = key
                    base_run = base_runs.get(key)
                    if base_run is None:
                        raise RuntimeError(f"base database has no Run for {model}/{key}")
                    task_output = (_json(patch_run["output"]) or {}).get("task_output") or {}
                    if not isinstance(task_output.get("tb_ctrf"), dict):
                        raise RuntimeError(f"patch Run lacks complete CTRF: {model}/{task_id}")
                    trace_id = str(patch_run["trace_id"] or "")
                    if not trace_id:
                        raise RuntimeError(f"patch Run lacks trace_id: {model}/{task_id}")
                    merged_trace_id = _copy_referenced_trace(
                        patch,
                        destination,
                        trace_id,
                        project_ids=project_ids,
                        trace_ids=trace_ids,
                        source_label=str(patch_path),
                    )
                    old_annotations = int(
                        destination.execute(
                            "SELECT COUNT(*) FROM experiment_run_annotations "
                            "WHERE experiment_run_id=?",
                            (base_run["id"],),
                        ).fetchone()[0]
                    )
                    patch_annotations = list(
                        patch.execute(
                            "SELECT * FROM experiment_run_annotations "
                            "WHERE experiment_run_id=? ORDER BY id",
                            (patch_run["id"],),
                        )
                    )
                    if not patch_annotations:
                        raise RuntimeError(f"patch Run lacks evaluation: {model}/{task_id}")
                    destination.execute(
                        "UPDATE experiment_runs SET trace_id=?, output=?, start_time=?, "
                        "end_time=?, prompt_token_count=?, completion_token_count=?, error=? "
                        "WHERE id=?",
                        (
                            merged_trace_id,
                            patch_run["output"],
                            patch_run["start_time"],
                            patch_run["end_time"],
                            patch_run["prompt_token_count"],
                            patch_run["completion_token_count"],
                            patch_run["error"],
                            base_run["id"],
                        ),
                    )
                    destination.execute(
                        "DELETE FROM experiment_run_annotations WHERE experiment_run_id=?",
                        (base_run["id"],),
                    )
                    for annotation in patch_annotations:
                        annotation_trace = annotation["trace_id"]
                        if annotation_trace:
                            annotation_trace = _copy_referenced_trace(
                                patch,
                                destination,
                                str(annotation_trace),
                                project_ids=project_ids,
                                trace_ids=trace_ids,
                                source_label=str(patch_path),
                            )
                        _insert_row(
                            destination,
                            "experiment_run_annotations",
                            annotation,
                            exclude={"id"},
                            overrides={
                                "experiment_run_id": base_run["id"],
                                "trace_id": annotation_trace,
                            },
                        )
                    expected_annotation_count += len(patch_annotations) - old_annotations
                    replacements.append(
                        {
                            "model": model,
                            "task_id": task_id,
                            "repetition": repetition,
                            "base_run_id": int(base_run["id"]),
                            "patch_run_id": int(patch_run["id"]),
                            "error_before": base_run["error"],
                            "error_after": patch_run["error"],
                            "score": patch_annotations[0]["score"],
                        }
                    )
            destination.commit()
        except Exception:
            destination.rollback()
            raise

        after = _counts(destination)
        if after["experiment_runs"] != before["experiment_runs"]:
            raise RuntimeError("ExperimentRun count changed during replacement merge")
        if after["experiments"] != before["experiments"]:
            raise RuntimeError("Experiment count changed during replacement merge")
        if after["experiment_run_annotations"] != expected_annotation_count:
            raise RuntimeError("evaluation count does not match replacement plan")
        integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = [tuple(row) for row in destination.execute("PRAGMA foreign_key_check")]
        if integrity != "ok" or foreign_keys:
            raise RuntimeError(
                f"merged database validation failed: integrity={integrity}, fks={foreign_keys}"
            )
        full_ctrf = int(
            destination.execute(
                "SELECT COUNT(*) FROM experiment_runs "
                "WHERE json_type(output, '$.task_output.tb_ctrf')='object'"
            ).fetchone()[0]
        )
        if full_ctrf != len(replacements):
            raise RuntimeError(
                f"expected {len(replacements)} complete CTRFs, found {full_ctrf}"
            )
        run_errors = int(
            destination.execute(
                "SELECT COUNT(*) FROM experiment_runs WHERE error IS NOT NULL"
            ).fetchone()[0]
        )
        model_summary: dict[str, Any] = {}
        for model, experiment in base_experiments.items():
            row = destination.execute(
                "SELECT COUNT(*) AS runs, SUM(run.error IS NULL) AS no_error, "
                "SUM(run.error IS NOT NULL) AS errors, "
                "SUM(json_type(run.output, '$.task_output.tb_ctrf')='object') AS complete_ctrf, "
                "SUM(annotation.score) AS reward_sum "
                "FROM experiment_runs AS run "
                "LEFT JOIN experiment_run_annotations AS annotation "
                "ON annotation.experiment_run_id=run.id "
                "AND annotation.name='terminal-bench' "
                "WHERE run.experiment_id=?",
                (experiment["id"],),
            ).fetchone()
            model_summary[model] = {
                "runs": int(row["runs"] or 0),
                "no_error": int(row["no_error"] or 0),
                "errors": int(row["errors"] or 0),
                "complete_ctrf": int(row["complete_ctrf"] or 0),
                "reward_sum": float(row["reward_sum"] or 0.0),
            }
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "copy_base_and_replace_by_model_task_repetition",
            "base_database": str(base_path),
            "base_sha256": _file_sha256(base_path),
            "patch_database": str(patch_path),
            "patch_sha256": _file_sha256(patch_path),
            "output_database": str(output_path),
            "alembic_version": str(base_version),
            "source_databases_unchanged": True,
            "before": before,
            "after": after,
            "replacement_count": len(replacements),
            "complete_ctrf_count": full_ctrf,
            "replacement_errors_remaining": sum(
                replacement["error_after"] is not None for replacement in replacements
            ),
            "total_run_errors": run_errors,
            "models": model_summary,
            "integrity_check": integrity,
            "foreign_key_violations": foreign_keys,
            "replacements": replacements,
        }
        report_path = output_path.with_suffix(".merge-report.json")
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report
    except Exception:
        if destination is not None:
            destination.close()
            destination = None
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        base.close()
        patch.close()
        if destination is not None:
            destination.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = merge(args.base, args.patch, args.output)
    summary = {
        key: report[key]
        for key in (
            "output_database",
            "replacement_count",
            "complete_ctrf_count",
            "replacement_errors_remaining",
            "integrity_check",
        )
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
