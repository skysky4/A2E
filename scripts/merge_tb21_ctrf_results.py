#!/usr/bin/env python3
"""Merge valid rerun results into a completed Terminal-Bench experiment.

The source databases are copied with SQLite's backup API.  Runs whose CTRF
result is missing or empty are then replaced, by (agent, task_id), with the
corresponding rerun result.  Neither input directory is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


RUN_COLUMNS = (
    "output",
    "start_time",
    "end_time",
    "prompt_token_count",
    "completion_token_count",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="full experiment root")
    parser.add_argument("rerun", type=Path, help="missing-CTRF rerun root")
    parser.add_argument("destination", type=Path, help="new merged root")
    return parser.parse_args()


def connect_readonly(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def primary_experiment(connection: sqlite3.Connection) -> tuple[int, int]:
    row = connection.execute(
        """
        SELECT experiment_id, COUNT(*) AS run_count
        FROM experiment_runs
        GROUP BY experiment_id
        ORDER BY run_count DESC, experiment_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("database contains no experiment runs")
    return int(row["experiment_id"]), int(row["run_count"])


def task_runs(
    connection: sqlite3.Connection, experiment_id: int
) -> dict[str, sqlite3.Row]:
    rows = connection.execute(
        """
        SELECT run.*, revision.metadata AS revision_metadata
        FROM experiment_runs AS run
        JOIN experiments_dataset_examples AS selected
          ON selected.experiment_id = run.experiment_id
         AND selected.dataset_example_id = run.dataset_example_id
        JOIN dataset_example_revisions AS revision
          ON revision.id = selected.dataset_example_revision_id
        WHERE run.experiment_id = ?
        ORDER BY run.id
        """,
        (experiment_id,),
    )
    result: dict[str, sqlite3.Row] = {}
    for row in rows:
        task_id = json.loads(row["revision_metadata"])["task_id"]
        if task_id in result:
            raise RuntimeError(f"duplicate task_id in experiment: {task_id}")
        result[task_id] = row
    return result


def task_output(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(row["output"])["task_output"]


def valid_ctrf(output: dict[str, Any]) -> bool:
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


def read_manifest(rerun_root: Path) -> list[tuple[str, str]]:
    manifest = rerun_root / "rerun-manifest.tsv"
    if not manifest.is_file():
        raise RuntimeError(f"missing rerun manifest: {manifest}")
    pairs: list[tuple[str, str]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise RuntimeError(f"invalid manifest line: {line!r}")
        pairs.append((fields[0], fields[1]))
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("rerun manifest contains duplicate pairs")
    return pairs


def replace_annotation(
    destination: sqlite3.Connection,
    source_run_id: int,
    rerun: sqlite3.Connection,
    rerun_run_id: int,
) -> None:
    rerun_annotations = list(
        rerun.execute(
            "SELECT * FROM experiment_run_annotations WHERE experiment_run_id = ?",
            (rerun_run_id,),
        )
    )
    source_annotations = list(
        destination.execute(
            "SELECT * FROM experiment_run_annotations WHERE experiment_run_id = ?",
            (source_run_id,),
        )
    )
    if len(rerun_annotations) != 1 or len(source_annotations) != 1:
        raise RuntimeError(
            "expected exactly one annotation for both source and rerun "
            f"(source={len(source_annotations)}, rerun={len(rerun_annotations)})"
        )
    old = source_annotations[0]
    new = rerun_annotations[0]
    fields = (
        "name",
        "annotator_kind",
        "label",
        "score",
        "explanation",
        "error",
        "metadata",
        "start_time",
        "end_time",
    )
    assignments = ", ".join(f"{field} = ?" for field in fields)
    destination.execute(
        f"UPDATE experiment_run_annotations SET {assignments}, trace_id = NULL "
        "WHERE id = ?",
        (*[new[field] for field in fields], old["id"]),
    )


def write_full_summary(
    destination: Path, pairs: set[tuple[str, str]]
) -> tuple[int, int, int]:
    fields = (
        "agent",
        "task",
        "result_source",
        "status",
        "resolved",
        "reward",
        "tests_passed",
        "tests_failed",
        "tests_total",
        "ctrf_error",
    )
    valid_count = 0
    resolved_count = 0
    total_count = 0
    with (destination / "ctrf-summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for database in sorted(destination.glob("*/a2e.db")):
            agent = database.parent.name
            connection = connect_readonly(database)
            try:
                experiment_id, run_count = primary_experiment(connection)
                if run_count != 81:
                    raise RuntimeError(
                        f"{agent}: expected 81 merged runs, found {run_count}"
                    )
                for task_id, row in sorted(
                    task_runs(connection, experiment_id).items()
                ):
                    output = task_output(row)
                    total_count += 1
                    valid_count += valid_ctrf(output)
                    resolved_count += output.get("resolved") is True
                    writer.writerow(
                        (
                            agent,
                            task_id,
                            "rerun" if (agent, task_id) in pairs else "original",
                            output.get("tb_status"),
                            output.get("resolved"),
                            output.get("tb_reward"),
                            output.get("tb_tests_passed"),
                            output.get("tb_tests_failed"),
                            output.get("tb_tests_total"),
                            output.get("tb_ctrf_error"),
                        )
                    )
            finally:
                connection.close()
    return total_count, valid_count, resolved_count


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    rerun = args.rerun.resolve()
    destination = args.destination.resolve()

    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")
    if not (rerun / "CTRF_COMPLETE").is_file():
        raise RuntimeError("rerun root does not contain CTRF_COMPLETE")

    manifest = read_manifest(rerun)
    pairs = set(manifest)
    agents = sorted({agent for agent, _ in manifest})
    if len(agents) != 8:
        raise RuntimeError(f"expected 8 agents in manifest, found {len(agents)}")

    # Validate all inputs before creating the destination.
    source_rows: dict[str, dict[str, sqlite3.Row]] = {}
    rerun_rows: dict[str, dict[str, sqlite3.Row]] = {}
    source_connections: dict[str, sqlite3.Connection] = {}
    rerun_connections: dict[str, sqlite3.Connection] = {}
    try:
        for agent in agents:
            source_db = source / agent / "a2e.db"
            rerun_db = rerun / agent / "a2e.db"
            if not source_db.is_file() or not rerun_db.is_file():
                raise RuntimeError(f"missing database for agent {agent}")
            source_connection = connect_readonly(source_db)
            rerun_connection = connect_readonly(rerun_db)
            source_connections[agent] = source_connection
            rerun_connections[agent] = rerun_connection
            source_experiment, source_count = primary_experiment(source_connection)
            rerun_experiment, _ = primary_experiment(rerun_connection)
            if source_count != 81:
                raise RuntimeError(
                    f"{agent}: expected 81 source runs, found {source_count}"
                )
            source_rows[agent] = task_runs(source_connection, source_experiment)
            rerun_rows[agent] = task_runs(rerun_connection, rerun_experiment)

        source_invalid = {
            (agent, task_id)
            for agent, rows in source_rows.items()
            for task_id, row in rows.items()
            if not valid_ctrf(task_output(row))
        }
        if source_invalid != pairs:
            raise RuntimeError(
                "manifest does not exactly match invalid source CTRF rows: "
                f"manifest_only={sorted(pairs - source_invalid)}, "
                f"source_only={sorted(source_invalid - pairs)}"
            )
        for agent, task_id in manifest:
            row = rerun_rows[agent].get(task_id)
            if row is None:
                raise RuntimeError(f"rerun result missing for {agent}/{task_id}")
            if not valid_ctrf(task_output(row)):
                raise RuntimeError(f"rerun CTRF is invalid for {agent}/{task_id}")

        destination.mkdir(parents=True)
        audit_rows: list[tuple[Any, ...]] = []
        for agent in agents:
            agent_destination = destination / agent
            agent_destination.mkdir()
            destination_db = agent_destination / "a2e.db"
            merged = sqlite3.connect(destination_db)
            merged.row_factory = sqlite3.Row
            try:
                source_connections[agent].backup(merged)
                for manifest_agent, task_id in manifest:
                    if manifest_agent != agent:
                        continue
                    old = source_rows[agent][task_id]
                    new = rerun_rows[agent][task_id]
                    new_output = task_output(new)
                    assignments = ", ".join(
                        f"{column} = ?" for column in RUN_COLUMNS
                    )
                    merged.execute(
                        f"UPDATE experiment_runs SET {assignments}, trace_id = NULL "
                        "WHERE id = ?",
                        (*[new[column] for column in RUN_COLUMNS], old["id"]),
                    )
                    replace_annotation(
                        merged,
                        int(old["id"]),
                        rerun_connections[agent],
                        int(new["id"]),
                    )
                    audit_rows.append(
                        (
                            agent,
                            task_id,
                            old["id"],
                            new["id"],
                            new_output.get("resolved"),
                            new_output.get("tb_reward"),
                            new_output.get("tb_tests_passed"),
                            new_output.get("tb_tests_failed"),
                            new_output.get("tb_tests_total"),
                        )
                    )
                merged.commit()
                integrity = merged.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise RuntimeError(f"{agent}: integrity_check returned {integrity}")
            finally:
                merged.close()

        with (destination / "replacement-manifest.tsv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                (
                    "agent",
                    "task",
                    "source_run_id",
                    "rerun_run_id",
                    "resolved",
                    "reward",
                    "tests_passed",
                    "tests_failed",
                    "tests_total",
                )
            )
            writer.writerows(audit_rows)

        total, valid, resolved = write_full_summary(destination, pairs)
        if total != 648 or valid != 648:
            raise RuntimeError(
                f"merged validation failed: total={total}, valid_ctrf={valid}"
            )
        metadata = {
            "source": str(source),
            "rerun": str(rerun),
            "destination": str(destination),
            "runs": total,
            "valid_ctrf": valid,
            "resolved": resolved,
            "replaced": len(audit_rows),
        }
        (destination / "merge-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (destination / "CTRF_COMPLETE").touch()
        (destination / "MERGE_COMPLETE").touch()
        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        for connection in source_connections.values():
            connection.close()
        for connection in rerun_connections.values():
            connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        raise
