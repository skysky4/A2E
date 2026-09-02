#!/usr/bin/env python3
"""Backfill run-level token fields + total_token_usage from trace spans.

UI reads sample-level ``experiment_runs.prompt_token_count`` /
``completion_token_count`` first; many harnesses (e.g. Agno) only record
tokens on LLM spans. This script:

1. Sums ``llm.token_count.*`` from spans (via span_store flattening).
2. Writes ``experiment_runs.prompt_token_count`` / ``completion_token_count``.
3. Updates ``total_token_usage`` annotation when tokens can be derived.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.eval_common import (  # noqa: E402
    _numeric_attr,
    _span_attributes,
    _task_output,
    _token_cost_spans,
    _token_usage_from_task_output,
)
from core.span_store import fetch_spans_by_trace_id  # noqa: E402


def _sum_prompt_completion(spans: list[dict[str, Any]]) -> tuple[float, float, int]:
    prompt_total = 0.0
    completion_total = 0.0
    used = 0
    for span in _token_cost_spans(spans):
        attrs = _span_attributes(span)
        prompt = _numeric_attr(attrs, "llm.token_count.prompt")
        completion = _numeric_attr(attrs, "llm.token_count.completion")
        if prompt is None and completion is None:
            total = _numeric_attr(attrs, "llm.token_count.total")
            if total is not None:
                prompt = total
                completion = 0.0
        if prompt is None and completion is None:
            continue
        prompt_total += float(prompt or 0)
        completion_total += float(completion or 0)
        used += 1
    return prompt_total, completion_total, used


def _json_meta(existing: Any, **extra: Any) -> str:
    meta: dict[str, Any] = {}
    if isinstance(existing, str) and existing.strip():
        try:
            meta = json.loads(existing)
        except json.JSONDecodeError:
            meta = {"previous_metadata_raw": existing}
    elif isinstance(existing, dict):
        meta = dict(existing)
    meta.update(extra)
    meta["backfilled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return json.dumps(meta, ensure_ascii=False)


def _tokens_from_root_cumulative(conn: sqlite3.Connection, trace_id: str) -> tuple[int, int, str] | None:
    if not trace_id:
        return None
    row = conn.execute(
        """
        SELECT cumulative_llm_token_count_prompt, cumulative_llm_token_count_completion
        FROM spans s
        JOIN traces t ON t.rowid = s.trace_rowid
        WHERE t.trace_id = ?
        ORDER BY (
          COALESCE(s.cumulative_llm_token_count_prompt, 0)
          + COALESCE(s.cumulative_llm_token_count_completion, 0)
        ) DESC, s.start_time DESC
        LIMIT 1
        """,
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    prompt = int(row[0] or 0)
    completion = int(row[1] or 0)
    if prompt + completion <= 0:
        return None
    return prompt, completion, "trace_root.cumulative_llm_token_count"


def _label_total_tokens(total: float) -> str:
    if total < 2000:
        return "low"
    if total < 10000:
        return "medium"
    return "high"


def backfill_db(db_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    stats = {
        "runs": 0,
        "run_tokens_updated": 0,
        "annotation_updated": 0,
        "annotation_already_scored": 0,
        "still_unscored": 0,
    }

    runs = conn.execute(
        """
        SELECT id, trace_id, dataset_example_id, output,
               prompt_token_count, completion_token_count
        FROM experiment_runs
        ORDER BY id
        """
    ).fetchall()
    stats["runs"] = len(runs)

    for run in runs:
        run_id = int(run["id"])
        trace_id = str(run["trace_id"] or "")
        example_id = str(run["dataset_example_id"] or "")
        spans = fetch_spans_by_trace_id(str(db_path), trace_id, limit=100_000) if trace_id else []

        prompt_sum, completion_sum, used_spans = _sum_prompt_completion(spans)
        total_from_spans = prompt_sum + completion_sum

        output_dict = _task_output(run["output"])
        reported = _token_usage_from_task_output(output_dict)
        total_from_output = reported[0] if reported else None

        root_tokens = _tokens_from_root_cumulative(conn, trace_id)

        # Prefer root cumulative (Agno-style cumulative telemetry), then incremental
        # span sums, then harness output field.
        if root_tokens is not None:
            prompt_val, completion_val, token_source = root_tokens
            total_tokens = float(prompt_val + completion_val)
        elif total_from_spans > 0:
            prompt_val = int(round(prompt_sum))
            completion_val = int(round(completion_sum))
            total_tokens = total_from_spans
            token_source = f"span_sum({used_spans} spans)"
        elif total_from_output is not None and total_from_output > 0:
            prompt_val = None
            completion_val = None
            total_tokens = float(total_from_output)
            token_source = reported[1] if reported else "output"
        else:
            prompt_val = None
            completion_val = None
            total_tokens = 0.0
            token_source = ""

        run_needs_update = prompt_val is not None and (
            (run["prompt_token_count"] or 0) != prompt_val
            or (run["completion_token_count"] or 0) != completion_val
        )
        if run_needs_update:
            if not dry_run:
                conn.execute(
                    """
                    UPDATE experiment_runs
                    SET prompt_token_count = ?, completion_token_count = ?
                    WHERE id = ?
                    """,
                    (prompt_val, completion_val, run_id),
                )
            stats["run_tokens_updated"] += 1

        ann = conn.execute(
            """
            SELECT id, score, label, metadata
            FROM experiment_run_annotations
            WHERE experiment_run_id = ? AND name = 'total_token_usage'
            """,
            (run_id,),
        ).fetchone()

        if total_tokens <= 0:
            stats["still_unscored"] += 1
            continue

        explanation = f"{total_tokens:.0f} tokens; {token_source}"
        label = _label_total_tokens(total_tokens)
        ann_needs_update = (
            ann is not None
            and (
                ann["score"] is None
                or str(ann["label"] or "").lower() == "unscored"
                or abs(float(ann["score"] or 0) - total_tokens) > 0.5
            )
        )
        if ann is not None and ann["score"] is not None and not ann_needs_update:
            stats["annotation_already_scored"] += 1
            continue
        if ann is None:
            stats["still_unscored"] += 1
            continue
        if not ann_needs_update:
            stats["annotation_already_scored"] += 1
            continue

        meta = _json_meta(
            ann["metadata"],
            source="backfill_total_token_usage_tb21",
            token_source=token_source,
        )
        if not dry_run:
            conn.execute(
                """
                UPDATE experiment_run_annotations
                SET score = ?, label = ?, explanation = ?, error = NULL, metadata = ?
                WHERE id = ?
                """,
                (float(total_tokens), label, explanation, meta, ann["id"]),
            )
        stats["annotation_updated"] += 1

    if not dry_run:
        conn.commit()
    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="SQLite DB path (local writable copy)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stats = backfill_db(args.db, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
