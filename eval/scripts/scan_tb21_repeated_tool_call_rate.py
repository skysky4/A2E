#!/usr/bin/env python3
"""Score repeated_tool_call_rate on local TB21 DBs (no writeback)."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.eval_common import _as_dict, _tool_calls_from_spans
from process_values.tool_eval import _locate_loop_windows, _loop_call_record

DBS = {
    "glm-5.3/agno": "/tmp/agno_v2.db",
    "glm-5.3/autogen-agentchat": "/home/yuchenyue/scratch_tb21_rtcr/glm-autogen.db",
    "glm-5.3/crewai": "/tmp/crewai_tb21.db",
    "glm-5.3/google-adk": "/home/yuchenyue/scratch_tb21_rtcr/glm-adk.db",
    "glm-5.3/langgraph": "/home/yuchenyue/scratch_tb21_rtcr/glm-langgraph.db",
    "glm-5.3/llama-index": "/tmp/llama_glm.db",
    "glm-5.3/openai-agents": "/home/yuchenyue/scratch_tb21_rtcr/glm-openai.db",
    "glm-5.3/smolagents": "/tmp/smol_tb21.db",
    "gpt-5.6-sol/agno": "/home/yuchenyue/scratch_tb21_rtcr/gpt-agno.db",
    "gpt-5.6-sol/autogen-agentchat": "/home/yuchenyue/scratch_tb21_rtcr/gpt-autogen.db",
    "gpt-5.6-sol/crewai": "/tmp/crewai_gpt.db",
    "gpt-5.6-sol/google-adk": "/home/yuchenyue/scratch_tb21_rtcr/gpt-adk.db",
    "gpt-5.6-sol/langgraph": "/home/yuchenyue/scratch_tb21_rtcr/gpt-langgraph.db",
    "gpt-5.6-sol/llama-index": "/home/yuchenyue/scratch_tb21_rtcr/gpt-llama.db",
    "gpt-5.6-sol/openai-agents": "/home/yuchenyue/scratch_tb21_rtcr/gpt-openai.db",
    "gpt-5.6-sol/smolagents": "/tmp/smol_v2.db",
}


def _loads(raw: object) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _collapse_consecutive_clones(calls: list[dict]) -> list[dict]:
    collapsed: list[dict] = []
    prev_key = None
    for call in calls:
        item = _as_dict(call)
        key = (
            str(item.get("name") or ""),
            json.dumps(item.get("arguments") or {}, sort_keys=True, default=str),
            json.dumps(item.get("result"), sort_keys=True, default=str)[:800],
        )
        if key == prev_key:
            continue
        collapsed.append(item)
        prev_key = key
    return collapsed


def _fetch_tool_spans(conn: sqlite3.Connection, trace_id: str | None) -> list[dict]:
    if not trace_id:
        return []
    rows = conn.execute(
        """
        SELECT s.span_kind, s.name, s.start_time, s.attributes
        FROM spans s
        JOIN traces t ON t.id = s.trace_rowid
        WHERE t.trace_id = ? AND UPPER(s.span_kind) = 'TOOL'
        ORDER BY s.start_time
        """,
        (trace_id,),
    ).fetchall()
    spans = []
    for kind, name, start, attrs in rows:
        spans.append(
            {
                "span_kind": kind,
                "name": name,
                "start_time": start,
                "attributes": _loads(attrs),
            }
        )
    return spans


def main() -> None:
    label_all: Counter[str] = Counter()
    by_correct: dict[str, Counter[str]] = {"correct": Counter(), "incorrect": Counter(), "other": Counter()}
    per_db: list[str] = []
    examples: dict[str, list[dict]] = defaultdict(list)

    for label, db in DBS.items():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT r.id, r.trace_id, r.output, corr.label AS corr_label
            FROM experiment_runs r
            LEFT JOIN experiment_run_annotations corr
              ON corr.experiment_run_id = r.id AND corr.name = 'correctness'
            """
        ).fetchall()
        db_labels: Counter[str] = Counter()
        n_span_tool = 0
        n_collapsed = 0
        for row in rows:
            output = _loads(row["output"])
            task = output.get("task_output") if isinstance(output.get("task_output"), dict) else output
            spans = _fetch_tool_spans(conn, row["trace_id"])
            n_span_tool += len(spans)
            calls = _tool_calls_from_spans(spans)
            collapsed = _collapse_consecutive_clones(calls)
            n_collapsed += len(collapsed)
            records = [rec for rec in (_loop_call_record(call) for call in collapsed) if rec]
            if not records:
                lab = "unscored"
            elif _locate_loop_windows(records):
                lab = "candidate"
            else:
                lab = "clean"
            label_all[lab] += 1
            db_labels[lab] += 1
            corr = str(row["corr_label"] or "other")
            if corr not in by_correct:
                corr = "other"
            by_correct[corr][lab] += 1
            if lab == "candidate" and len(examples[label]) < 2:
                examples[label].append(
                    {
                        "run_id": row["id"],
                        "correctness": row["corr_label"],
                        "n_spans": len(spans),
                        "n_collapsed": len(collapsed),
                        "windows": _locate_loop_windows(records),
                    }
                )
        conn.close()
        line = (
            f"{label}\tn={len(rows)}\t{dict(db_labels)}\t"
            f"tool_spans={n_span_tool} collapsed_calls={n_collapsed}"
        )
        per_db.append(line)
        print(line, flush=True)

    def _mean(xs: list[float]) -> str:
        if not xs:
            return "n/a"
        return f"{sum(xs) / len(xs):.3f} (n={len(xs)})"

    print("\n==== LABEL TOTALS")
    for k, v in label_all.most_common():
        print(f"{k}\t{v}")
    print("\n==== LABEL x correctness")
    for corr, ctr in by_correct.items():
        print(corr, dict(ctr))
    print("\n==== EXAMPLES (locator candidates)")
    for db_label, exs in examples.items():
        for ex in exs:
            print(db_label, json.dumps(ex, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
