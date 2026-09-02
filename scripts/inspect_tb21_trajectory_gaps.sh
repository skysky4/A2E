#!/usr/bin/env bash
# Show ONE complete TB21 trajectory: dataset ref + task_output + span timeline.
#
# Usage:
#   bash scripts/inspect_tb21_trajectory_gaps.sh --db a2e-tb21-gpt-5.6-sol-agno.db
#   bash scripts/inspect_tb21_trajectory_gaps.sh --db a2e-tb21-gpt-5.6-sol-agno.db --run-id 3
#   bash scripts/inspect_tb21_trajectory_gaps.sh --summary   # aggregate stats (old mode)
#
# Env:
#   RUN_ID=1              default experiment_run.id

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-1}"
DB_PATH=""
MODE="full"
SPAN_ATTR_LIMIT="${SPAN_ATTR_LIMIT:-1200}"

usage() {
  cat <<EOF
Show one complete trajectory from a TB21 A2E SQLite DB.

Examples:
  bash scripts/inspect_tb21_trajectory_gaps.sh --db $ROOT/a2e-tb21-gpt-5.6-sol-agno.db
  bash scripts/inspect_tb21_trajectory_gaps.sh --db $ROOT/a2e-tb21-gpt-5.6-sol-agno.db --run-id 5
  bash scripts/inspect_tb21_trajectory_gaps.sh --summary --db $ROOT/a2e-tb21-gpt-5.6-sol-agno.db

Default DB (if --db omitted): $ROOT/a2e-tb21-gpt-5.6-sol-agno.db
Default run-id: 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --summary)
      MODE="summary"
      shift
      ;;
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --db)
      DB_PATH="${2:?missing value for --db}"
      shift 2
      ;;
    *)
      if [[ -z "$DB_PATH" ]]; then
        DB_PATH="$1"
      else
        echo "error: unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$DB_PATH" ]]; then
  DB_PATH="$ROOT/a2e-tb21-gpt-5.6-sol-agno.db"
fi
if [[ "$DB_PATH" != /* ]]; then
  DB_PATH="$ROOT/$DB_PATH"
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "error: db not found: $DB_PATH" >&2
  exit 1
fi

command -v python3 >/dev/null || { echo "error: python3 required" >&2; exit 1; }

if [[ "$MODE" == "summary" ]]; then
  exec bash "$ROOT/scripts/inspect_tb21_trajectory_summary.sh" --db "$DB_PATH" --run-id "$RUN_ID"
fi

export DB_PATH RUN_ID SPAN_ATTR_LIMIT ROOT
python3 <<'PY'
import json
import os
import sqlite3
import textwrap
from datetime import datetime

DB = os.environ["DB_PATH"]
RUN_ID = int(os.environ["RUN_ID"])
ATTR_LIMIT = int(os.environ.get("SPAN_ATTR_LIMIT", "1200"))


def pp(obj, indent=2):
    print(json.dumps(obj, indent=indent, ensure_ascii=False, default=str))


def trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def span_summary(span_kind: str, name: str, attrs: dict) -> str:
    lines = []
    if span_kind == "TOOL":
        tool = attrs.get("tool") or {}
        inp = attrs.get("input") or {}
        out = attrs.get("output") or {}
        lines.append(f"tool.name={tool.get('name')!r}")
        if tool.get("description"):
            lines.append(f"tool.description={trunc(str(tool.get('description')), 200)!r}")
        if tool.get("parameters"):
            lines.append(f"tool.parameters={trunc(json.dumps(tool.get('parameters'), ensure_ascii=False), 300)}")
        if inp.get("value"):
            lines.append(f"input.value={trunc(str(inp.get('value')), 400)!r}")
        if out.get("value"):
            lines.append(f"output.value={trunc(str(out.get('value')), 400)!r}")
    elif span_kind == "LLM":
        llm = attrs.get("llm") or {}
        tc = llm.get("token_count") or {}
        if llm.get("model_name"):
            lines.append(f"model={llm.get('model_name')!r}")
        if tc:
            lines.append(f"token_count={tc}")
        tools = llm.get("tools") or []
        if tools:
            names = []
            for t in tools[:8]:
                td = t.get("tool") if isinstance(t, dict) else t
                if isinstance(td, dict):
                    names.append(td.get("name") or (json.loads(td.get("json_schema") or "{}") if isinstance(td.get("json_schema"), str) else {}).get("function", {}).get("name"))
                elif isinstance(t, dict):
                    names.append(t.get("name"))
            lines.append(f"tools={[n for n in names if n]!r}")
        msgs = llm.get("output_messages") or llm.get("input_messages") or []
        if msgs:
            lines.append(f"messages={len(msgs)}")
    elif span_kind == "EVALUATOR":
        lines.append(trunc(json.dumps(attrs, ensure_ascii=False), 300))
    else:
        lines.append(trunc(json.dumps(attrs, ensure_ascii=False), 400))
    return "\n      ".join(lines) if lines else "(empty attributes)"


conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

row = conn.execute(
    """
    SELECT er.id AS run_id, er.trace_id, er.dataset_example_id, er.start_time, er.end_time,
           er.prompt_token_count, er.completion_token_count, er.output AS run_output,
           e.name AS experiment_name,
           der.input AS ref_input, der.output AS ref_output, der.metadata AS ref_metadata
    FROM experiment_runs er
    JOIN experiments e ON e.id = er.experiment_id
    LEFT JOIN dataset_example_revisions der ON der.dataset_example_id = er.dataset_example_id
    WHERE er.id = ?
    """,
    (RUN_ID,),
).fetchone()

if not row:
    runs = [r[0] for r in conn.execute("SELECT id FROM experiment_runs ORDER BY id LIMIT 10")]
    raise SystemExit(f"run id {RUN_ID} not found. try one of: {runs}")

ref_in = json.loads(row["ref_input"] or "{}")
ref_out = json.loads(row["ref_output"] or "{}")
ref_meta = json.loads(row["ref_metadata"] or "{}")
run_out = json.loads(row["run_output"] or "{}")
task_output = run_out.get("task_output") or run_out
trace_id = row["trace_id"]

print("=" * 72)
print("FULL TRAJECTORY")
print("=" * 72)
print(f"db:           {DB}")
print(f"experiment:   {row['experiment_name']}")
print(f"run_id:       {row['run_id']}")
print(f"example_id:   {row['dataset_example_id']}")
print(f"trace_id:     {trace_id or '(none)'}")
print(f"run window:   {row['start_time']} -> {row['end_time']}")
print(f"run tokens:   prompt={row['prompt_token_count']} completion={row['completion_token_count']}")
print()

print("-" * 72)
print("1) DATASET INPUT (what the task asks)")
print("-" * 72)
pp(ref_in)
print()

print("-" * 72)
print("2) DATASET REFERENCE OUTPUT (ground truth for eval — tool_recall reads this)")
print("-" * 72)
pp(ref_out)
actions = ref_out.get("expected_actions") or []
outputs = ref_out.get("expected_outputs") or []
if not actions and not outputs:
    print(">>> expected_actions / expected_outputs are EMPTY — tool_recall cannot score")
print()

if ref_meta:
    print("-" * 72)
    print("3) DATASET METADATA")
    print("-" * 72)
    pp(ref_meta)
    print()

print("-" * 72)
print("4) HARNESS task_output (what the run recorded)")
print("-" * 72)
pp(task_output)
missing = [k for k in ("elapsed_time", "elapsed_seconds", "duration", "cost", "total_token_usage", "token_usage", "trace_id") if task_output.get(k) in (None, "", [])]
if missing:
    print(f">>> missing in task_output (eval may derive from spans): {missing}")
print()

print("-" * 72)
print("5) FULL experiment_runs.output JSON")
print("-" * 72)
pp(run_out)
print()

if trace_id:
    spans = conn.execute(
        """
        SELECT s.span_kind, s.name, s.start_time, s.end_time, s.attributes,
               s.llm_token_count_prompt, s.llm_token_count_completion
        FROM spans s
        JOIN traces t ON t.rowid = s.trace_rowid
        WHERE t.trace_id = ?
        ORDER BY s.start_time, s.rowid
        """,
        (trace_id,),
    ).fetchall()
else:
    spans = []

print("-" * 72)
print(f"6) SPAN TIMELINE ({len(spans)} spans, chronological)")
print("-" * 72)
if not trace_id:
    print(">>> no trace_id on experiment_run — span timeline unavailable")
elif not spans:
    print(">>> trace_id present but no spans in DB")
else:
    for i, sp in enumerate(spans, 1):
        attrs = json.loads(sp["attributes"] or "{}")
        kind = sp["span_kind"] or "?"
        print(f"[{i:02d}] {sp['start_time']} -> {sp['end_time']}  {kind:9}  {sp['name']}")
        if sp["llm_token_count_prompt"] is not None or sp["llm_token_count_completion"] is not None:
            print(f"      db_token_cols: prompt={sp['llm_token_count_prompt']} completion={sp['llm_token_count_completion']}")
        print(f"      {span_summary(kind, sp['name'], attrs)}")
        raw = json.dumps(attrs, ensure_ascii=False)
        if len(raw) > ATTR_LIMIT:
            print(f"      (full attributes truncated, {len(raw)} chars; set SPAN_ATTR_LIMIT to increase)")
        print()

print("-" * 72)
print("7) EVAL ANNOTATIONS on this run")
print("-" * 72)
ann = conn.execute(
    """
    SELECT name, label, score, substr(explanation, 1, 160) AS explanation
    FROM experiment_run_annotations
    WHERE experiment_run_id = ?
    ORDER BY name
    """,
    (RUN_ID,),
).fetchall()
if not ann:
    print("(none)")
else:
    unscored = [a for a in ann if a["label"] == "unscored" or a["score"] is None]
    print(f"total {len(ann)} annotations, unscored {len(unscored)}")
    for a in ann:
        flag = " [UNSCORED]" if a["label"] == "unscored" or a["score"] is None else ""
        print(f"  {a['name']:28} score={a['score']} label={a['label']}{flag}")
        if flag and a["explanation"]:
            print(f"    {a['explanation']}")
print()
print("=" * 72)
print("Tip: another run →  bash scripts/inspect_tb21_trajectory_gaps.sh --db ... --run-id N")
print("Tip: batch stats  →  bash scripts/inspect_tb21_trajectory_gaps.sh --summary --db ...")
PY
