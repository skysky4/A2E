#!/usr/bin/env bash
# Dump one raw TB21/A2E trajectory from SQLite (read-only, no eval annotations).
#
# Usage:
#   bash scripts/dump_raw_trajectory.sh
#   bash scripts/dump_raw_trajectory.sh --db a2e-tb21-gpt-5.6-sol-google-adk\(1\).db --run-id 1
#   bash scripts/dump_raw_trajectory.sh --db path/to.db --run-id 5 --out /tmp/run5.json
#
# Output: single JSON object to stdout (or --out file): run fields, dataset ref, spans verbatim.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH=""
RUN_ID="${RUN_ID:-1}"
OUT_PATH=""

usage() {
  cat <<EOF
Usage: bash scripts/dump_raw_trajectory.sh [options]

Options:
  --db PATH       SQLite DB (default: $ROOT/a2e-tb21-gpt-5.6-sol-google-adk(1).db if present)
  --run-id N      experiment_runs.id (default: 1)
  --out PATH      write JSON to file instead of stdout
  -h, --help      show this help

Reads only: experiment_runs, dataset_example_revisions, traces, spans.
Does NOT include experiment_run_annotations or any eval labels.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --run-id) RUN_ID="${2:?missing value for --run-id}"; shift 2 ;;
    --out) OUT_PATH="${2:?missing value for --out}"; shift 2 ;;
    --db) DB_PATH="${2:?missing value for --db}"; shift 2 ;;
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
  if [[ -f "$ROOT/a2e-tb21-gpt-5.6-sol-google-adk(1).db" ]]; then
    DB_PATH="$ROOT/a2e-tb21-gpt-5.6-sol-google-adk(1).db"
  else
    DB_PATH="$ROOT/a2e-tb21-gpt-5.6-sol-google-adk.db"
  fi
fi
[[ "$DB_PATH" != /* ]] && DB_PATH="$ROOT/$DB_PATH"

if [[ ! -f "$DB_PATH" ]]; then
  echo "error: db not found: $DB_PATH" >&2
  exit 1
fi

command -v python3 >/dev/null || { echo "error: python3 required" >&2; exit 1; }

export DB_PATH RUN_ID OUT_PATH
python3 <<'PY'
import json
import os
import sqlite3
import sys

db = os.environ["DB_PATH"]
run_id = int(os.environ["RUN_ID"])
out_path = os.environ.get("OUT_PATH") or ""

con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
cur = con.cursor()

cur.execute(
    """
    SELECT er.id AS run_id, er.trace_id, er.start_time, er.end_time,
           er.prompt_token_count, er.completion_token_count, er.error, er.output,
           der.input AS dataset_input, der.output AS dataset_output,
           der.metadata AS dataset_metadata
    FROM experiment_runs er
    JOIN dataset_example_revisions der ON der.dataset_example_id = er.dataset_example_id
    WHERE er.id = ?
    LIMIT 1
    """,
    (run_id,),
)
run = cur.fetchone()
if not run:
    print(f"error: run_id={run_id} not found in {db}", file=sys.stderr)
    sys.exit(1)

cur.execute(
    """
    SELECT s.span_id, s.parent_id, s.name, s.span_kind, s.start_time, s.end_time,
           s.attributes, s.events, s.status_code, s.status_message,
           s.llm_token_count_prompt, s.llm_token_count_completion
    FROM spans s
    JOIN traces t ON t.id = s.trace_rowid
    WHERE t.trace_id = ?
    ORDER BY s.start_time, s.id
    """,
    (run["trace_id"],),
)
spans = [dict(row) for row in cur.fetchall()]

payload = {
    "source_db": db,
    "run_id": run["run_id"],
    "trace_id": run["trace_id"],
    "start_time": run["start_time"],
    "end_time": run["end_time"],
    "prompt_token_count": run["prompt_token_count"],
    "completion_token_count": run["completion_token_count"],
    "error": run["error"],
    "output": json.loads(run["output"]) if run["output"] else None,
    "dataset_input": json.loads(run["dataset_input"]) if run["dataset_input"] else None,
    "dataset_output": json.loads(run["dataset_output"]) if run["dataset_output"] else None,
    "dataset_metadata": json.loads(run["dataset_metadata"]) if run["dataset_metadata"] else None,
    "spans": [
        {
            **{k: span[k] for k in span if k not in ("attributes", "events")},
            "attributes": json.loads(span["attributes"]) if span["attributes"] else {},
            "events": json.loads(span["events"]) if span["events"] else [],
        }
        for span in spans
    ],
}

text = json.dumps(payload, ensure_ascii=False, indent=2)
if out_path:
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
else:
    sys.stdout.write(text)
    sys.stdout.write("\n")

con.close()
PY
