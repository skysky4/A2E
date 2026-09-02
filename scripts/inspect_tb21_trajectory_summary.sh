#!/usr/bin/env bash
# Aggregate TB21 trajectory / eval gap stats (companion to inspect_tb21_trajectory_gaps.sh --summary).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
DB_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?}"; shift 2 ;;
    --db) DB_PATH="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done

[[ -n "$DB_PATH" ]] || DB_PATH="$ROOT/a2e-tb21-gpt-5.6-sol-agno.db"
[[ "$DB_PATH" != /* ]] && DB_PATH="$ROOT/$DB_PATH"
[[ -f "$DB_PATH" ]] || { echo "error: db not found: $DB_PATH" >&2; exit 1; }

echo "SUMMARY: $DB_PATH"
sqlite3 -column -header "$DB_PATH" "SELECT id, name FROM experiments;"
echo
sqlite3 -column -header "$DB_PATH" <<'SQL'
SELECT name,
       SUM(CASE WHEN label='unscored' OR score IS NULL THEN 1 ELSE 0 END) AS unscored,
       COUNT(*) AS total
FROM experiment_run_annotations
WHERE name != 'tb_resolved'
GROUP BY name
HAVING unscored > 0
ORDER BY unscored DESC, name;
SQL

python3 - "$DB_PATH" <<'PY'
import json, sqlite3, sys
db = sys.argv[1]
conn = sqlite3.connect(db)
rows = conn.execute("SELECT output FROM dataset_example_revisions").fetchall()
empty = sum(1 for (o,) in rows if not (json.loads(o or "{}").get("expected_actions") or []))
print(f"\ndataset expected_actions empty: {empty}/{len(rows)}")
rows = conn.execute("SELECT output FROM experiment_runs").fetchall()
fields = ["elapsed_time", "cost", "total_token_usage", "tool_calls", "trace_id"]
for f in fields:
    n = sum(1 for (raw,) in rows if (json.loads(raw or "{}").get("task_output") or json.loads(raw or "{}")).get(f) not in (None, "", []))
    print(f"task_output.{f} present: {n}/{len(rows)}")
PY
