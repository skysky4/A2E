#!/usr/bin/env bash
# Rerun the two Google ADK TB2.1 tasks that did not produce ctrf.json.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

TASKS=(hf-model-inference overfull-hbox)
MODEL="gpt-5.6-sol"
CONCURRENCY="${TB21_CTRF_RERUN_CONCURRENCY:-1}"
SAMPLE_SEED="${TB21_CTRF_RERUN_SAMPLE_SEED:-20260817}"
HTTP_PORT="${TB21_CTRF_RERUN_PORT:-18412}"
GRPC_PORT="${TB21_CTRF_RERUN_GRPC_PORT:-18413}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${TB21_CTRF_RERUN_ROOT:-${REPO_ROOT}/.a2e-tb21-gpt-5.6-sol-results/google-adk-missing-ctrf-rerun-${STAMP}}"
TASK_PY="${REPO_ROOT}/task/.venv/bin/python"
PREWARM="${TB21_CTRF_RERUN_PREWARM:-1}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/rerun_tb21_google_adk_missing_ctrf.sh [options]

Reruns exactly these Terminal-Bench 2.1 tasks with Google ADK + GPT-5.6-sol:
  hf-model-inference
  overfull-hbox

The tasks run sequentially by default to reduce verifier contention. The script
prewarms only their verifier dependencies, writes to a fresh SQLite database,
and fails unless both tasks produce a parseable CTRF report.

Options:
  --dry-run       Validate and print the planned rerun without starting it.
  --skip-prewarm  Do not prewarm the two verifier dependency sets.
  -h, --help      Show this help.

Environment overrides:
  TB21_CTRF_RERUN_CONCURRENCY          default: 1
  TB21_CTRF_RERUN_SAMPLE_SEED          default: 20260817
  TB21_CTRF_RERUN_PORT                 default: 18412
  TB21_CTRF_RERUN_GRPC_PORT            default: 18413
  TB21_CTRF_RERUN_ROOT                 fresh output directory
  TB21_CTRF_RERUN_PREWARM              0 or 1; default: 1
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-prewarm) PREWARM=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$CONCURRENCY" "$HTTP_PORT" "$GRPC_PORT"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Concurrency and ports must be positive integers." >&2
    exit 2
  fi
done
if ((HTTP_PORT > 65535 || GRPC_PORT > 65535)); then
  echo "Ports must not exceed 65535." >&2
  exit 2
fi
if [[ "$HTTP_PORT" == "$GRPC_PORT" ]]; then
  echo "A2E HTTP and gRPC ports must differ." >&2
  exit 2
fi
if [[ "$PREWARM" != "0" && "$PREWARM" != "1" ]]; then
  echo "TB21_CTRF_RERUN_PREWARM must be 0 or 1." >&2
  exit 2
fi
if [[ ! -x "$TASK_PY" ]]; then
  echo "Task Python environment is missing: $TASK_PY" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing output directory: $RUN_ROOT" >&2
  exit 1
fi
for task_id in "${TASKS[@]}"; do
  task_dir="${REPO_ROOT}/task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/vendor/tasks/${task_id}"
  if [[ ! -d "$task_dir" ]]; then
    echo "Unknown local TB2.1 task: $task_id" >&2
    exit 1
  fi
done

echo "Google ADK missing-CTRF rerun"
echo "  tasks:       ${TASKS[*]}"
echo "  model:       $MODEL"
echo "  concurrency: $CONCURRENCY"
echo "  ports:       $HTTP_PORT/$GRPC_PORT"
echo "  output:      $RUN_ROOT"
echo "  prewarm:     $PREWARM"

prewarm_command=(
  "$TASK_PY" "${REPO_ROOT}/scripts/prewarm_tb21_verifier_cache.py"
  --task-id hf-model-inference
  --task-id overfull-hbox
)

if ((DRY_RUN)); then
  if ((PREWARM)); then
    "${prewarm_command[@]}" --dry-run
  fi
  printf 'Planned rerun command:\n  '
  printf '%q ' env \
    "TB21_GOOGLE_ADK_GPT56_RUN_ROOT=$RUN_ROOT" \
    "TB21_GPT56_A2E_PORT=$HTTP_PORT" \
    "TB21_GPT56_A2E_GRPC_PORT=$GRPC_PORT" \
    "TB21_SAMPLE_SEED=$SAMPLE_SEED" \
    bash "${REPO_ROOT}/scripts/run_tb21_google_adk_gpt56sol.sh" \
    --task-id hf-model-inference \
    --task-id overfull-hbox \
    --concurrency "$CONCURRENCY"
  printf '\n'
  exit 0
fi

if ((PREWARM)); then
  echo "Prewarming verifier dependencies for the two selected tasks..."
  "${prewarm_command[@]}"
fi

export TB21_GOOGLE_ADK_GPT56_RUN_ROOT="$RUN_ROOT"
export TB21_GPT56_A2E_PORT="$HTTP_PORT"
export TB21_GPT56_A2E_GRPC_PORT="$GRPC_PORT"
export TB21_SAMPLE_SEED="$SAMPLE_SEED"

bash "${REPO_ROOT}/scripts/run_tb21_google_adk_gpt56sol.sh" \
  --task-id hf-model-inference \
  --task-id overfull-hbox \
  --concurrency "$CONCURRENCY"

DATABASE="${RUN_ROOT}/a2e.db"
SUMMARY="${RUN_ROOT}/ctrf-summary.tsv"
"$TASK_PY" - "$DATABASE" "$SUMMARY" "${TASKS[@]}" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

database = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected = set(sys.argv[3:])
if not database.is_file():
    raise SystemExit(f"missing rerun database: {database}")

connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
query = """
    SELECT dataset_example_revisions.metadata, experiment_runs.output
    FROM experiment_runs
    JOIN dataset_example_revisions
      ON dataset_example_revisions.dataset_example_id = experiment_runs.dataset_example_id
"""
rows = []
try:
    for metadata_raw, output_raw in connection.execute(query):
        task_id = json.loads(metadata_raw)["task_id"]
        output = json.loads(output_raw)["task_output"]
        rows.append((task_id, output))
finally:
    connection.close()

found = {task_id for task_id, _ in rows}
if found != expected or len(rows) != len(expected):
    raise SystemExit(
        f"unexpected rerun selection: expected={sorted(expected)}, found={sorted(found)}"
    )

lines = [
    "task\tstatus\tresolved\treward\ttests_passed\ttests_total\tctrf_error"
]
invalid = []
for task_id, output in sorted(rows):
    ctrf_error = output.get("tb_ctrf_error")
    tests_total = output.get("tb_tests_total")
    fields = (
        task_id,
        output.get("tb_status"),
        output.get("resolved"),
        output.get("tb_reward"),
        output.get("tb_tests_passed"),
        tests_total,
        ctrf_error,
    )
    lines.append("\t".join("" if value is None else str(value) for value in fields))
    if ctrf_error is not None or not isinstance(tests_total, int) or tests_total <= 0:
        invalid.append(task_id)

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
if invalid:
    raise SystemExit("missing or invalid CTRF report for: " + ", ".join(invalid))
PY

touch "${RUN_ROOT}/CTRF_COMPLETE"
echo "Both CTRF reports are present and parseable."
echo "Database: $DATABASE"
echo "Summary:  $SUMMARY"
