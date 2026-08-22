#!/usr/bin/env bash
# Run Google ADK on Terminal-Bench 2.1 with GPT-5.6-sol.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
WATCHDOG_LIB="$ROOT/scripts/tb21_runner_watchdog.sh"

MODEL="gpt-5.6-sol"
AGENT="google-adk"
CONCURRENCY="${TB21_CONCURRENCY:-25}"
SAMPLE_SEED="${TB21_SAMPLE_SEED:-20260817}"
INCLUDE_SECURITY=0
REQUESTED_N=""
TASK_IDS=()
DRY_RUN=0
PREWARM=0
RUNNER_EXIT_GRACE="${TB21_RUNNER_EXIT_GRACE:-60}"
RUNNER_TERM_GRACE="${TB21_RUNNER_TERM_GRACE:-15}"

usage() {
  cat <<'EOF'
Usage: scripts/run_tb21_google_adk_gpt56sol.sh [options]

Runs Google ADK with GPT-5.6-sol on Terminal-Bench 2.1. By default it runs
the 81 non-security tasks with concurrency 25.

Options:
  --include-security   Run all 89 TB2.1 tasks instead of the default 81.
  --n N                Run only N selected tasks.
  --task-id ID          Run this exact task; repeat to select multiple tasks.
  --concurrency N      Concurrent tasks (default: 25).
  --prewarm            Prewarm trusted TB2.1 verifier dependencies first.
  --dry-run            Validate configuration and print the planned command.
  -h, --help           Show this help.

Environment overrides:
  GPT56_API_BASE                     GPT-5.6-sol OpenAI-compatible API base;
                                     defaults to OPENAI_API_BASE from .env.
  TB21_GOOGLE_ADK_GPT56_RUN_ROOT     Result directory. Reusing a complete
                                     directory safely skips the experiment.
  TB21_GPT56_A2E_PORT                A2E HTTP port (default: 18212).
  TB21_GPT56_A2E_GRPC_PORT           A2E gRPC port (default: 18213).
  TB21_SAMPLE_SEED                   Sampling seed (default: 20260817).
  TB21_RUNNER_EXIT_GRACE             Seconds to reclaim a runner after its
                                     completion marker (default: 60).
  TB21_RUNNER_TERM_GRACE             SIGTERM grace period (default: 15).

Examples:
  bash scripts/run_tb21_google_adk_gpt56sol.sh
  bash scripts/run_tb21_google_adk_gpt56sol.sh --n 2 --concurrency 2
  bash scripts/run_tb21_google_adk_gpt56sol.sh --include-security
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while (($#)); do
  case "$1" in
    --include-security)
      INCLUDE_SECURITY=1
      shift
      ;;
    --n)
      (($# >= 2)) || die "--n requires a value"
      REQUESTED_N="$2"
      shift 2
      ;;
    --task-id)
      (($# >= 2)) || die "--task-id requires a value"
      TASK_IDS+=("$2")
      shift 2
      ;;
    --concurrency)
      (($# >= 2)) || die "--concurrency requires a value"
      CONCURRENCY="$2"
      shift 2
      ;;
    --prewarm)
      PREWARM=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

is_positive_integer "$CONCURRENCY" || die "concurrency must be a positive integer"
is_nonnegative_integer "$RUNNER_EXIT_GRACE" \
  || die "TB21_RUNNER_EXIT_GRACE must be a non-negative integer"
is_nonnegative_integer "$RUNNER_TERM_GRACE" \
  || die "TB21_RUNNER_TERM_GRACE must be a non-negative integer"

if [[ -n "$REQUESTED_N" ]] && ((${#TASK_IDS[@]})); then
  die "--n and --task-id cannot be used together"
fi
if ((${#TASK_IDS[@]})); then
  TASK_COUNT="${#TASK_IDS[@]}"
elif [[ -n "$REQUESTED_N" ]]; then
  is_positive_integer "$REQUESTED_N" || die "--n must be a positive integer"
  TASK_COUNT="$REQUESTED_N"
elif ((INCLUDE_SECURITY)); then
  TASK_COUNT=89
else
  TASK_COUNT=81
fi
MAX_TASK_COUNT=$((INCLUDE_SECURITY ? 89 : 81))
((TASK_COUNT <= MAX_TASK_COUNT)) \
  || die "requested $TASK_COUNT tasks, but this split contains $MAX_TASK_COUNT"

[[ -x "$TASK_PY" ]] \
  || die "missing task environment: run 'cd task && uv sync --frozen --all-packages'"
[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
[[ -f "$WATCHDOG_LIB" ]] || die "missing runner watchdog: $WATCHDOG_LIB"
[[ -f "$ROOT/.env" ]] || die "missing $ROOT/.env"

# shellcheck disable=SC1090
source "$WATCHDOG_LIB"

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

API_BASE="${GPT56_API_BASE:-${OPENAI_API_BASE:-}}"
[[ -n "$API_BASE" ]] || die "set GPT56_API_BASE or OPENAI_API_BASE"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is not set"

A2E_PORT="${TB21_GPT56_A2E_PORT:-18212}"
A2E_GRPC_PORT="${TB21_GPT56_A2E_GRPC_PORT:-18213}"
for port in "$A2E_PORT" "$A2E_GRPC_PORT"; do
  is_positive_integer "$port" || die "ports must be positive integers"
  ((port <= 65535)) || die "invalid port: $port"
done
[[ "$A2E_PORT" != "$A2E_GRPC_PORT" ]] || die "A2E HTTP and gRPC ports must differ"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
DEFAULT_RUN_ROOT="$ROOT/.a2e-tb21-gpt-5.6-sol-results/google-adk-c${CONCURRENCY}-${RUN_STAMP}"
RUN_ROOT="${TB21_GOOGLE_ADK_GPT56_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$(pwd -P)/$RUN_ROOT"
fi

exclude_args=()
if ((!INCLUDE_SECURITY)); then
  exclude_args=(--exclude-category security)
fi

task_args=()
for task_id in "${TASK_IDS[@]}"; do
  task_dir="$ROOT/task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/vendor/tasks/$task_id"
  [[ -d "$task_dir" ]] || die "unknown Terminal-Bench 2.1 task: $task_id"
  task_args+=(--task-id "$task_id")
done

RUN_ID="tb21-gpt56sol-c${CONCURRENCY}-google-adk-${RUN_STAMP}"
command=(
  "$TASK_PY" "$ROOT/task/examples/run_experiment.py"
  --dataset terminal-bench-2.1
  --model "$MODEL"
  --evaluators tb_resolved
  --n "$TASK_COUNT"
  --sample-seed "$SAMPLE_SEED"
  --concurrency "$CONCURRENCY"
  --api-base "$API_BASE"
  --endpoint "http://127.0.0.1:$A2E_PORT"
  "${exclude_args[@]}"
  "${task_args[@]}"
  --agent "$AGENT"
  --run-id "$RUN_ID"
)

echo "TB2.1 × Google ADK × GPT-5.6-sol"
echo "  tasks: $TASK_COUNT"
if ((${#TASK_IDS[@]})); then
  echo "  task IDs: ${TASK_IDS[*]}"
fi
echo "  security tasks: $([[ $INCLUDE_SECURITY == 1 ]] && echo included || echo excluded)"
echo "  concurrency: $CONCURRENCY"
echo "  database: $RUN_ROOT/a2e.db"

if ((DRY_RUN)); then
  echo "Dry run; planned invocation:"
  printf '  '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
DATABASE="$RUN_ROOT/a2e.db"

if [[ -f "$RUN_ROOT/DONE" ]]; then
  echo "SKIP: DONE marker exists at $RUN_ROOT/DONE"
  exit 0
fi
if tb21_database_complete "$TASK_PY" "$DATABASE" "$TASK_COUNT"; then
  touch "$RUN_ROOT/DONE"
  echo "SKIP: database already contains $TASK_COUNT evaluated runs"
  exit 0
fi

if ((PREWARM)); then
  echo "Prewarming TB2.1 verifier cache..."
  "$TASK_PY" "$ROOT/scripts/prewarm_tb21_verifier_cache.py"
elif [[ ! -d "$ROOT/.a2e-cache/tb21-verifier/uv-0.9.5" ]]; then
  die "TB2.1 verifier cache missing; rerun with --prewarm"
fi

assert_port_available() {
  local port=$1
  "$TASK_PY" -c '
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
' "$port" || die "local port $port is already in use"
}
assert_port_available "$A2E_PORT"
assert_port_available "$A2E_GRPC_PORT"

export OPENAI_API_BASE="$API_BASE"
export A2E_MODEL="$MODEL"
export A2E_PORT
export A2E_GRPC_PORT
export A2E_SQL_DATABASE_URL="sqlite:///$DATABASE"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$A2E_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

A2E_PID=""
RUNNER_PID=""

stop_runner() {
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    tb21_terminate_pid "$RUNNER_PID" "$RUNNER_TERM_GRACE"
    wait "$RUNNER_PID" 2>/dev/null || true
  fi
  RUNNER_PID=""
}

stop_a2e() {
  if [[ -n "$A2E_PID" ]] && kill -0 "$A2E_PID" 2>/dev/null; then
    tb21_terminate_pid "$A2E_PID" "$RUNNER_TERM_GRACE"
    wait "$A2E_PID" 2>/dev/null || true
  fi
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_runner
  stop_a2e
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_service() {
  local url=$1
  local pid=$2
  local log_file=$3
  local attempt
  for attempt in {1..90}; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A2E server exited before becoming ready" >&2
      tail -80 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "A2E server did not become ready: $url" >&2
  tail -80 "$log_file" >&2 || true
  return 1
}

"$A2E_BIN" serve >"$RUN_ROOT/server.log" 2>&1 &
A2E_PID=$!
wait_for_service "http://127.0.0.1:$A2E_PORT/healthz" \
  "$A2E_PID" "$RUN_ROOT/server.log"

echo "START $AGENT ($(date --iso-8601=seconds))"
started=$SECONDS
"${command[@]}" >"$RUN_ROOT/runner.log" 2>&1 &
RUNNER_PID=$!
set +e
wait_for_tb21_runner \
  "$RUNNER_PID" "$RUN_ROOT/runner.log" "$DATABASE" "$TASK_COUNT" \
  "$TASK_PY" "$RUNNER_EXIT_GRACE" "$RUNNER_TERM_GRACE"
status=$?
set -e
RUNNER_PID=""
duration=$((SECONDS - started))

if ((TB21_RUNNER_FORCED_SHUTDOWN)); then
  printf 'completion was persisted; runner required forced shutdown\n' \
    >"$RUN_ROOT/FORCED_SHUTDOWN"
  echo "RECOVERED: completed database verified; stuck runner reclaimed" >&2
elif ((TB21_RUNNER_INVALID_COMPLETION)); then
  echo "INVALID: completion marker found but database is incomplete" >&2
fi

stop_a2e
printf 'agent\texit_status\tduration_seconds\tlog\n' >"$RUN_ROOT/process-summary.tsv"
printf '%s\t%s\t%s\t%s\n' \
  "$AGENT" "$status" "$duration" "$RUN_ROOT/runner.log" \
  >>"$RUN_ROOT/process-summary.tsv"

if ((status != 0)) || ! tb21_database_complete "$TASK_PY" "$DATABASE" "$TASK_COUNT"; then
  echo "FAILED status=$status duration=${duration}s; see $RUN_ROOT/runner.log" >&2
  exit 1
fi

"$TASK_PY" - "$DATABASE" >"$RUN_ROOT/experiment-summary.tsv" <<'PY'
import json
import sqlite3
import sys

database = sys.argv[1]
connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
print("experiment\tstatus\tresolved\treward\ttests_passed\ttests_total\tturns\ttool_calls")
query = """
    SELECT experiments.name, experiment_runs.output
    FROM experiments
    JOIN experiment_runs ON experiment_runs.experiment_id = experiments.id
    ORDER BY experiment_runs.id
"""
try:
    for row in connection.execute(query):
        output = json.loads(row["output"])["task_output"]
        fields = (
            row["name"],
            output.get("status"),
            output.get("resolved"),
            output.get("tb_reward"),
            output.get("tb_tests_passed"),
            output.get("tb_tests_total"),
            output.get("turns"),
            len(output.get("tool_calls") or []),
        )
        print("\t".join("" if value is None else str(value) for value in fields))
finally:
    connection.close()
PY

touch "$RUN_ROOT/DONE"
echo "DONE status=0 duration=${duration}s"
echo "Results: $RUN_ROOT"
echo "Database: $DATABASE"
echo "Summary: $RUN_ROOT/experiment-summary.tsv"
