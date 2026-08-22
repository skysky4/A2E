#!/usr/bin/env bash
# Run the eight OpenAI-compatible A2E agents on Terminal-Bench 2.1 with GLM-5.3.
#
# Each agent runs the selected TB2.1 split with --concurrency 25. Agents are
# intentionally run one after another, so total benchmark concurrency stays at
# 25 instead of multiplying across agents. Each agent gets an independent A2E server
# lifecycle and SQLite database. The GLM compatibility proxy is shared.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
PROXY_SCRIPT="$ROOT/scripts/glm_openai_compat_proxy.py"
WATCHDOG_LIB="$ROOT/scripts/tb21_runner_watchdog.sh"

MODEL="glm-5.3"
CONCURRENCY="${TB21_CONCURRENCY:-25}"
SAMPLE_SEED="${TB21_SAMPLE_SEED:-20260817}"
INCLUDE_SECURITY=0
REQUESTED_N=""
DRY_RUN=0
PREWARM=0
RUNNER_EXIT_GRACE="${TB21_RUNNER_EXIT_GRACE:-60}"
RUNNER_TERM_GRACE="${TB21_RUNNER_TERM_GRACE:-15}"

usage() {
  cat <<'EOF'
Usage: scripts/run_tb21_glm53_8_agents.sh [options]

Runs these agents sequentially with GLM-5.3; each agent processes TB2.1 tasks
with concurrency 25:
  agno, openai-agents, langgraph, smolagents, google-adk,
  llama-index, crewai, autogen-agentchat

Options:
  --include-security   Run all 89 TB2.1 tasks. Default: 81 non-security tasks.
  --n N                Override the number of selected tasks.
  --concurrency N      Per-agent task concurrency. Default: 25.
  --prewarm            Prewarm trusted TB2.1 verifier dependencies first.
  --dry-run            Validate configuration and print the planned commands.
  -h, --help           Show this help.

Environment overrides:
  GLM_COMPAT_UPSTREAM_BASE_URL   Real upstream API base; defaults to the
                                 OPENAI_API_BASE loaded from .env.
  TB21_RUN_ROOT                 Result directory. Reusing it skips agents with
                                 an existing <agent>/DONE marker.
  TB21_PROXY_PORT               Local compatibility proxy port (default 18111).
  TB21_A2E_PORT                 Local A2E HTTP port (default 18112).
  TB21_A2E_GRPC_PORT            Local A2E gRPC port (default 18113).
  TB21_SAMPLE_SEED              Sampling seed (default 20260817).
  TB21_RUNNER_EXIT_GRACE        Seconds to wait after the completion marker
                                before reclaiming a stuck runner (default 60).
  TB21_RUNNER_TERM_GRACE        Seconds between SIGTERM and SIGKILL (default 15).

Examples:
  bash scripts/run_tb21_glm53_8_agents.sh
  bash scripts/run_tb21_glm53_8_agents.sh --include-security
  TB21_RUN_ROOT=/path/to/existing-run \
    bash scripts/run_tb21_glm53_8_agents.sh   # resume completed-agent markers
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
if [[ -n "$REQUESTED_N" ]]; then
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

[[ -x "$TASK_PY" ]] || die "missing task environment: run 'cd task && uv sync --frozen --all-packages'"
[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
[[ -f "$PROXY_SCRIPT" ]] || die "missing compatibility proxy: $PROXY_SCRIPT"
[[ -f "$WATCHDOG_LIB" ]] || die "missing runner watchdog: $WATCHDOG_LIB"
[[ -f "$ROOT/.env" ]] || die "missing $ROOT/.env"

# shellcheck disable=SC1090
source "$WATCHDOG_LIB"

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

UPSTREAM_BASE_URL="${GLM_COMPAT_UPSTREAM_BASE_URL:-${OPENAI_API_BASE:-}}"
[[ -n "$UPSTREAM_BASE_URL" ]] || die "set GLM_COMPAT_UPSTREAM_BASE_URL or OPENAI_API_BASE"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is not set"

PROXY_PORT="${TB21_PROXY_PORT:-18111}"
A2E_PORT="${TB21_A2E_PORT:-18112}"
A2E_GRPC_PORT="${TB21_A2E_GRPC_PORT:-18113}"
for port in "$PROXY_PORT" "$A2E_PORT" "$A2E_GRPC_PORT"; do
  is_positive_integer "$port" || die "ports must be positive integers"
  ((port <= 65535)) || die "invalid port: $port"
done
[[ "$PROXY_PORT" != "$A2E_PORT" && "$PROXY_PORT" != "$A2E_GRPC_PORT" ]] \
  || die "proxy and A2E ports must differ"
[[ "$A2E_PORT" != "$A2E_GRPC_PORT" ]] || die "A2E HTTP and gRPC ports must differ"

if [[ "$UPSTREAM_BASE_URL" == "http://127.0.0.1:$PROXY_PORT"* \
   || "$UPSTREAM_BASE_URL" == "http://localhost:$PROXY_PORT"* ]]; then
  die "upstream points back to this proxy; set GLM_COMPAT_UPSTREAM_BASE_URL to the real API"
fi

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
DEFAULT_RUN_ROOT="$ROOT/.a2e-tb2.1-glm-5.3-result/full-8-agents-c${CONCURRENCY}-${RUN_STAMP}"
RUN_ROOT="${TB21_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
if ((DRY_RUN)); then
  if [[ "$RUN_ROOT" != /* ]]; then
    RUN_ROOT="$(pwd -P)/$RUN_ROOT"
  fi
else
  mkdir -p "$RUN_ROOT"
  RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
fi

AUTOGEN_ROOT="$ROOT/task/agents/autogen_agentchat"
AUTOGEN_SRC="$AUTOGEN_ROOT/src"
autogen_sites=("$AUTOGEN_ROOT"/.venv/lib/python*/site-packages)
if ((${#autogen_sites[@]} != 1)) || [[ ! -d "${autogen_sites[0]}" ]]; then
  die "AutoGen isolated environment missing; run 'cd task/agents/autogen_agentchat && uv sync --index-strategy unsafe-best-match'"
fi
AUTOGEN_SITE="${autogen_sites[0]}"

agents=(
  agno
  openai-agents
  langgraph
  smolagents
  google-adk
  llama-index
  crewai
  autogen-agentchat
)

exclude_args=()
if ((!INCLUDE_SECURITY)); then
  exclude_args=(--exclude-category security)
fi

runner_base=(
  "$TASK_PY" "$ROOT/task/examples/run_experiment.py"
  --dataset terminal-bench-2.1
  --model "$MODEL"
  --evaluators tb_resolved
  --n "$TASK_COUNT"
  --sample-seed "$SAMPLE_SEED"
  --concurrency "$CONCURRENCY"
  --api-base "http://127.0.0.1:$PROXY_PORT/v1"
  --endpoint "http://127.0.0.1:$A2E_PORT"
  "${exclude_args[@]}"
)

echo "TB2.1 × GLM-5.3 eight-agent run"
echo "  tasks per agent: $TASK_COUNT"
echo "  security tasks: $([[ $INCLUDE_SECURITY == 1 ]] && echo included || echo excluded)"
echo "  concurrency: $CONCURRENCY per agent (agents run sequentially)"
echo "  results: $RUN_ROOT"

if ((DRY_RUN)); then
  echo
  echo "Dry run; planned invocations:"
  for agent in "${agents[@]}"; do
    echo "  database: $RUN_ROOT/$agent/a2e.db"
    printf '  '
    printf '%q ' "${runner_base[@]}" --agent "$agent" --run-id "tb21-glm53-c${CONCURRENCY}-${agent}-${RUN_STAMP}"
    printf '\n'
  done
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
assert_port_available "$PROXY_PORT"
assert_port_available "$A2E_PORT"
assert_port_available "$A2E_GRPC_PORT"

export GLM_COMPAT_UPSTREAM_BASE_URL="$UPSTREAM_BASE_URL"
export OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1"
export A2E_MODEL="$MODEL"
export A2E_PORT
export A2E_GRPC_PORT
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$A2E_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

PROXY_PID=""
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
    kill "$A2E_PID" 2>/dev/null || true
    wait "$A2E_PID" 2>/dev/null || true
  fi
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_runner
  stop_a2e
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_service() {
  local url=$1
  local name=$2
  local pid=$3
  local log_file=$4
  local attempt
  for attempt in {1..90}; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming ready" >&2
      tail -80 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$name did not become ready: $url" >&2
  tail -80 "$log_file" >&2 || true
  return 1
}

"$TASK_PY" "$PROXY_SCRIPT" --port "$PROXY_PORT" \
  >"$RUN_ROOT/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_service "http://127.0.0.1:$PROXY_PORT/healthz" \
  "GLM compatibility proxy" "$PROXY_PID" "$RUN_ROOT/proxy.log"

if [[ ! -f "$RUN_ROOT/process-summary.tsv" ]]; then
  printf 'agent\texit_status\tduration_seconds\tlog\n' >"$RUN_ROOT/process-summary.tsv"
fi
overall_status=0

for agent in "${agents[@]}"; do
  agent_dir="$RUN_ROOT/$agent"
  mkdir -p "$agent_dir"
  if [[ -f "$agent_dir/DONE" ]]; then
    echo "SKIP $agent (DONE marker exists)"
    continue
  fi
  if tb21_database_complete "$TASK_PY" "$agent_dir/a2e.db" "$TASK_COUNT"; then
    touch "$agent_dir/DONE"
    echo "SKIP $agent (database already contains a complete evaluated experiment)"
    continue
  fi

  export A2E_SQL_DATABASE_URL="sqlite:///$agent_dir/a2e.db"
  "$A2E_BIN" serve >"$agent_dir/server.log" 2>&1 &
  A2E_PID=$!
  if ! wait_for_service "http://127.0.0.1:$A2E_PORT/healthz" \
    "A2E server for $agent" "$A2E_PID" "$agent_dir/server.log"; then
    stop_a2e
    overall_status=1
    printf '%s\t%s\t%s\t%s\n' \
      "$agent" "server_start_failed" "0" "$agent_dir/server.log" \
      >>"$RUN_ROOT/process-summary.tsv"
    continue
  fi

  run_id="tb21-glm53-c${CONCURRENCY}-${agent}-${RUN_STAMP}"
  command=("${runner_base[@]}" --agent "$agent" --run-id "$run_id")
  echo "START $agent ($(date --iso-8601=seconds))"
  started=$SECONDS
  if [[ "$agent" == "autogen-agentchat" ]]; then
    PYTHONPATH="$AUTOGEN_SRC:$AUTOGEN_SITE:${PYTHONPATH:-}" \
      "${command[@]}" >"$agent_dir/runner.log" 2>&1 &
  else
    "${command[@]}" >"$agent_dir/runner.log" 2>&1 &
  fi
  RUNNER_PID=$!
  set +e
  wait_for_tb21_runner \
    "$RUNNER_PID" "$agent_dir/runner.log" "$agent_dir/a2e.db" "$TASK_COUNT" \
    "$TASK_PY" "$RUNNER_EXIT_GRACE" "$RUNNER_TERM_GRACE"
  status=$?
  set -e
  RUNNER_PID=""
  if ((TB21_RUNNER_FORCED_SHUTDOWN)); then
    printf 'completion was persisted; runner required forced shutdown\n' \
      >"$agent_dir/FORCED_SHUTDOWN"
    echo "RECOVERED $agent: completed database verified; stuck runner reclaimed" >&2
  elif ((TB21_RUNNER_INVALID_COMPLETION)); then
    echo "INVALID $agent: completion marker found but database is incomplete" >&2
  fi
  duration=$((SECONDS - started))
  printf '%s\t%s\t%s\t%s\n' \
    "$agent" "$status" "$duration" "$agent_dir/runner.log" \
    >>"$RUN_ROOT/process-summary.tsv"
  curl -fsS "http://127.0.0.1:$PROXY_PORT/metrics" \
    >"$agent_dir/proxy-metrics-after.json" || true
  stop_a2e

  if ((status == 0)); then
    touch "$agent_dir/DONE"
    echo "DONE $agent status=0 duration=${duration}s"
  else
    overall_status=1
    echo "FAILED $agent status=$status duration=${duration}s; see $agent_dir/runner.log" >&2
  fi
done

curl -fsS "http://127.0.0.1:$PROXY_PORT/metrics" \
  >"$RUN_ROOT/proxy-metrics-final.json" || true

"$TASK_PY" - "$RUN_ROOT" >"$RUN_ROOT/experiment-summary.tsv" <<'PY' || true
import json
import pathlib
import sqlite3
import sys

root = pathlib.Path(sys.argv[1])
print("agent\texperiment\tstatus\tresolved\treward\ttests_passed\ttests_total\tturns\ttool_calls")
query = """
    SELECT experiments.name, experiment_runs.output
    FROM experiments
    JOIN experiment_runs ON experiment_runs.experiment_id = experiments.id
    ORDER BY experiments.id
"""
for database in sorted(root.glob("*/a2e.db")):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(query)
        for row in rows:
            output = json.loads(row["output"])["task_output"]
            fields = (
                database.parent.name,
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

echo
echo "Finished. Results: $RUN_ROOT"
echo "Process summary: $RUN_ROOT/process-summary.tsv"
echo "Experiment summary: $RUN_ROOT/experiment-summary.tsv"
echo "Agent databases: $RUN_ROOT/<agent>/a2e.db"
exit "$overall_status"
