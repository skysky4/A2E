#!/usr/bin/env bash
# Rerun every GLM-5.3 Terminal-Bench 2.1 agent/task pair whose original
# experiment did not produce a non-empty, parseable CTRF report.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

MODEL="glm-5.3"
CONCURRENCY="${TB21_CTRF_RERUN_CONCURRENCY:-4}"
SAMPLE_SEED="${TB21_CTRF_RERUN_SAMPLE_SEED:-20260817}"
PROXY_PORT="${TB21_CTRF_RERUN_PROXY_PORT:-18511}"
HTTP_PORT="${TB21_CTRF_RERUN_PORT:-18512}"
GRPC_PORT="${TB21_CTRF_RERUN_GRPC_PORT:-18513}"
SOURCE_ROOT="${TB21_CTRF_SOURCE_ROOT:-${REPO_ROOT}/.a2e-tb2.1-glm-5.3-result/full-8-agents-c32-20260820-002801}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${TB21_CTRF_RERUN_ROOT:-${REPO_ROOT}/.a2e-tb2.1-glm-5.3-result/missing-ctrf-rerun-c${CONCURRENCY}-${STAMP}}"
PREWARM="${TB21_CTRF_RERUN_PREWARM:-1}"
RUNNER_EXIT_GRACE="${TB21_RUNNER_EXIT_GRACE:-60}"
RUNNER_TERM_GRACE="${TB21_RUNNER_TERM_GRACE:-15}"
DRY_RUN=0

TASK_PY="${REPO_ROOT}/task/.venv/bin/python"
A2E_BIN="${REPO_ROOT}/server/.venv/bin/a2e"
PROXY_SCRIPT="${REPO_ROOT}/scripts/glm_openai_compat_proxy.py"
WATCHDOG_LIB="${REPO_ROOT}/scripts/tb21_runner_watchdog.sh"
PREWARM_SCRIPT="${REPO_ROOT}/scripts/prewarm_tb21_verifier_cache.py"
TASKS_ROOT="${REPO_ROOT}/task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/vendor/tasks"

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

usage() {
  cat <<'EOF'
Usage: bash scripts/rerun_tb21_glm53_missing_ctrf.sh [options]

Reads the completed GLM-5.3 eight-agent run, finds every agent/task result
whose CTRF file was missing, invalid, or reported zero tests, and reruns only
those pairs. Agents run sequentially; tasks within one agent run concurrently.
The default task concurrency is 4.

Options:
  --source-root DIR   Original eight-agent result directory.
  --output-root DIR   Fresh directory for rerun databases and logs.
  --concurrency N     Concurrent tasks within each agent (default: 4).
  --skip-prewarm      Do not prewarm the selected verifier dependencies.
  --dry-run           Discover missing pairs and print commands only.
  -h, --help          Show this help.

Environment overrides:
  TB21_CTRF_SOURCE_ROOT               original result directory
  TB21_CTRF_RERUN_ROOT                fresh rerun output directory
  TB21_CTRF_RERUN_CONCURRENCY         default: 4
  TB21_CTRF_RERUN_SAMPLE_SEED         default: 20260817
  TB21_CTRF_RERUN_PROXY_PORT          default: 18511
  TB21_CTRF_RERUN_PORT                A2E HTTP port; default: 18512
  TB21_CTRF_RERUN_GRPC_PORT           A2E gRPC port; default: 18513
  TB21_CTRF_RERUN_PREWARM             0 or 1; default: 1
  GLM_COMPAT_UPSTREAM_BASE_URL        real upstream API base; defaults to the
                                      OPENAI_API_BASE loaded from .env

The script succeeds only if every rerun produces a parseable CTRF report with
at least one test. It writes CTRF_COMPLETE under the output root on success.
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
    --source-root)
      (($# >= 2)) || die "--source-root requires a value"
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a value"
      RUN_ROOT="$2"
      shift 2
      ;;
    --concurrency)
      (($# >= 2)) || die "--concurrency requires a value"
      CONCURRENCY="$2"
      shift 2
      ;;
    --skip-prewarm)
      PREWARM=0
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
is_positive_integer "$PROXY_PORT" || die "proxy port must be a positive integer"
is_positive_integer "$HTTP_PORT" || die "A2E HTTP port must be a positive integer"
is_positive_integer "$GRPC_PORT" || die "A2E gRPC port must be a positive integer"
is_nonnegative_integer "$RUNNER_EXIT_GRACE" \
  || die "TB21_RUNNER_EXIT_GRACE must be a non-negative integer"
is_nonnegative_integer "$RUNNER_TERM_GRACE" \
  || die "TB21_RUNNER_TERM_GRACE must be a non-negative integer"
for port in "$PROXY_PORT" "$HTTP_PORT" "$GRPC_PORT"; do
  ((port <= 65535)) || die "invalid port: $port"
done
[[ "$PROXY_PORT" != "$HTTP_PORT" && "$PROXY_PORT" != "$GRPC_PORT" \
   && "$HTTP_PORT" != "$GRPC_PORT" ]] || die "all three ports must differ"
[[ "$PREWARM" == "0" || "$PREWARM" == "1" ]] \
  || die "TB21_CTRF_RERUN_PREWARM must be 0 or 1"

[[ -x "$TASK_PY" ]] \
  || die "missing task environment: run 'cd task && uv sync --frozen --all-packages'"
[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
[[ -f "$PROXY_SCRIPT" ]] || die "missing compatibility proxy: $PROXY_SCRIPT"
[[ -f "$WATCHDOG_LIB" ]] || die "missing runner watchdog: $WATCHDOG_LIB"
[[ -f "$PREWARM_SCRIPT" ]] || die "missing verifier prewarm script: $PREWARM_SCRIPT"
[[ -f "$REPO_ROOT/.env" ]] || die "missing $REPO_ROOT/.env"
[[ -d "$SOURCE_ROOT" ]] || die "source result directory does not exist: $SOURCE_ROOT"
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"

if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$(pwd -P)/$RUN_ROOT"
fi
if [[ -e "$RUN_ROOT" ]]; then
  die "refusing to reuse existing output directory: $RUN_ROOT"
fi

# shellcheck disable=SC1090
source "$WATCHDOG_LIB"

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/.env"
set +a

UPSTREAM_BASE_URL="${GLM_COMPAT_UPSTREAM_BASE_URL:-${OPENAI_API_BASE:-}}"
[[ -n "$UPSTREAM_BASE_URL" ]] \
  || die "set GLM_COMPAT_UPSTREAM_BASE_URL or OPENAI_API_BASE"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is not set"
if [[ "$UPSTREAM_BASE_URL" == "http://127.0.0.1:$PROXY_PORT"* \
   || "$UPSTREAM_BASE_URL" == "http://localhost:$PROXY_PORT"* ]]; then
  die "upstream points back to this proxy; set GLM_COMPAT_UPSTREAM_BASE_URL to the real API"
fi

MANIFEST="$(mktemp "${TMPDIR:-/tmp}/tb21-glm53-missing-ctrf.XXXXXX.tsv")"
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
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  rm -f -- "$MANIFEST"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"$TASK_PY" - "$SOURCE_ROOT" "${agents[@]}" >"$MANIFEST" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

source = Path(sys.argv[1])
agents = sys.argv[2:]

for agent in agents:
    database = source / agent / "a2e.db"
    if not database.is_file():
        raise SystemExit(f"missing source database for {agent}: {database}")

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        experiment = connection.execute(
            """
            SELECT experiment_id, COUNT(*) AS run_count
            FROM experiment_runs
            GROUP BY experiment_id
            ORDER BY run_count DESC, experiment_id DESC
            LIMIT 1
            """
        ).fetchone()
        if experiment is None or experiment["run_count"] != 81:
            found = None if experiment is None else experiment["run_count"]
            raise SystemExit(
                f"expected an 81-run source experiment for {agent}, found {found}"
            )

        rows = connection.execute(
            """
            SELECT revision.metadata, run.output
            FROM experiment_runs AS run
            JOIN experiments_dataset_examples AS selected
              ON selected.experiment_id = run.experiment_id
             AND selected.dataset_example_id = run.dataset_example_id
            JOIN dataset_example_revisions AS revision
              ON revision.id = selected.dataset_example_revision_id
            WHERE run.experiment_id = ?
            ORDER BY run.id
            """,
            (experiment["experiment_id"],),
        )
        for row in rows:
            metadata = json.loads(row["metadata"])
            output = json.loads(row["output"])["task_output"]
            total = output.get("tb_tests_total")
            passed = output.get("tb_tests_passed")
            failed = output.get("tb_tests_failed")
            valid = (
                output.get("tb_ctrf_error") is None
                and isinstance(total, int)
                and total > 0
                and isinstance(passed, int)
                and isinstance(failed, int)
            )
            if not valid:
                print(f"{agent}\t{metadata['task_id']}")
    finally:
        connection.close()
PY

mapfile -t unique_tasks < <(cut -f2 "$MANIFEST" | sort -u)
if ((${#unique_tasks[@]} == 0)); then
  echo "All source runs already contain valid, non-empty CTRF reports."
  exit 0
fi

pair_count="$(wc -l <"$MANIFEST")"
echo "GLM-5.3 missing-CTRF rerun"
echo "  source:      $SOURCE_ROOT"
echo "  output:      $RUN_ROOT"
echo "  concurrency: $CONCURRENCY (agents sequential; tasks concurrent)"
echo "  pairs:       $pair_count"
echo "  tasks:       ${unique_tasks[*]}"
echo "  prewarm:     $PREWARM"
echo
printf 'agent\ttask\n'
cat "$MANIFEST"

for task_id in "${unique_tasks[@]}"; do
  [[ -d "$TASKS_ROOT/$task_id" ]] || die "unknown local TB2.1 task: $task_id"
done

prewarm_command=("$TASK_PY" "$PREWARM_SCRIPT")
for task_id in "${unique_tasks[@]}"; do
  prewarm_command+=(--task-id "$task_id")
done

AUTOGEN_ROOT="$REPO_ROOT/task/agents/autogen_agentchat"
AUTOGEN_SRC="$AUTOGEN_ROOT/src"
autogen_sites=("$AUTOGEN_ROOT"/.venv/lib/python*/site-packages)
if ((${#autogen_sites[@]} != 1)) || [[ ! -d "${autogen_sites[0]}" ]]; then
  die "AutoGen isolated environment missing; run 'cd task/agents/autogen_agentchat && uv sync --index-strategy unsafe-best-match'"
fi
AUTOGEN_SITE="${autogen_sites[0]}"

if ((DRY_RUN)); then
  if ((PREWARM)); then
    printf '\nPlanned prewarm command:\n  '
    printf '%q ' "${prewarm_command[@]}"
    printf '\n'
  fi
  printf '\nPlanned agent reruns:\n'
  for agent in "${agents[@]}"; do
    mapfile -t task_ids < <(awk -F '\t' -v wanted="$agent" '$1 == wanted {print $2}' "$MANIFEST")
    ((${#task_ids[@]})) || continue
    task_args=()
    for task_id in "${task_ids[@]}"; do
      task_args+=(--task-id "$task_id")
    done
    command=(
      "$TASK_PY" "$REPO_ROOT/task/examples/run_experiment.py"
      --dataset terminal-bench-2.1
      --model "$MODEL"
      --evaluators tb_resolved
      --n "${#task_ids[@]}"
      --sample-seed "$SAMPLE_SEED"
      --concurrency "$CONCURRENCY"
      --api-base "http://127.0.0.1:$PROXY_PORT/v1"
      --endpoint "http://127.0.0.1:$HTTP_PORT"
      "${task_args[@]}"
      --agent "$agent"
      --run-id "tb21-glm53-missing-ctrf-$agent-$STAMP"
    )
    printf '  [%s] ' "$agent"
    printf '%q ' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
cp "$MANIFEST" "$RUN_ROOT/rerun-manifest.tsv"

if ((PREWARM)); then
  echo
  echo "Prewarming verifier dependencies for ${#unique_tasks[@]} selected tasks..."
  "${prewarm_command[@]}"
elif [[ ! -d "$REPO_ROOT/.a2e-cache/tb21-verifier/uv-0.9.5" ]]; then
  die "TB2.1 verifier cache missing; omit --skip-prewarm"
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
assert_port_available "$HTTP_PORT"
assert_port_available "$GRPC_PORT"

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

validate_rerun_database() {
  local database=$1
  local summary=$2
  local agent=$3
  shift 3
  "$TASK_PY" - "$database" "$summary" "$agent" "$@" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

database = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
agent = sys.argv[3]
expected = set(sys.argv[4:])

connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
try:
    experiment = connection.execute(
        "SELECT MAX(id) FROM experiments"
    ).fetchone()[0]
    rows = list(
        connection.execute(
            """
            SELECT revision.metadata, run.output
            FROM experiment_runs AS run
            JOIN experiments_dataset_examples AS selected
              ON selected.experiment_id = run.experiment_id
             AND selected.dataset_example_id = run.dataset_example_id
            JOIN dataset_example_revisions AS revision
              ON revision.id = selected.dataset_example_revision_id
            WHERE run.experiment_id = ?
            ORDER BY run.id
            """,
            (experiment,),
        )
    )
finally:
    connection.close()

parsed = []
for row in rows:
    task_id = json.loads(row["metadata"])["task_id"]
    output = json.loads(row["output"])["task_output"]
    parsed.append((task_id, output))

found = {task_id for task_id, _ in parsed}
if found != expected or len(parsed) != len(expected):
    raise SystemExit(
        f"unexpected selection for {agent}: expected={sorted(expected)}, "
        f"found={sorted(found)}, rows={len(parsed)}"
    )

lines = [
    "agent\ttask\tstatus\tresolved\treward\ttests_passed\t"
    "tests_failed\ttests_total\tctrf_error"
]
invalid = []
for task_id, output in sorted(parsed):
    total = output.get("tb_tests_total")
    passed = output.get("tb_tests_passed")
    failed = output.get("tb_tests_failed")
    ctrf_error = output.get("tb_ctrf_error")
    fields = (
        agent,
        task_id,
        output.get("tb_status"),
        output.get("resolved"),
        output.get("tb_reward"),
        passed,
        failed,
        total,
        ctrf_error,
    )
    lines.append("\t".join("" if value is None else str(value) for value in fields))
    valid = (
        ctrf_error is None
        and isinstance(total, int)
        and total > 0
        and isinstance(passed, int)
        and isinstance(failed, int)
    )
    if not valid:
        invalid.append(task_id)

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
if invalid:
    raise SystemExit(
        f"missing or invalid CTRF after rerun for {agent}: " + ", ".join(invalid)
    )
PY
}

export GLM_COMPAT_UPSTREAM_BASE_URL="$UPSTREAM_BASE_URL"
export OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1"
export A2E_MODEL="$MODEL"
export A2E_PORT="$HTTP_PORT"
export A2E_GRPC_PORT="$GRPC_PORT"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$HTTP_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

"$TASK_PY" "$PROXY_SCRIPT" --port "$PROXY_PORT" \
  >"$RUN_ROOT/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_service "http://127.0.0.1:$PROXY_PORT/healthz" \
  "GLM compatibility proxy" "$PROXY_PID" "$RUN_ROOT/proxy.log"

printf 'agent\texit_status\tduration_seconds\ttasks\tlog\n' \
  >"$RUN_ROOT/process-summary.tsv"
overall_status=0

for agent in "${agents[@]}"; do
  mapfile -t task_ids < <(awk -F '\t' -v wanted="$agent" '$1 == wanted {print $2}' "$MANIFEST")
  ((${#task_ids[@]})) || continue

  agent_dir="$RUN_ROOT/$agent"
  mkdir -p "$agent_dir"
  database="$agent_dir/a2e.db"
  export A2E_SQL_DATABASE_URL="sqlite:///$database"

  "$A2E_BIN" serve >"$agent_dir/server.log" 2>&1 &
  A2E_PID=$!
  if ! wait_for_service "http://127.0.0.1:$HTTP_PORT/healthz" \
    "A2E server for $agent" "$A2E_PID" "$agent_dir/server.log"; then
    stop_a2e
    overall_status=1
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$agent" "server_start_failed" "0" "${task_ids[*]}" "$agent_dir/server.log" \
      >>"$RUN_ROOT/process-summary.tsv"
    continue
  fi

  task_args=()
  for task_id in "${task_ids[@]}"; do
    task_args+=(--task-id "$task_id")
  done
  command=(
    "$TASK_PY" "$REPO_ROOT/task/examples/run_experiment.py"
    --dataset terminal-bench-2.1
    --model "$MODEL"
    --evaluators tb_resolved
    --n "${#task_ids[@]}"
    --sample-seed "$SAMPLE_SEED"
    --concurrency "$CONCURRENCY"
    --api-base "http://127.0.0.1:$PROXY_PORT/v1"
    --endpoint "http://127.0.0.1:$HTTP_PORT"
    "${task_args[@]}"
    --agent "$agent"
    --run-id "tb21-glm53-missing-ctrf-$agent-$STAMP"
  )

  echo
  echo "START $agent tasks=${task_ids[*]} ($(date --iso-8601=seconds))"
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
    "$RUNNER_PID" "$agent_dir/runner.log" "$database" "${#task_ids[@]}" \
    "$TASK_PY" "$RUNNER_EXIT_GRACE" "$RUNNER_TERM_GRACE"
  status=$?
  set -e
  RUNNER_PID=""
  duration=$((SECONDS - started))

  if ((TB21_RUNNER_FORCED_SHUTDOWN)); then
    printf 'completion was persisted; runner required forced shutdown\n' \
      >"$agent_dir/FORCED_SHUTDOWN"
    echo "RECOVERED $agent: completed database verified; stuck runner reclaimed" >&2
  elif ((TB21_RUNNER_INVALID_COMPLETION)); then
    echo "INVALID $agent: completion marker found but database is incomplete" >&2
  fi

  curl -fsS "http://127.0.0.1:$PROXY_PORT/metrics" \
    >"$agent_dir/proxy-metrics-after.json" || true
  stop_a2e

  if ((status == 0)) \
     && tb21_database_complete "$TASK_PY" "$database" "${#task_ids[@]}" \
     && validate_rerun_database \
          "$database" "$agent_dir/ctrf-summary.tsv" "$agent" "${task_ids[@]}"; then
    touch "$agent_dir/DONE"
    echo "DONE $agent status=0 duration=${duration}s"
  else
    status=1
    overall_status=1
    echo "FAILED $agent duration=${duration}s; see $agent_dir/runner.log" >&2
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$agent" "$status" "$duration" "${task_ids[*]}" "$agent_dir/runner.log" \
    >>"$RUN_ROOT/process-summary.tsv"
done

curl -fsS "http://127.0.0.1:$PROXY_PORT/metrics" \
  >"$RUN_ROOT/proxy-metrics-final.json" || true

{
  printf 'agent\ttask\tstatus\tresolved\treward\ttests_passed\ttests_failed\ttests_total\tctrf_error\n'
  for summary in "$RUN_ROOT"/*/ctrf-summary.tsv; do
    [[ -f "$summary" ]] || continue
    tail -n +2 "$summary"
  done
} >"$RUN_ROOT/ctrf-summary.tsv"

if ((overall_status == 0)); then
  touch "$RUN_ROOT/CTRF_COMPLETE"
  echo
  echo "All $pair_count missing/invalid CTRF pairs were rerun successfully."
  echo "Marker:   $RUN_ROOT/CTRF_COMPLETE"
else
  echo
  echo "One or more CTRF reruns failed validation." >&2
fi
echo "Results:  $RUN_ROOT"
echo "Manifest: $RUN_ROOT/rerun-manifest.tsv"
echo "Summary:  $RUN_ROOT/ctrf-summary.tsv"
exit "$overall_status"
