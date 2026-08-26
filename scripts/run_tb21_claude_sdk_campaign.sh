#!/usr/bin/env bash
# Run Claude SDK on Terminal-Bench 2.1 with GLM-5.3 and GPT-5.6-sol.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
CAMPAIGN_RUNNER="$ROOT/task/examples/run_campaign.py"
VALIDATOR="$ROOT/scripts/validate_tb21_campaign.py"
CONCURRENCY_PROBE="$ROOT/scripts/probe_trial_process_concurrency.py"
PREWARM_SCRIPT="$ROOT/scripts/prewarm_tb21_verifier_cache.py"
VERIFIER_CACHE="$ROOT/.a2e-cache/tb21-verifier/uv-0.9.5"

TASK_COUNT=81
DRY_RUN=0
PREWARM=0
RERUN_FAILED=0
RUNNER_TERM_GRACE="${TB21_RUNNER_TERM_GRACE:-15}"
SAMPLE_SEED="${TB21_SAMPLE_SEED:-20260822}"

usage() {
  cat <<'EOF'
Usage: scripts/run_tb21_claude_sdk_campaign.sh [options]

Runs one Campaign containing:
  glm-5.3 × terminal-bench-2.1 × claude-sdk
  gpt-5.6-sol × terminal-bench-2.1 × claude-sdk

The default is the complete 81-task non-security split for each model
(162 Trials total). The script starts an isolated A2E Server and validates
Campaign completion, database persistence, and concurrency high-water marks.

Options:
  --n N                Run N shared non-security tasks per model (default: 81).
  --prewarm            Prewarm trusted TB2.1 verifier dependencies first.
  --rerun-failed       On resume, append attempts for failed Trials.
  --dry-run            Expand and print the Campaign without credentials,
                       Docker, verifier cache, or A2E Server.
  -h, --help           Show this help.

Model environment:
  GLM_API_BASE         GLM OpenAI-compatible endpoint. Falls back to
                       GLM_COMPAT_UPSTREAM_BASE_URL, then OPENAI_API_BASE.
  GLM_API_KEY          GLM credential. Falls back to OPENAI_API_KEY.
  GPT56_API_BASE       GPT-5.6-sol endpoint. Falls back to OPENAI_API_BASE.
  OPENAI_API_KEY       GPT-5.6-sol credential.

Run and concurrency overrides:
  TB21_CLAUDE_CAMPAIGN_ROOT       Reuse this directory to resume.
  TB21_CONCURRENT_TRIALS          Default 64 for the full run.
  TB21_ACTIVE_CELLS               Default 2.
  TB21_CONCURRENT_SANDBOXES       Default 32 for the full run.
  TB21_MODEL_CONCURRENCY          Default 32 per-model ceiling for the full run.
  TB21_TOTAL_MODEL_CONCURRENCY    Default 32 across both models.
  TB21_CONCURRENT_GRADERS         Default 32 for the full run.
  TB21_CONCURRENT_UPLOADS         Default 16 for the full run.
  TB21_QUEUE_CAPACITY             Default 64.
  TB21_CLAUDE_A2E_PORT            Default 18412.
  TB21_CLAUDE_A2E_GRPC_PORT       Default 18413.
  TB21_SAMPLE_SEED                Default 20260822.

Examples:
  bash scripts/run_tb21_claude_sdk_campaign.sh --dry-run
  bash scripts/run_tb21_claude_sdk_campaign.sh --n 4 --prewarm
  bash scripts/run_tb21_claude_sdk_campaign.sh --prewarm
  TB21_CLAUDE_CAMPAIGN_ROOT=/path/to/run \
    bash scripts/run_tb21_claude_sdk_campaign.sh
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

min_int() {
  if (($1 < $2)); then
    echo "$1"
  else
    echo "$2"
  fi
}

while (($#)); do
  case "$1" in
    --n)
      (($# >= 2)) || die "--n requires a value"
      TASK_COUNT="$2"
      shift 2
      ;;
    --prewarm)
      PREWARM=1
      shift
      ;;
    --rerun-failed)
      RERUN_FAILED=1
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

is_positive_integer "$TASK_COUNT" || die "--n must be a positive integer"
((TASK_COUNT <= 81)) || die "the non-security TB2.1 split contains 81 tasks"
is_positive_integer "$SAMPLE_SEED" || die "TB21_SAMPLE_SEED must be positive"
is_positive_integer "$RUNNER_TERM_GRACE" || die "TB21_RUNNER_TERM_GRACE must be positive"

TOTAL_TRIALS=$((TASK_COUNT * 2))
DEFAULT_GLOBAL="$(min_int 64 "$TOTAL_TRIALS")"
DEFAULT_SANDBOX="$(min_int 32 "$TOTAL_TRIALS")"
DEFAULT_MODEL="$(min_int 32 "$TASK_COUNT")"
DEFAULT_MODEL="$(min_int "$DEFAULT_MODEL" "$DEFAULT_SANDBOX")"
((DEFAULT_MODEL > 0)) || DEFAULT_MODEL=1
DEFAULT_TOTAL_MODEL="$(min_int 32 "$TOTAL_TRIALS")"
DEFAULT_GRADER="$DEFAULT_SANDBOX"
DEFAULT_UPLOAD="$(min_int 16 "$TOTAL_TRIALS")"

GLOBAL_CONCURRENCY="${TB21_CONCURRENT_TRIALS:-$DEFAULT_GLOBAL}"
ACTIVE_CELLS="${TB21_ACTIVE_CELLS:-2}"
SANDBOX_CONCURRENCY="${TB21_CONCURRENT_SANDBOXES:-$DEFAULT_SANDBOX}"
MODEL_CONCURRENCY="${TB21_MODEL_CONCURRENCY:-$DEFAULT_MODEL}"
TOTAL_MODEL_CONCURRENCY="${TB21_TOTAL_MODEL_CONCURRENCY:-$DEFAULT_TOTAL_MODEL}"
GRADER_CONCURRENCY="${TB21_CONCURRENT_GRADERS:-$DEFAULT_GRADER}"
UPLOAD_CONCURRENCY="${TB21_CONCURRENT_UPLOADS:-$DEFAULT_UPLOAD}"
QUEUE_CAPACITY="${TB21_QUEUE_CAPACITY:-64}"
A2E_PORT="${TB21_CLAUDE_A2E_PORT:-18412}"
A2E_GRPC_PORT="${TB21_CLAUDE_A2E_GRPC_PORT:-18413}"

for value in \
  "$GLOBAL_CONCURRENCY" "$ACTIVE_CELLS" "$SANDBOX_CONCURRENCY" \
  "$MODEL_CONCURRENCY" "$TOTAL_MODEL_CONCURRENCY" \
  "$GRADER_CONCURRENCY" "$UPLOAD_CONCURRENCY" \
  "$QUEUE_CAPACITY" "$A2E_PORT" "$A2E_GRPC_PORT"; do
  is_positive_integer "$value" || die "concurrency and port values must be positive integers"
done
((A2E_PORT <= 65535 && A2E_GRPC_PORT <= 65535)) || die "ports must be <= 65535"
[[ "$A2E_PORT" != "$A2E_GRPC_PORT" ]] || die "A2E HTTP and gRPC ports must differ"
((ACTIVE_CELLS >= 2)) || die "TB21_ACTIVE_CELLS must be at least 2 for the dual-model test"
((QUEUE_CAPACITY >= GLOBAL_CONCURRENCY)) \
  || die "TB21_QUEUE_CAPACITY must be >= TB21_CONCURRENT_TRIALS"
((GLOBAL_CONCURRENCY <= TOTAL_TRIALS)) \
  || die "global concurrency cannot saturate with only $TOTAL_TRIALS Trials"
((SANDBOX_CONCURRENCY <= GLOBAL_CONCURRENCY)) \
  || die "sandbox concurrency must be <= global concurrency"
((MODEL_CONCURRENCY <= TASK_COUNT)) \
  || die "model concurrency cannot saturate with only $TASK_COUNT Trials per model"
((TOTAL_MODEL_CONCURRENCY <= SANDBOX_CONCURRENCY)) \
  || die "total model concurrency must be <= sandbox concurrency for this test"
((TOTAL_MODEL_CONCURRENCY <= MODEL_CONCURRENCY * 2)) \
  || die "total model concurrency exceeds the sum of both model pool limits"

[[ -x "$TASK_PY" ]] \
  || die "missing task environment: run 'cd task && uv sync --frozen --all-packages'"
[[ -f "$CAMPAIGN_RUNNER" ]] || die "missing Campaign runner: $CAMPAIGN_RUNNER"
[[ -f "$VALIDATOR" ]] || die "missing Campaign validator: $VALIDATOR"
command -v jq >/dev/null || die "jq is required"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
DEFAULT_RUN_ROOT="$ROOT/.a2e-tb21-claude-sdk/full-c${GLOBAL_CONCURRENCY}-s${SANDBOX_CONCURRENCY}-m${MODEL_CONCURRENCY}-${RUN_STAMP}"
RUN_ROOT="${TB21_CLAUDE_CAMPAIGN_ROOT:-$DEFAULT_RUN_ROOT}"
if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$(pwd -P)/$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"

CONFIG_DIR="$RUN_ROOT/config"
MODELS_DIR="$CONFIG_DIR/models"
CAMPAIGN_CONFIG="$CONFIG_DIR/campaign.yaml"
CAMPAIGN_RUNS_DIR="$RUN_ROOT/campaigns"
DATABASE="$RUN_ROOT/a2e.db"
PLAN_JSON="$RUN_ROOT/campaign-plan.json"
SERVER_LOG="$RUN_ROOT/server.log"
RUNNER_LOG="$RUN_ROOT/campaign-runner.log"
REPORT_JSON="$RUN_ROOT/concurrency-report.json"
REPORT_TSV="$RUN_ROOT/concurrency-report.tsv"
MODEL_SUMMARY="$RUN_ROOT/model-summary.tsv"
PROBE_REPORT="$RUN_ROOT/concurrency-probe.json"

TEMP_CONFIG="$(mktemp -d "$RUN_ROOT/.generated-config.XXXXXX")"
A2E_PID=""

stop_a2e() {
  if [[ -n "$A2E_PID" ]] && kill -0 "$A2E_PID" 2>/dev/null; then
    kill -TERM "$A2E_PID" 2>/dev/null || true
    local deadline=$((SECONDS + RUNNER_TERM_GRACE))
    while kill -0 "$A2E_PID" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 "$A2E_PID" 2>/dev/null; then
      kill -KILL "$A2E_PID" 2>/dev/null || true
    fi
    wait "$A2E_PID" 2>/dev/null || true
  fi
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_a2e
  if [[ -d "$TEMP_CONFIG" ]]; then
    rm -r "$TEMP_CONFIG"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

mkdir -p "$TEMP_CONFIG/models"
cat >"$TEMP_CONFIG/campaign.yaml" <<EOF
schema_version: 1
name: tb21-claude-sdk-dual-model
models:
  - glm-5.3
  - gpt-5.6-sol
benchmarks:
  - id: terminal-bench-2.1
    sample:
      n: $TASK_COUNT
      seed: $SAMPLE_SEED
      exclude_categories:
        - security
    graders:
      - id: terminal-bench
        mode: inline
        required: true
harnesses:
  - claude-sdk
repetitions: 1
matrix:
  exclude: []
execution:
  n_concurrent_trials: $GLOBAL_CONCURRENCY
  n_active_cells: $ACTIVE_CELLS
  n_concurrent_sandboxes: $SANDBOX_CONCURRENCY
  n_concurrent_model_sessions: $TOTAL_MODEL_CONCURRENCY
  n_concurrent_graders: $GRADER_CONCURRENCY
  n_concurrent_uploads: $UPLOAD_CONCURRENCY
  queue_capacity: $QUEUE_CAPACITY
  cancellation_grace_seconds: 30
  timeout_seconds: null
  retry:
    max_retries: 0
artifacts:
  retain: failures
EOF

cat >"$TEMP_CONFIG/models/glm-5.3.yaml" <<EOF
schema_version: 1
id: glm-5.3
provider: zai
model: glm-5.3
upstream_protocol: openai_chat_completions
connection:
  base_url_env: GLM_API_BASE
  api_key_env: GLM_API_KEY
capabilities:
  tools: true
  streaming: true
  vision: false
  structured_output: false
concurrency:
  group: zai-glm
  max_sessions: $MODEL_CONCURRENCY
gateway:
  interfaces:
    - openai_chat_completions
    - anthropic_messages
middleware:
  - glm_tool_call_compat
EOF

cat >"$TEMP_CONFIG/models/gpt-5.6-sol.yaml" <<EOF
schema_version: 1
id: gpt-5.6-sol
provider: openai-compatible
model: gpt-5.6-sol
upstream_protocol: openai_chat_completions
connection:
  base_url_env: GPT56_API_BASE
  api_key_env: OPENAI_API_KEY
capabilities:
  tools: true
  streaming: true
  vision: false
  structured_output: true
concurrency:
  group: gpt-5.6-sol
  max_sessions: $MODEL_CONCURRENCY
gateway:
  interfaces:
    - openai_chat_completions
    - anthropic_messages
middleware: []
EOF

if [[ -d "$CONFIG_DIR" ]]; then
  diff -ru "$CONFIG_DIR" "$TEMP_CONFIG" >/dev/null \
    || die "existing run configuration differs; choose a new TB21_CLAUDE_CAMPAIGN_ROOT"
  rm -r "$TEMP_CONFIG"
  TEMP_CONFIG=""
else
  mv "$TEMP_CONFIG" "$CONFIG_DIR"
  TEMP_CONFIG=""
fi
mkdir -p "$CAMPAIGN_RUNS_DIR"

shopt -s nullglob
campaign_dirs=("$CAMPAIGN_RUNS_DIR"/campaign-*)
shopt -u nullglob
if ((${#campaign_dirs[@]} > 1)); then
  die "expected at most one Campaign directory under $CAMPAIGN_RUNS_DIR"
fi

if ((${#campaign_dirs[@]} == 0)); then
  "$TASK_PY" "$CAMPAIGN_RUNNER" \
    --config "$CAMPAIGN_CONFIG" \
    --models-dir "$MODELS_DIR" \
    --runs-dir "$CAMPAIGN_RUNS_DIR" \
    --dry-run | tee "$PLAN_JSON"
  CAMPAIGN_ID="$(jq -er '.campaign_id' "$PLAN_JSON")" \
    || die "Campaign dry-run did not return campaign_id"
  CAMPAIGN_DIR="$CAMPAIGN_RUNS_DIR/$CAMPAIGN_ID"
else
  CAMPAIGN_DIR="${campaign_dirs[0]}"
  "$TASK_PY" "$CAMPAIGN_RUNNER" \
    --resume "$CAMPAIGN_DIR" \
    --models-dir "$MODELS_DIR" \
    --dry-run | tee "$PLAN_JSON"
fi

if ((DRY_RUN)); then
  echo "DRY RUN complete"
  echo "Plan: $PLAN_JSON"
  echo "Run root: $RUN_ROOT"
  exit 0
fi

validator_command=(
  "$TASK_PY" "$VALIDATOR"
  --campaign-dir "$CAMPAIGN_DIR"
  --database "$DATABASE"
  --expected-tasks-per-model "$TASK_COUNT"
  --expected-model glm-5.3
  --expected-model gpt-5.6-sol
  --expect-limit "global=$GLOBAL_CONCURRENCY"
  --expect-limit "sandbox=$SANDBOX_CONCURRENCY"
  --expect-limit "model:total=$TOTAL_MODEL_CONCURRENCY"
  --expect-limit "model:zai-glm=$MODEL_CONCURRENCY"
  --expect-limit "model:gpt-5.6-sol=$MODEL_CONCURRENCY"
  --expect-limit "grader=$GRADER_CONCURRENCY"
  --expect-limit "upload=$UPLOAD_CONCURRENCY"
  --require-saturated global
  --require-saturated sandbox
  --require-saturated model:total
  --require-exercised model:zai-glm
  --require-exercised model:gpt-5.6-sol
  --require-exercised grader
  --require-exercised upload
  --json-output "$REPORT_JSON"
  --tsv-output "$REPORT_TSV"
  --model-summary-output "$MODEL_SUMMARY"
)

if [[ -f "$RUN_ROOT/DONE" ]]; then
  if "${validator_command[@]}"; then
    echo "SKIP: completed Campaign already passed validation"
    echo "Results: $RUN_ROOT"
    exit 0
  fi
  die "DONE exists but validation failed; inspect $REPORT_JSON"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export GLM_API_BASE="${GLM_API_BASE:-${GLM_COMPAT_UPSTREAM_BASE_URL:-${OPENAI_API_BASE:-}}}"
export GLM_API_KEY="${GLM_API_KEY:-${OPENAI_API_KEY:-}}"
export GPT56_API_BASE="${GPT56_API_BASE:-${OPENAI_API_BASE:-}}"
[[ -n "$GLM_API_BASE" ]] || die "set GLM_API_BASE (or its documented fallback)"
[[ -n "$GLM_API_KEY" ]] || die "set GLM_API_KEY (or OPENAI_API_KEY)"
[[ -n "$GPT56_API_BASE" ]] || die "set GPT56_API_BASE (or OPENAI_API_BASE)"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "set OPENAI_API_KEY"

[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
command -v docker >/dev/null || die "docker CLI is not installed"
command -v curl >/dev/null || die "curl is required"
docker info >/dev/null 2>&1 || die "Docker daemon is not available"
if ((PREWARM)); then
  "$TASK_PY" "$PREWARM_SCRIPT"
fi
[[ -d "$VERIFIER_CACHE" ]] \
  || die "TB2.1 verifier cache missing; rerun with --prewarm"

echo "Probing $SANDBOX_CONCURRENCY-way synchronous Trial process concurrency"
"$TASK_PY" "$CONCURRENCY_PROBE" \
  --concurrency "$SANDBOX_CONCURRENCY" \
  --blocking-seconds 0.5 \
  --output "$PROBE_REPORT"

assert_port_available() {
  "$TASK_PY" -c '
import socket
import sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
' "$1" || die "local port $1 is already in use"
}
assert_port_available "$A2E_PORT"
assert_port_available "$A2E_GRPC_PORT"

export A2E_PORT
export A2E_GRPC_PORT
export A2E_SQL_DATABASE_URL="sqlite:///$DATABASE"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$A2E_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

wait_for_service() {
  local url=$1
  local pid=$2
  local attempt
  for attempt in {1..90}; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A2E Server exited before becoming ready" >&2
      tail -80 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "A2E Server did not become ready: $url" >&2
  tail -80 "$SERVER_LOG" >&2 || true
  return 1
}

echo "Claude SDK × TB2.1 dual-model Campaign"
echo "  tasks per model: $TASK_COUNT"
echo "  total Trials: $TOTAL_TRIALS"
echo "  concurrency: global=$GLOBAL_CONCURRENCY sandbox=$SANDBOX_CONCURRENCY model-per-pool=$MODEL_CONCURRENCY model-total=$TOTAL_MODEL_CONCURRENCY grader=$GRADER_CONCURRENCY upload=$UPLOAD_CONCURRENCY"
echo "  run root: $RUN_ROOT"

"$A2E_BIN" serve >"$SERVER_LOG" 2>&1 &
A2E_PID=$!
wait_for_service "http://127.0.0.1:$A2E_PORT/healthz" "$A2E_PID"

campaign_command=(
  "$TASK_PY" "$CAMPAIGN_RUNNER"
  --resume "$CAMPAIGN_DIR"
  --models-dir "$MODELS_DIR"
)
if ((RERUN_FAILED)); then
  campaign_command+=(--rerun-failed)
fi

set +e
"${campaign_command[@]}" 2>&1 | tee "$RUNNER_LOG"
campaign_status=${PIPESTATUS[0]}
set -e
stop_a2e

set +e
"${validator_command[@]}"
validation_status=$?
set -e
if ((campaign_status != 0)); then
  echo "Campaign exited with status $campaign_status; see $RUNNER_LOG" >&2
fi
if ((validation_status != 0)); then
  echo "Concurrency/result validation failed; see $REPORT_JSON" >&2
fi
if ((campaign_status != 0 || validation_status != 0)); then
  exit 1
fi

touch "$RUN_ROOT/DONE"
echo "DONE: Campaign and concurrency validation passed"
echo "Results: $RUN_ROOT"
echo "Concurrency: $REPORT_TSV"
echo "Process probe: $PROBE_REPORT"
echo "Models: $MODEL_SUMMARY"
echo "Database: $DATABASE"
