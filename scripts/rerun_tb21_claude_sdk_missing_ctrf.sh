#!/usr/bin/env bash
# Rerun only the exact Claude SDK model/task pairs missing a valid TB2.1 CTRF.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
CAMPAIGN_RUNNER="$ROOT/task/examples/run_campaign.py"
HELPER="$ROOT/scripts/tb21_missing_ctrf_rerun.py"
PREWARM_SCRIPT="$ROOT/scripts/prewarm_tb21_verifier_cache.py"
VERIFIER_CACHE="$ROOT/.a2e-cache/tb21-verifier/uv-0.9.5"

MODEL_CONCURRENCY="${TB21_CTRF_RERUN_MODEL_CONCURRENCY:-16}"
A2E_PORT="${TB21_CTRF_RERUN_PORT:-18512}"
A2E_GRPC_PORT="${TB21_CTRF_RERUN_GRPC_PORT:-18513}"
TERM_GRACE="${TB21_CTRF_RERUN_TERM_GRACE:-45}"
SOURCE_ROOT="${TB21_CTRF_SOURCE_ROOT:-}"
RUN_ROOT="${TB21_CTRF_RERUN_ROOT:-}"
DRY_RUN=0
PREWARM=0
RERUN_FAILED=0

usage() {
  cat <<'EOF'
Usage: scripts/rerun_tb21_claude_sdk_missing_ctrf.sh [options]

Scans a previous dual-model Claude SDK × TB2.1 Campaign and reruns only exact
model/task pairs whose persisted result has no valid CTRF summary. The rerun
uses a fresh SQLite database and preserves raw ctrf.json files.

Options:
  --source DIR       Source full-run root containing campaigns/ and a2e.db.
  --root DIR         Rerun output root; set this again to resume.
  --concurrency N    Maximum concurrency per model Campaign (default: 16).
  --prewarm          Prewarm TB2.1 verifier dependencies before running.
  --rerun-failed     On resume, append attempts for failed rerun Trials.
  --dry-run          Generate manifests/configs and print exact Campaign plans.
  -h, --help         Show this message.

Environment aliases:
  TB21_CTRF_SOURCE_ROOT
  TB21_CTRF_RERUN_ROOT
  TB21_CTRF_RERUN_MODEL_CONCURRENCY
  TB21_CTRF_RERUN_PORT
  TB21_CTRF_RERUN_GRPC_PORT

The original database is never modified. A successful run writes:
  rerun-plan.json, source-manifest.tsv, a2e.db,
  ctrf-rerun-report.json, and CTRF_COMPLETE.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (($#)); do
  case "$1" in
    --source)
      (($# >= 2)) || die "--source requires a directory"
      SOURCE_ROOT=$2
      shift 2
      ;;
    --root)
      (($# >= 2)) || die "--root requires a directory"
      RUN_ROOT=$2
      shift 2
      ;;
    --concurrency)
      (($# >= 2)) || die "--concurrency requires an integer"
      MODEL_CONCURRENCY=$2
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

is_positive_integer "$MODEL_CONCURRENCY" || die "concurrency must be positive"
is_positive_integer "$A2E_PORT" || die "HTTP port must be positive"
is_positive_integer "$A2E_GRPC_PORT" || die "gRPC port must be positive"
((A2E_PORT <= 65535 && A2E_GRPC_PORT <= 65535)) || die "ports must be <= 65535"
[[ "$A2E_PORT" != "$A2E_GRPC_PORT" ]] || die "HTTP and gRPC ports must differ"
[[ -x "$TASK_PY" ]] || die "missing task environment: run 'cd task && uv sync --frozen --all-packages'"
[[ -f "$HELPER" ]] || die "missing helper: $HELPER"
[[ -f "$CAMPAIGN_RUNNER" ]] || die "missing Campaign runner: $CAMPAIGN_RUNNER"
command -v jq >/dev/null || die "jq is required"

if [[ -z "$SOURCE_ROOT" ]]; then
  mapfile -t source_candidates < <(
    find "$ROOT/.a2e-tb21-claude-sdk" -mindepth 1 -maxdepth 1 \
      -type d -name 'full-*' -printf '%T@\t%p\n' 2>/dev/null | sort -nr
  )
  ((${#source_candidates[@]} > 0)) \
    || die "no source full-* run found; pass --source"
  SOURCE_ROOT="${source_candidates[0]#*$'\t'}"
fi
[[ -d "$SOURCE_ROOT" ]] || die "source directory does not exist: $SOURCE_ROOT"
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$ROOT/.a2e-tb21-claude-sdk/missing-ctrf-rerun-c${MODEL_CONCURRENCY}-$(date +%Y%m%d-%H%M%S)"
fi
if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$(pwd -P)/$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"

"$TASK_PY" "$HELPER" prepare \
  --repo "$ROOT" \
  --source "$SOURCE_ROOT" \
  --output "$RUN_ROOT" \
  --model-concurrency "$MODEL_CONCURRENCY"

PLAN="$RUN_ROOT/rerun-plan.json"
TOTAL_TRIALS="$(jq -er '.total_trials' "$PLAN")"
mapfile -t MODEL_SLUGS < <(jq -r '.models[].slug' "$PLAN")
((${#MODEL_SLUGS[@]} > 0)) || die "rerun plan contains no models"

declare -A CAMPAIGN_DIRS
declare -A CAMPAIGN_LOGS
for model_slug in "${MODEL_SLUGS[@]}"; do
  config="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .config' "$PLAN")"
  models_dir="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .models_dir' "$PLAN")"
  runs_dir="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .runs_dir' "$PLAN")"
  log="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .log' "$PLAN")"
  mkdir -p "$runs_dir"
  shopt -s nullglob
  campaign_candidates=("$runs_dir"/campaign-*)
  shopt -u nullglob
  ((${#campaign_candidates[@]} <= 1)) \
    || die "$model_slug has multiple Campaign directories under $runs_dir"
  plan_output="$RUN_ROOT/campaign-plan-$model_slug.json"
  if ((${#campaign_candidates[@]} == 0)); then
    "$TASK_PY" "$CAMPAIGN_RUNNER" \
      --config "$config" \
      --models-dir "$models_dir" \
      --runs-dir "$runs_dir" \
      --dry-run >"$plan_output"
    campaign_id="$(jq -er '.campaign_id' "$plan_output")"
    CAMPAIGN_DIRS[$model_slug]="$runs_dir/$campaign_id"
  else
    CAMPAIGN_DIRS[$model_slug]="${campaign_candidates[0]}"
    "$TASK_PY" "$CAMPAIGN_RUNNER" \
      --resume "${CAMPAIGN_DIRS[$model_slug]}" \
      --models-dir "$models_dir" \
      --dry-run >"$plan_output"
  fi
  CAMPAIGN_LOGS[$model_slug]="$log"
done

echo "TB2.1 missing-CTRF rerun"
echo "  source: $SOURCE_ROOT"
echo "  output: $RUN_ROOT"
echo "  exact Trials: $TOTAL_TRIALS"
for model_slug in "${MODEL_SLUGS[@]}"; do
  model="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .model' "$PLAN")"
  count="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .task_count' "$PLAN")"
  concurrency="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .concurrency' "$PLAN")"
  echo "  $model: $count Trials, concurrency=$concurrency"
done

if ((DRY_RUN)); then
  echo "DRY RUN complete"
  echo "Manifest: $RUN_ROOT/source-manifest.tsv"
  echo "Plan: $PLAN"
  exit 0
fi

if [[ -f "$RUN_ROOT/CTRF_COMPLETE" ]]; then
  "$TASK_PY" "$HELPER" validate --output "$RUN_ROOT" \
    || die "CTRF_COMPLETE exists but validation failed"
  echo "SKIP: rerun already completed and validated"
  exit 0
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export GLM_API_BASE="${GLM_API_BASE:-${GLM_COMPAT_UPSTREAM_BASE_URL:-${OPENAI_API_BASE:-${GPT56_API_BASE:-}}}}"
export GLM_API_KEY="${GLM_API_KEY:-${OPENAI_API_KEY:-}}"
export GPT56_API_BASE="${GPT56_API_BASE:-${OPENAI_API_BASE:-${GLM_API_BASE:-}}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${GLM_API_KEY:-}}"
[[ -n "$GLM_API_BASE" ]] || die "set GLM_API_BASE (or OPENAI_API_BASE)"
[[ -n "$GLM_API_KEY" ]] || die "set GLM_API_KEY (or OPENAI_API_KEY)"
[[ -n "$GPT56_API_BASE" ]] || die "set GPT56_API_BASE (or OPENAI_API_BASE)"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "set OPENAI_API_KEY"
[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
command -v docker >/dev/null || die "docker CLI is not installed"
command -v curl >/dev/null || die "curl is required"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
if ((PREWARM)); then
  "$TASK_PY" "$PREWARM_SCRIPT"
fi
[[ -d "$VERIFIER_CACHE" ]] \
  || die "TB2.1 verifier cache missing; rerun with --prewarm"

assert_port_available() {
  "$TASK_PY" -c '
import socket, sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
' "$1" || die "local port $1 is already in use"
}
assert_port_available "$A2E_PORT"
assert_port_available "$A2E_GRPC_PORT"

DATABASE="$RUN_ROOT/a2e.db"
SERVER_LOG="$RUN_ROOT/server.log"
export A2E_PORT
export A2E_GRPC_PORT
export A2E_SQL_DATABASE_URL="sqlite:///$DATABASE"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$A2E_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

A2E_PID=""
CAMPAIGN_PIDS=()

stop_campaigns() {
  local pid
  for pid in "${CAMPAIGN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  local deadline=$((SECONDS + TERM_GRACE))
  for pid in "${CAMPAIGN_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
  CAMPAIGN_PIDS=()
}

stop_server() {
  if [[ -n "$A2E_PID" ]] && kill -0 "$A2E_PID" 2>/dev/null; then
    kill -TERM "$A2E_PID" 2>/dev/null || true
    wait "$A2E_PID" 2>/dev/null || true
  fi
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_campaigns
  stop_server
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_service() {
  local attempt
  for attempt in {1..90}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$A2E_PORT/healthz" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$A2E_PID" 2>/dev/null; then
      tail -80 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 1
  done
  tail -80 "$SERVER_LOG" >&2 || true
  return 1
}

"$A2E_BIN" serve >"$SERVER_LOG" 2>&1 &
A2E_PID=$!
wait_for_service || die "A2E Server did not become ready"

for model_slug in "${MODEL_SLUGS[@]}"; do
  models_dir="$(jq -er --arg slug "$model_slug" '.models[] | select(.slug==$slug) | .models_dir' "$PLAN")"
  command=(
    "$TASK_PY" "$CAMPAIGN_RUNNER"
    --resume "${CAMPAIGN_DIRS[$model_slug]}"
    --models-dir "$models_dir"
  )
  if ((RERUN_FAILED)); then
    command+=(--rerun-failed)
  fi
  "${command[@]}" >"${CAMPAIGN_LOGS[$model_slug]}" 2>&1 &
  CAMPAIGN_PIDS+=("$!")
done

campaign_status=0
for index in "${!CAMPAIGN_PIDS[@]}"; do
  pid="${CAMPAIGN_PIDS[$index]}"
  model_slug="${MODEL_SLUGS[$index]}"
  if ! wait "$pid"; then
    campaign_status=1
    echo "$model_slug Campaign failed; recent log:" >&2
    tail -80 "${CAMPAIGN_LOGS[$model_slug]}" >&2 || true
  fi
done
CAMPAIGN_PIDS=()
stop_server
((campaign_status == 0)) || exit "$campaign_status"

"$TASK_PY" "$HELPER" validate --output "$RUN_ROOT"
echo "DONE: every selected Trial has raw CTRF, verified digest, evaluation, and DB payload"
echo "Results: $RUN_ROOT"
echo "Database: $DATABASE"
echo "Manifest: $RUN_ROOT/source-manifest.tsv"
echo "Report: $RUN_ROOT/ctrf-rerun-report.json"
