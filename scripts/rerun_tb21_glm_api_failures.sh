#!/usr/bin/env bash
# Rerun only failed GLM Trials caused by upstream 503 account-pool errors.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
CAMPAIGN_RUNNER="$ROOT/task/examples/run_campaign.py"
HELPER="$ROOT/scripts/tb21_missing_ctrf_rerun.py"
PREWARM_SCRIPT="$ROOT/scripts/prewarm_tb21_verifier_cache.py"
VERIFIER_CACHE="$ROOT/.a2e-cache/tb21-verifier/uv-0.9.5"

RUN_ROOT="${TB21_GLM_API_RERUN_ROOT:-}"
A2E_PORT="${TB21_GLM_API_RERUN_PORT:-18512}"
A2E_GRPC_PORT="${TB21_GLM_API_RERUN_GRPC_PORT:-18513}"
TERM_GRACE="${TB21_GLM_API_RERUN_TERM_GRACE:-45}"
DRY_RUN=0
PREWARM=0

usage() {
  cat <<'EOF'
Usage: scripts/rerun_tb21_glm_api_failures.sh [options]

Resumes only the GLM Campaign in an existing missing-CTRF rerun. It refuses
to start unless every currently failed GLM Trial is a retryable upstream
"503 No available accounts" failure. GPT timeout Trials are never selected.

Options:
  --root DIR    Existing missing-CTRF rerun root. Defaults to the newest one.
  --prewarm     Prewarm TB2.1 verifier dependencies before running.
  --dry-run     Print the exact selected Trials without starting anything.
  -h, --help    Show this message.

The existing SQLite database is reused. Successful attempts replace the
corresponding ExperimentRun rows that currently contain an error.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_port() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] && ((1 <= 10#$1 && 10#$1 <= 65535))
}

while (($#)); do
  case "$1" in
    --root)
      (($# >= 2)) || die "--root requires a directory"
      RUN_ROOT=$2
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

[[ -x "$TASK_PY" ]] || die "missing task environment"
[[ -f "$CAMPAIGN_RUNNER" && -f "$HELPER" ]] || die "missing Campaign scripts"
command -v jq >/dev/null || die "jq is required"
is_port "$A2E_PORT" || die "invalid HTTP port: $A2E_PORT"
is_port "$A2E_GRPC_PORT" || die "invalid gRPC port: $A2E_GRPC_PORT"
[[ "$A2E_PORT" != "$A2E_GRPC_PORT" ]] || die "HTTP and gRPC ports must differ"

if [[ -z "$RUN_ROOT" ]]; then
  mapfile -t candidates < <(
    find "$ROOT/.a2e-tb21-claude-sdk" -mindepth 1 -maxdepth 1 \
      -type d -name 'missing-ctrf-rerun-*' -printf '%T@\t%p\n' 2>/dev/null | sort -nr
  )
  ((${#candidates[@]} > 0)) || die "no missing-CTRF rerun found; pass --root"
  RUN_ROOT="${candidates[0]#*$'\t'}"
fi
[[ -d "$RUN_ROOT" ]] || die "run root does not exist: $RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
[[ -f "$RUN_ROOT/rerun-plan.json" ]] || die "missing rerun plan in $RUN_ROOT"
[[ -f "$RUN_ROOT/a2e.db" ]] || die "missing database in $RUN_ROOT"

INVOCATION="$RUN_ROOT/glm-api-reruns/$(date +%Y%m%d-%H%M%S)-$$"
TARGETS="$INVOCATION/targets.json"
mkdir -p "$INVOCATION"
"$TASK_PY" "$HELPER" select-glm-api-failures \
  --output "$RUN_ROOT" \
  --targets "$TARGETS"

TARGET_COUNT="$(jq -er '.targets | length' "$TARGETS")"
echo "GLM upstream API recovery"
echo "  run root: $RUN_ROOT"
echo "  selected Trials: $TARGET_COUNT"
jq -r '.targets[] | "  \(.task_id) [\(.trial_id)] attempt \(.attempt_before) -> next"' "$TARGETS"

if ((TARGET_COUNT == 0)); then
  echo "SKIP: there are no failed GLM 503 account-pool Trials"
  exit 0
fi
if ((DRY_RUN)); then
  echo "DRY RUN complete; GPT Campaign and timeout Trials are excluded"
  echo "Targets: $TARGETS"
  exit 0
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
export GLM_API_BASE="${GLM_API_BASE:-${GLM_COMPAT_UPSTREAM_BASE_URL:-${OPENAI_API_BASE:-}}}"
export GLM_API_KEY="${GLM_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$GLM_API_BASE" ]] || die "set GLM_API_BASE (or OPENAI_API_BASE)"
[[ -n "$GLM_API_KEY" ]] || die "set GLM_API_KEY (or OPENAI_API_KEY)"
[[ -x "$A2E_BIN" ]] || die "missing server environment: run 'cd server && uv sync'"
command -v docker >/dev/null || die "docker CLI is not installed"
command -v curl >/dev/null || die "curl is required"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
if ((PREWARM)); then
  "$TASK_PY" "$PREWARM_SCRIPT"
fi
[[ -d "$VERIFIER_CACHE" ]] || die "TB2.1 verifier cache missing; use --prewarm"

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

MODEL_DIR="$(jq -er '.models[] | select(.model=="glm-5.3") | .models_dir' "$RUN_ROOT/rerun-plan.json")"
CAMPAIGN="$(jq -er '.campaign' "$TARGETS")"
SERVER_LOG="$INVOCATION/server.log"
CAMPAIGN_LOG="$INVOCATION/campaign.log"
export A2E_PORT
export A2E_GRPC_PORT
export A2E_SQL_DATABASE_URL="sqlite:///$RUN_ROOT/a2e.db"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$A2E_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

A2E_PID=""
CAMPAIGN_PID=""

stop_processes() {
  if [[ -n "$CAMPAIGN_PID" ]] && kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
    kill -INT "$CAMPAIGN_PID" 2>/dev/null || true
    local deadline=$((SECONDS + TERM_GRACE))
    while kill -0 "$CAMPAIGN_PID" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    kill -TERM "$CAMPAIGN_PID" 2>/dev/null || true
    wait "$CAMPAIGN_PID" 2>/dev/null || true
  fi
  CAMPAIGN_PID=""
  if [[ -n "$A2E_PID" ]] && kill -0 "$A2E_PID" 2>/dev/null; then
    kill -TERM "$A2E_PID" 2>/dev/null || true
    wait "$A2E_PID" 2>/dev/null || true
  fi
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_processes
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"$A2E_BIN" serve >"$SERVER_LOG" 2>&1 &
A2E_PID=$!
for _attempt in {1..90}; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$A2E_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  kill -0 "$A2E_PID" 2>/dev/null || die "A2E Server exited; see $SERVER_LOG"
  sleep 1
done
curl -fsS --max-time 2 "http://127.0.0.1:$A2E_PORT/healthz" >/dev/null \
  || die "A2E Server did not become ready; see $SERVER_LOG"

"$TASK_PY" "$CAMPAIGN_RUNNER" \
  --resume "$CAMPAIGN" \
  --models-dir "$MODEL_DIR" \
  --rerun-failed >"$CAMPAIGN_LOG" 2>&1 &
CAMPAIGN_PID=$!
set +e
wait "$CAMPAIGN_PID"
campaign_status=$?
set -e
CAMPAIGN_PID=""
stop_processes

if ((campaign_status != 0)); then
  tail -100 "$CAMPAIGN_LOG" >&2 || true
  die "GLM Campaign exited with status $campaign_status"
fi

"$TASK_PY" "$HELPER" validate-glm-api-rerun \
  --output "$RUN_ROOT" \
  --targets "$TARGETS"
echo "DONE: selected GLM API failures were replaced by successful attempts"
echo "Report: $INVOCATION/report.json"
echo "Database: $RUN_ROOT/a2e.db"
