#!/usr/bin/env bash
# Start an isolated A2E Server and run, resume, or regrade one Campaign.

set -Eeuo pipefail

CALLER_DIR="$(pwd -P)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PY="$ROOT/task/.venv/bin/python"
A2E_BIN="$ROOT/server/.venv/bin/a2e"
CAMPAIGN_RUNNER="$ROOT/task/examples/run_campaign.py"
ENV_FILE="$ROOT/.env"

HTTP_PORT="${A2E_CAMPAIGN_HTTP_PORT:-6006}"
GRPC_PORT="${A2E_CAMPAIGN_GRPC_PORT:-4317}"
START_TIMEOUT="${A2E_CAMPAIGN_START_TIMEOUT:-90}"
TERM_GRACE="${A2E_CAMPAIGN_TERM_GRACE:-15}"
DATABASE="${A2E_CAMPAIGN_DATABASE:-}"
RUNS_DIR="$ROOT/task/runs"
PREWARM=0
DRY_RUN=0
MODE=""
TARGET=""
FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_campaign.sh --config CAMPAIGN.yaml [options]
  scripts/run_campaign.sh --resume RUN_DIR [--rerun-failed] [options]
  scripts/run_campaign.sh --regrade RUN_DIR [--grader ID] [options]

The script loads the repository .env, starts an isolated A2E Server backed by
a persistent SQLite database, waits for it to become healthy, invokes the
Campaign runner, and stops the Server when the Campaign exits.

Wrapper options:
  --database PATH   SQLite database path. The default is
                    .a2e-campaigns/<campaign-id>/a2e.db.
  --http-port PORT  A2E HTTP/OTLP port (default: 6006).
  --grpc-port PORT  A2E OTLP gRPC port (default: 4317).
  --prewarm         Prewarm the trusted TB2.1 verifier cache before running.
  -h, --help        Show this help.

All run_campaign.py options, including --dry-run, --rerun-failed, --grader,
--models-dir, --runs-dir, and --verbose, are forwarded unchanged. A dry run
does not start the A2E Server. Reusing a config whose Campaign directory was
already initialized automatically resumes that immutable Campaign.

Examples:
  scripts/run_campaign.sh \
    --config task/campaigns/tb21-crewai-smolagents-gpt56-glm53.yaml
  scripts/run_campaign.sh --resume task/runs/campaign-c19e7e74248264ab
  scripts/run_campaign.sh --resume task/runs/campaign-c19e7e74248264ab \
    --rerun-failed
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  (($# >= 2)) || die "$1 requires a value"
}

resolve_existing_path() {
  local value=$1
  if [[ "$value" == /* ]]; then
    [[ -e "$value" ]] || die "path does not exist: $value"
    realpath "$value"
  elif [[ -e "$CALLER_DIR/$value" ]]; then
    realpath "$CALLER_DIR/$value"
  elif [[ -e "$ROOT/$value" ]]; then
    realpath "$ROOT/$value"
  else
    die "path does not exist: $value"
  fi
}

resolve_output_path() {
  local value=$1
  if [[ "$value" == /* ]]; then
    realpath -m "$value"
  else
    realpath -m "$CALLER_DIR/$value"
  fi
}

is_port() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] && ((10#$1 <= 65535))
}

while (($#)); do
  case "$1" in
    --config|--resume|--regrade)
      require_value "$@"
      [[ -z "$MODE" ]] || die "choose exactly one of --config, --resume, or --regrade"
      MODE="${1#--}"
      TARGET=$2
      shift 2
      ;;
    --database)
      require_value "$@"
      DATABASE=$2
      shift 2
      ;;
    --http-port)
      require_value "$@"
      HTTP_PORT=$2
      shift 2
      ;;
    --grpc-port)
      require_value "$@"
      GRPC_PORT=$2
      shift 2
      ;;
    --prewarm)
      PREWARM=1
      shift
      ;;
    --runs-dir)
      require_value "$@"
      RUNS_DIR="$(resolve_output_path "$2")"
      FORWARD_ARGS+=("$1" "$RUNS_DIR")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$MODE" ]] || die "one of --config, --resume, or --regrade is required"
[[ -x "$TASK_PY" ]] || die "missing task environment; run: cd task && uv sync --frozen --all-packages"
[[ -x "$A2E_BIN" ]] || die "missing server environment; run: cd server && uv sync"
[[ -f "$CAMPAIGN_RUNNER" ]] || die "missing Campaign runner: $CAMPAIGN_RUNNER"
[[ -f "$ENV_FILE" ]] || die "missing environment file: $ENV_FILE"
is_port "$HTTP_PORT" || die "invalid HTTP port: $HTTP_PORT"
is_port "$GRPC_PORT" || die "invalid gRPC port: $GRPC_PORT"
[[ "$HTTP_PORT" != "$GRPC_PORT" ]] || die "HTTP and gRPC ports must differ"
[[ "$START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "A2E_CAMPAIGN_START_TIMEOUT must be positive"
[[ "$TERM_GRACE" =~ ^[0-9]+$ ]] || die "A2E_CAMPAIGN_TERM_GRACE must be non-negative"

TARGET="$(resolve_existing_path "$TARGET")"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

if [[ "$MODE" == "config" ]]; then
  RUN_KEY="$($TASK_PY -c '
import sys
from ageneval.task.orchestrator.controller import expand_campaign_id
from ageneval.task.orchestrator.schema import load_campaign
print(expand_campaign_id(load_campaign(sys.argv[1])))
' "$TARGET")"
  EXISTING_RUN="$RUNS_DIR/$RUN_KEY"
  if [[ -d "$EXISTING_RUN" ]]; then
    if [[ -f "$EXISTING_RUN/config.json" && -f "$EXISTING_RUN/lock.json" ]]; then
      echo "Campaign already initialized; resuming: $EXISTING_RUN" >&2
      MODE="resume"
      TARGET="$EXISTING_RUN"
    else
      die "Campaign directory exists but is not initialized: $EXISTING_RUN"
    fi
  fi
else
  RUN_KEY="$(basename "$TARGET")"
fi

CAMPAIGN_ARGS=("--$MODE" "$TARGET" "${FORWARD_ARGS[@]}")

if ((DRY_RUN)); then
  exec "$TASK_PY" "$CAMPAIGN_RUNNER" "${CAMPAIGN_ARGS[@]}"
fi

if ((PREWARM)); then
  "$TASK_PY" "$ROOT/scripts/prewarm_tb21_verifier_cache.py"
fi

STATE_DIR="$ROOT/.a2e-campaigns/$RUN_KEY"
mkdir -p "$STATE_DIR"

if [[ -z "$DATABASE" ]]; then
  DATABASE="$STATE_DIR/a2e.db"
else
  DATABASE="$(resolve_output_path "$DATABASE")"
fi
mkdir -p "$(dirname "$DATABASE")"

SERVER_LOG="$STATE_DIR/server.log"
export A2E_PORT="$HTTP_PORT"
export A2E_GRPC_PORT="$GRPC_PORT"
export A2E_SQL_DATABASE_URL="sqlite:///$DATABASE"
export A2E_COLLECTOR_ENDPOINT="http://127.0.0.1:$HTTP_PORT"
export OTEL_EXPORTER_OTLP_ENDPOINT="$A2E_COLLECTOR_ENDPOINT"
export A2E_SANDBOX_CLEANUP="${A2E_SANDBOX_CLEANUP:-1}"

"$TASK_PY" -c '
import socket
import sys

for raw_port in sys.argv[1:]:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", int(raw_port)))
    finally:
        sock.close()
' "$HTTP_PORT" "$GRPC_PORT" || die "A2E port is already in use"

A2E_PID=""

stop_a2e() {
  local deadline
  [[ -n "$A2E_PID" ]] || return 0
  kill -0 "$A2E_PID" 2>/dev/null || return 0
  kill -TERM "$A2E_PID" 2>/dev/null || true
  deadline=$((SECONDS + TERM_GRACE))
  while kill -0 "$A2E_PID" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "$A2E_PID" 2>/dev/null; then
    kill -KILL "$A2E_PID" 2>/dev/null || true
  fi
  wait "$A2E_PID" 2>/dev/null || true
  A2E_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_a2e
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

"$A2E_BIN" serve >"$SERVER_LOG" 2>&1 &
A2E_PID=$!

READY=0
for ((attempt = 1; attempt <= START_TIMEOUT; attempt++)); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$HTTP_PORT/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$A2E_PID" 2>/dev/null; then
    echo "A2E Server exited before becoming ready; log: $SERVER_LOG" >&2
    tail -100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

if ((READY == 0)); then
  echo "A2E Server did not become ready; log: $SERVER_LOG" >&2
  tail -100 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "A2E Server: http://127.0.0.1:$HTTP_PORT"
echo "Database: $DATABASE"
echo "Server log: $SERVER_LOG"
echo "Campaign mode: $MODE $TARGET"

"$TASK_PY" "$CAMPAIGN_RUNNER" "${CAMPAIGN_ARGS[@]}"
