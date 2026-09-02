#!/usr/bin/env bash
# Run V2 full eval on TB21 harness DBs and write back via local A2E server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${A2E_BASE_URL:-http://127.0.0.1:6006}"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
LOG_DIR="${AEP_TB21_EVAL_LOG_DIR:-$ROOT/eval/outputs/tb21_v2_writeback_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY="${AEP_EVAL_CONCURRENCY:-8}"
PORT="${A2E_SERVE_PORT:-6006}"

ONLY_MISSING="${AEP_EVAL_ONLY_MISSING:-1}"

mkdir -p "$LOG_DIR"

DBS=()
if [[ $# -gt 0 ]]; then
  DBS=("$@")
else
  mapfile -t DBS < <(ls "$ROOT"/a2e-tb21-gpt-5.6-sol-*.db 2>/dev/null | sort)
fi
if [[ "$ONLY_MISSING" == "1" ]]; then
  FILTERED=()
  for db_path in "${DBS[@]}"; do
    [[ "$db_path" != /* ]] && db_path="$ROOT/$db_path"
    n="$(python3 -c "import sqlite3; print(sqlite3.connect('$db_path').execute(\"SELECT COUNT(DISTINCT name) FROM experiment_run_annotations WHERE name != 'tb_resolved'\").fetchone()[0])")"
    if [[ "$n" -lt 20 ]]; then
      FILTERED+=("$db_path")
    else
      echo "SKIP already eval'd $(basename "$db_path") ($n metrics)" | tee -a "$LOG_DIR/batch.log"
    fi
  done
  DBS=("${FILTERED[@]}")
fi

if [[ ${#DBS[@]} -eq 0 ]]; then
  echo "nothing to eval" | tee "$LOG_DIR/batch.log"
  exit 0
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export A2E_EVAL_LLM_PROVIDER="${A2E_EVAL_LLM_PROVIDER:-${AE2_EVAL_LLM_PROVIDER:-openai}}"
export A2E_EVAL_LLM_MODEL="${A2E_EVAL_LLM_MODEL:-${AE2_EVAL_LLM_MODEL:-deepseek-v4-pro}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${AE2_EVAL_LLM_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${AE2_EVAL_LLM_API_KEY:-}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-${USER:-aep}}"
mkdir -p "$UV_CACHE_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY / AE2_EVAL_LLM_API_KEY not set (source $ENV_FILE)" >&2
  exit 1
fi

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_server() {
  local url="$1"
  for _ in $(seq 1 60); do
    if curl -sf "$url/a2e_version" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "error: server did not become ready at $url" >&2
  return 1
}

start_server() {
  local db_path="$1"
  stop_server
  export A2E_SQL_DATABASE_URL="sqlite:////${db_path#/}"
  cd "$ROOT/server"
  nohup uv run a2e serve --host 127.0.0.1 --port "$PORT" >>"$LOG_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  wait_server "$BASE_URL"
}

trap stop_server EXIT

echo "TB21 V2 eval writeback (${#DBS[@]} db(s), sequential)" | tee "$LOG_DIR/batch.log"
echo "concurrency=$CONCURRENCY logs=$LOG_DIR" | tee -a "$LOG_DIR/batch.log"

for db_path in "${DBS[@]}"; do
  db_name="$(basename "$db_path")"
  exp_id="$(python3 - <<PY
import base64, sqlite3
conn = sqlite3.connect("$db_path")
row = conn.execute("SELECT id FROM experiments ORDER BY id LIMIT 1").fetchone()
if not row:
    raise SystemExit("no experiment in $db_path")
print(base64.b64encode(f"Experiment:{row[0]}".encode()).decode())
PY
)"
  safe_name="${db_name%.db}"
  log_file="$LOG_DIR/${safe_name}.log"

  echo "=== $db_name experiment=$exp_id ===" | tee -a "$LOG_DIR/batch.log"
  start_server "$db_path"
  export A2E_SQL_DATABASE_URL="sqlite:////${db_path#/}"

  cd "$ROOT/server"
  if uv run python ../eval/scripts/run_eval.py \
    --base-url "$BASE_URL" \
    --experiment-id "$exp_id" \
    --part all \
    --force \
    --concurrency "$CONCURRENCY" \
    --log-level INFO \
    >>"$log_file" 2>&1; then
    echo "OK $db_name" | tee -a "$LOG_DIR/batch.log"
  else
    echo "FAIL $db_name (see $log_file)" | tee -a "$LOG_DIR/batch.log"
  fi
done

stop_server
echo "done: $LOG_DIR/batch.log"
