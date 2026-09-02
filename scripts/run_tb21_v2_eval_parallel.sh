#!/usr/bin/env bash
# Parallel V2 full eval + writeback for multiple TB21 SQLite DBs.
#
# Usage:
#   bash scripts/run_tb21_v2_eval_parallel.sh
#   bash scripts/run_tb21_v2_eval_parallel.sh /path/to/a2e-tb21-*.db ...
#   AEP_EVAL_ONLY_MISSING=1 bash scripts/run_tb21_v2_eval_parallel.sh
#
# Env:
#   AEP_EVAL_CONCURRENCY=16    per-DB async concurrency
#   AEP_EVAL_BASE_PORT=6006      first HTTP port (one DB per port)
#   AEP_EVAL_BASE_GRPC=4317      first gRPC port
#   AEP_EVAL_ONLY_MISSING=1      skip DBs that already have >=20 metrics
#   AEP_EVAL_METRICS=m1,m2      optional subset passed to run_eval --metrics
#   AEP_EVAL_FORCE=1             pass --force to run_eval

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
LOG_DIR="${AEP_TB21_EVAL_LOG_DIR:-$ROOT/eval/outputs/tb21_v2_parallel_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY="${AEP_EVAL_CONCURRENCY:-16}"
BASE_PORT="${AEP_EVAL_BASE_PORT:-6006}"
BASE_GRPC="${AEP_EVAL_BASE_GRPC:-4317}"
ONLY_MISSING="${AEP_EVAL_ONLY_MISSING:-1}"
EVAL_METRICS="${AEP_EVAL_METRICS:-}"
EVAL_FORCE="${AEP_EVAL_FORCE:-0}"

mkdir -p "$LOG_DIR"

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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY / AE2_EVAL_LLM_API_KEY not set (source $ENV_FILE)" >&2
  exit 1
fi

DBS=("$@")
if [[ ${#DBS[@]} -eq 0 ]]; then
  mapfile -t DBS < <(ls "$ROOT"/a2e-tb21-gpt-5.6-sol-*.db 2>/dev/null | sort)
fi

if [[ ${#DBS[@]} -eq 0 ]]; then
  echo "error: no DB files found" >&2
  exit 1
fi

filter_missing() {
  local db_path="$1"
  if [[ "$ONLY_MISSING" != "1" ]]; then
    echo "$db_path"
    return 0
  fi
  python3 - "$db_path" <<'PY'
import sqlite3, sys
db = sys.argv[1]
conn = sqlite3.connect(db)
metrics = conn.execute(
    "SELECT COUNT(DISTINCT name) FROM experiment_run_annotations WHERE name != 'tb_resolved'"
).fetchone()[0]
rows = conn.execute(
    "SELECT COUNT(*) FROM experiment_run_annotations WHERE name != 'tb_resolved'"
).fetchone()[0]
# complete: 81 runs * 23 metrics = 1863
raise SystemExit(0 if rows < 1863 else 1)
PY
}

cat > "$LOG_DIR/run_one.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
db_path="\$1"
port="\$2"
grpc_port="\$3"
log="\$4"
concurrency="\$5"
ROOT="\$6"
EVAL_METRICS="${EVAL_METRICS}"
EVAL_FORCE="${EVAL_FORCE}"
SCRIPT
cat >> "$LOG_DIR/run_one.sh" <<'SCRIPT'


BASE="http://127.0.0.1:${port}"
name=$(basename "$db_path")
exp_id=$(python3 -c "import base64,sqlite3; r=sqlite3.connect('$db_path').execute('SELECT id FROM experiments LIMIT 1').fetchone(); print(base64.b64encode(f'Experiment:{r[0]}'.encode()).decode())")

export A2E_SQL_DATABASE_URL="sqlite:////${db_path#/}"
export UV_CACHE_DIR="/tmp/uv-cache-${port}-$$"
mkdir -p "$UV_CACHE_DIR"
cd "$ROOT/server"
uv run a2e serve --host 127.0.0.1 --port "$port" --grpc-port "$grpc_port" >>"${log}.server.log" 2>&1 &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 120); do
  curl -sf "$BASE/a2e_version" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "$BASE/a2e_version" >/dev/null 2>&1 || { echo "FAIL $name server not ready"; exit 1; }

force_flag=()
if [[ "${EVAL_FORCE:-0}" == "1" ]]; then
  force_flag=(--force)
else
  rows=$(python3 -c "import sqlite3; print(sqlite3.connect('$db_path').execute(\"SELECT COUNT(*) FROM experiment_run_annotations WHERE name!='tb_resolved'\").fetchone()[0])")
  if [[ "$rows" -eq 0 ]]; then
    force_flag=(--force)
  fi
fi

if [[ -n "${EVAL_METRICS:-}" ]]; then
  uv run python ../eval/scripts/run_eval.py \
    --base-url "$BASE" \
    --experiment-id "$exp_id" \
    --metrics "$EVAL_METRICS" \
    "${force_flag[@]}" \
    --concurrency "$concurrency" \
    --log-level INFO >>"$log" 2>&1
else
  uv run python ../eval/scripts/run_eval.py \
    --base-url "$BASE" \
    --experiment-id "$exp_id" \
    --part all \
    "${force_flag[@]}" \
    --concurrency "$concurrency" \
    --log-level INFO >>"$log" 2>&1
fi

echo "OK $name"
SCRIPT
chmod +x "$LOG_DIR/run_one.sh"

PENDING=()
for db in "${DBS[@]}"; do
  [[ "$db" != /* ]] && db="$ROOT/$db"
  [[ -f "$db" ]] || { echo "SKIP missing $db" | tee -a "$LOG_DIR/batch.log"; continue; }
  if filter_missing "$db"; then
    PENDING+=("$db")
  else
    echo "SKIP already eval'd $(basename "$db")" | tee -a "$LOG_DIR/batch.log"
  fi
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "nothing to eval" | tee "$LOG_DIR/batch.log"
  exit 0
fi

echo "TB21 parallel V2 eval: ${#PENDING[@]} db(s), concurrency=$CONCURRENCY" | tee "$LOG_DIR/batch.log"
echo "LOG_DIR=$LOG_DIR" | tee -a "$LOG_DIR/batch.log"

pids=()
idx=0
for db_path in "${PENDING[@]}"; do
  name=$(basename "$db_path")
  port=$((BASE_PORT + idx))
  grpc_port=$((BASE_GRPC + idx))
  log="$LOG_DIR/${name%.db}.log"
  echo "START $name port=$port grpc=$grpc_port" | tee -a "$LOG_DIR/batch.log"
  "$LOG_DIR/run_one.sh" "$db_path" "$port" "$grpc_port" "$log" "$CONCURRENCY" "$ROOT" \
    >>"$LOG_DIR/batch.log" 2>&1 &
  pids+=($!)
  idx=$((idx + 1))
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail=1
    echo "FAIL pid=${pids[$i]} db=$(basename "${PENDING[$i]}")" | tee -a "$LOG_DIR/batch.log"
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "done OK: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
else
  echo "done WITH FAILURES: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
  exit 1
fi
