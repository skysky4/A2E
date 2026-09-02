#!/usr/bin/env bash
# Offline SQLite V2 eval for gpt-5.6-sol official export DBs.
#
# Direct path (default): copy zhangmingxuan export read-only -> local SQLite ->
# run_gpt56sol_official_sqlite_eval.py (read/eval/writeback in SQLite, NO a2e serve).
# Checkpoint rsync evaluated DB to yuchenyue GPFS; never modify zhangmingxuan originals.
#
# Parallelism:
#   - DB level: full only by default (trajok duplicates trajectories, no spans)
#     Set AEP_EVAL_DB_PARALLEL=1 to also eval trajok.db (full + trajok concurrently).
#   - Experiment level: N experiments per DB at once (AEP_EVAL_EXPERIMENT_PARALLEL)
#   - Task level: --concurrency inside each experiment (AEP_EVAL_CONCURRENCY)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${AEP_GPFS_ENV:-$ROOT/config/gpfs_yuchenyue.env}"
DATA_ROOT="${AEP_GPT56SOL_DATA_ROOT:-/mnt/shared-storage-gpfs2/agenteval/zhangmingxuan/gpt-5.6-sol}"
GPFS_DIR="${AEP_GPT56SOL_GPFS_DIR:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/trajectories}"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
LOG_DIR="${AEP_GPT56SOL_EVAL_LOG_DIR:-${AEP_GPT56SOL_GPFS_LOG:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/logs}/official_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY="${AEP_EVAL_CONCURRENCY:-8}"
EXP_PARALLEL="${AEP_EVAL_EXPERIMENT_PARALLEL:-3}"
EVAL_FORCE="${AEP_EVAL_FORCE:-0}"
DB_PARALLEL="${AEP_EVAL_DB_PARALLEL:-0}"
EVAL_TRAJOK="${AEP_EVAL_TRAJOK:-0}"
SYNC_GPFS="${AEP_GPT56SOL_SYNC_GPFS:-1}"

LOCAL_DIR="${AEP_GPT56SOL_LOCAL_DIR:-${AEP_GPT56SOL_GPFS_WORK:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/work}}"
DB_FULL="${AEP_GPT56SOL_DB_FULL:-$LOCAL_DIR/gpt56sol-full-writable.db}"
DB_TRAJOK="${AEP_GPT56SOL_DB_TRAJOK:-$LOCAL_DIR/gpt56sol-trajok-writable.db}"
GPFS_FULL="${AEP_GPT56SOL_GPFS_FULL:-$GPFS_DIR/a2e-gpt-5.6-sol-official-full-evaluated.db}"
GPFS_TRAJOK="${AEP_GPT56SOL_GPFS_TRAJOK:-$GPFS_DIR/a2e-gpt-5.6-sol-official-trajok-evaluated.db}"
SRC_DIR="${AEP_GPT56SOL_SRC_DIR:-$DATA_ROOT/trajectories}"
EVAL_PY="$ROOT/eval/scripts/run_gpt56sol_official_sqlite_eval.py"

mkdir -p "$LOG_DIR" "$LOCAL_DIR" "$GPFS_DIR"

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
  echo "error: OPENAI_API_KEY not set" >&2
  exit 1
fi

ensure_local_db() {
  local src="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    # Never clobber an in-progress evaluated copy with the read-only export.
    local dest_ann
    dest_ann="$(sqlite3 "$dest" "SELECT COUNT(*) FROM experiment_run_annotations;" 2>/dev/null || echo 0)"
    if [[ "${dest_ann:-0}" -gt 0 ]]; then
      echo "keep existing evaluated copy $(basename "$dest") annotations=$dest_ann" | tee -a "$LOG_DIR/batch.log"
      return 0
    fi
  fi
  if [[ ! -f "$dest" || "$src" -nt "$dest" ]]; then
    echo "copy read-only $(basename "$src") -> $dest" | tee -a "$LOG_DIR/batch.log"
    cp -f "$src" "$dest"
  fi
}

sync_eval_db_to_gpfs() {
  local db="$1"
  local gpfs_dest="$2"
  [[ "$SYNC_GPFS" == "1" ]] || return 0
  mkdir -p "$GPFS_DIR"
  sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
  rsync -a --inplace "$db" "$gpfs_dest"
  echo "SYNC $db -> $gpfs_dest" | tee -a "$LOG_DIR/batch.log"
}

run_sqlite_eval() {
  cd "$ROOT/server"
  uv run python "$EVAL_PY" "$@"
}

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "error: DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi

ensure_local_db "$SRC_DIR/a2e-gpt-5.6-sol-official-full.db" "$DB_FULL"
if [[ "$DB_PARALLEL" == "1" || "$EVAL_TRAJOK" == "1" ]]; then
  ensure_local_db "$SRC_DIR/a2e-gpt-5.6-sol-official-trajok.db" "$DB_TRAJOK"
fi

list_experiment_ids() {
  local db="$1"
  python3 - "$db" <<'PY'
import sqlite3, sys
db = sys.argv[1]
conn = sqlite3.connect(db)
for (eid,) in conn.execute(
    "SELECT id FROM experiments ORDER BY id"
).fetchall():
    print(eid)
PY
}

run_db_parallel_experiments() {
  local db="$1"
  local log_prefix="$2"
  local force_flag=()
  [[ "$EVAL_FORCE" == "1" ]] && force_flag=(--force)

  mapfile -t EXP_IDS < <(list_experiment_ids "$db")
  echo "START $(basename "$db") experiments=${#EXP_IDS[@]} exp_parallel=$EXP_PARALLEL concurrency=$CONCURRENCY" \
    | tee -a "$LOG_DIR/batch.log"

  local fail=0
  local running=0
  for eid in "${EXP_IDS[@]}"; do
    while (( running >= EXP_PARALLEL )); do
      if wait -n; then
        :
      else
        fail=1
      fi
      running=$((running - 1))
    done
    local safe="exp${eid}"
    echo "LAUNCH $(basename "$db") experiment_id=$eid" | tee -a "$LOG_DIR/batch.log"
    (
      run_sqlite_eval \
        --db "$db" \
        --experiment-id "$eid" \
        --part all \
        "${force_flag[@]}" \
        --concurrency "$CONCURRENCY" \
        --log-level INFO \
        >>"${log_prefix}.${safe}.log" 2>&1
      if [[ "$(basename "$db")" == *trajok* ]]; then
        sync_eval_db_to_gpfs "$db" "$GPFS_TRAJOK"
      else
        sync_eval_db_to_gpfs "$db" "$GPFS_FULL"
      fi
    ) &
    running=$((running + 1))
  done
  while (( running > 0 )); do
    if wait -n; then
      :
    else
      fail=1
    fi
    running=$((running - 1))
  done
  if [[ "$fail" -eq 0 ]]; then
    echo "OK $(basename "$db")" | tee -a "$LOG_DIR/batch.log"
  else
    echo "FAIL $(basename "$db") (see ${log_prefix}.exp*.log)" | tee -a "$LOG_DIR/batch.log"
    return 1
  fi
}

echo "gpt-5.6-sol offline sqlite eval (direct SQLite, no server)" | tee "$LOG_DIR/batch.log"
echo "AEP_GPFS_ROOT=$AEP_GPFS_ROOT" | tee -a "$LOG_DIR/batch.log"
echo "DATA_ROOT=$DATA_ROOT (read-only source)" | tee -a "$LOG_DIR/batch.log"
echo "DB_FULL=$DB_FULL" | tee -a "$LOG_DIR/batch.log"
echo "GPFS_FULL=$GPFS_FULL" | tee -a "$LOG_DIR/batch.log"
echo "DB_TRAJOK=$DB_TRAJOK" | tee -a "$LOG_DIR/batch.log"
echo "GPFS_TRAJOK=$GPFS_TRAJOK" | tee -a "$LOG_DIR/batch.log"
echo "LOG_DIR=$LOG_DIR exp_parallel=$EXP_PARALLEL concurrency=$CONCURRENCY db_parallel=$DB_PARALLEL eval_trajok=$EVAL_TRAJOK sync_gpfs=$SYNC_GPFS" | tee -a "$LOG_DIR/batch.log"

fail=0
if [[ "$DB_PARALLEL" == "1" ]]; then
  run_db_parallel_experiments "$DB_FULL" "$LOG_DIR/official-full" &
  p1=$!
  run_db_parallel_experiments "$DB_TRAJOK" "$LOG_DIR/official-trajok" &
  p2=$!
  wait "$p1" || fail=1
  wait "$p2" || fail=1
else
  run_db_parallel_experiments "$DB_FULL" "$LOG_DIR/official-full" || fail=1
  if [[ "$EVAL_TRAJOK" == "1" ]]; then
    run_db_parallel_experiments "$DB_TRAJOK" "$LOG_DIR/official-trajok" || fail=1
  else
    echo "SKIP trajok (full.db only; set AEP_EVAL_TRAJOK=1 or AEP_EVAL_DB_PARALLEL=1 to include)" | tee -a "$LOG_DIR/batch.log"
  fi
fi

sync_eval_db_to_gpfs "$DB_FULL" "$GPFS_FULL"
if [[ "$EVAL_TRAJOK" == "1" && -f "$DB_TRAJOK" ]]; then
  sync_eval_db_to_gpfs "$DB_TRAJOK" "$GPFS_TRAJOK"
fi

if [[ "$fail" -eq 0 ]]; then
  echo "done OK: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
else
  echo "done WITH FAILURES: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
  exit 1
fi
