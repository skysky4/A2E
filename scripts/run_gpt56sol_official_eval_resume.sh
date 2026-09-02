#!/usr/bin/env bash
# Resume offline V2 eval for gpt-5.6-sol official DB (skip fully-scored experiments).
#
# Path: read zhangmingxuan export (read-only) -> eval in local SQLite copy -> write
# annotations directly to SQLite (NO a2e serve / HTTP writeback). Checkpoint rsync
# to yuchenyue GPFS when each experiment finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${AEP_GPFS_ENV:-$ROOT/config/gpfs_yuchenyue.env}"
SRC_ROOT="${AEP_GPT56SOL_DATA_ROOT:-/mnt/shared-storage-gpfs2/agenteval/zhangmingxuan/gpt-5.6-sol}"
GPFS_DIR="${AEP_GPT56SOL_GPFS_DIR:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/trajectories}"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
LOG_DIR="${AEP_GPT56SOL_EVAL_LOG_DIR:-${AEP_GPT56SOL_GPFS_LOG:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/logs}/official_resume_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY="${AEP_EVAL_CONCURRENCY:-6}"
EXP_PARALLEL="${AEP_EVAL_EXPERIMENT_PARALLEL:-2}"
EVAL_FORCE="${AEP_EVAL_FORCE:-0}"
FULL_METRICS="${AEP_EVAL_FULL_METRICS:-23}"
SYNC_GPFS="${AEP_GPT56SOL_SYNC_GPFS:-1}"

LOCAL_DIR="${AEP_GPT56SOL_LOCAL_DIR:-${AEP_GPT56SOL_STAGING_DIR:-$ROOT/gpt56sol-eval}}"
GPFS_WORK="${AEP_GPT56SOL_GPFS_WORK:-${AEP_GPFS_ROOT}/gpt-5.6-sol-eval/work}"
DB_FULL="${AEP_GPT56SOL_DB_FULL:-$LOCAL_DIR/gpt56sol-full-writable.db}"
GPFS_FULL="${AEP_GPT56SOL_GPFS_FULL:-$GPFS_DIR/a2e-gpt-5.6-sol-official-full-evaluated.db}"
GPFS_WORK_DB="${GPFS_WORK}/gpt56sol-full-writable.db"
EVAL_PY="$ROOT/eval/scripts/run_gpt56sol_official_sqlite_eval.py"

mkdir -p "$LOG_DIR" "$LOCAL_DIR" "$GPFS_DIR" "$GPFS_WORK"

pull_eval_db_from_gpfs() {
  [[ "$SYNC_GPFS" == "1" && -f "$GPFS_WORK_DB" ]] || return 0
  if [[ ! -f "$DB_FULL" || "$GPFS_WORK_DB" -nt "$DB_FULL" ]]; then
    echo "PULL $GPFS_WORK_DB -> $DB_FULL" | tee -a "$LOG_DIR/batch.log"
    rsync -a --inplace "$GPFS_WORK_DB" "$DB_FULL"
  fi
}

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

if [[ ! -f "$DB_FULL" ]]; then
  src_db="${AEP_GPT56SOL_SRC_DIR:-$SRC_ROOT/trajectories}/a2e-gpt-5.6-sol-official-full.db"
  if [[ -f "$GPFS_WORK_DB" ]]; then
    pull_eval_db_from_gpfs
  elif [[ -f "$src_db" ]]; then
    mkdir -p "$LOCAL_DIR"
    echo "seed local staging from read-only source: $src_db -> $DB_FULL" | tee -a "$LOG_DIR/batch.log"
    cp -f "$src_db" "$DB_FULL"
  else
    echo "error: no DB at $DB_FULL, $GPFS_WORK_DB, or $src_db" >&2
    exit 1
  fi
else
  pull_eval_db_from_gpfs
fi

sync_eval_db_to_gpfs() {
  local db="$1"
  [[ "$SYNC_GPFS" == "1" ]] || return 0
  mkdir -p "$GPFS_DIR" "$GPFS_WORK"
  sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
  rsync -a --inplace "$db" "$GPFS_WORK_DB"
  rsync -a --inplace "$db" "$GPFS_FULL"
  echo "SYNC $db -> $GPFS_WORK_DB + $GPFS_FULL" | tee -a "$LOG_DIR/batch.log"
}

run_sqlite_eval() {
  # Uses server uv env for Python deps only; no a2e serve / HTTP API.
  cd "$ROOT/server"
  uv run python "$EVAL_PY" "$@"
}

mapfile -t EXP_IDS < <(
  python3 - "$DB_FULL" "$FULL_METRICS" <<'PY'
import sqlite3, sys

db, full_metrics = sys.argv[1], int(sys.argv[2])
conn = sqlite3.connect(db)
rows = []
for eid, name, runs in conn.execute(
    """
    SELECT e.id, e.name,
           (SELECT COUNT(*) FROM experiment_runs er WHERE er.experiment_id = e.id)
    FROM experiments e
    ORDER BY e.id
    """
):
    full_runs = conn.execute(
        """
        SELECT COUNT(*) FROM experiment_runs er
        WHERE er.experiment_id = ?
          AND (
            SELECT COUNT(DISTINCT a.name)
            FROM experiment_run_annotations a
            WHERE a.experiment_run_id = er.id
          ) >= ?
        """,
        (eid, full_metrics),
    ).fetchone()[0]
    if full_runs < runs:
        rows.append((runs - full_runs, runs, eid, name))

# Smaller remaining work first (finish partial tau/gdpval before deepsearchqa).
rows.sort(key=lambda x: (x[0], x[2]))
for remaining, _runs, eid, _name in rows:
    print(eid)
PY
)

if [[ "${#EXP_IDS[@]}" -eq 0 ]]; then
  echo "all experiments complete (>= ${FULL_METRICS} metrics per run): $DB_FULL" | tee "$LOG_DIR/batch.log"
  exit 0
fi

echo "gpt-5.6-sol resume eval (direct SQLite, no server)" | tee "$LOG_DIR/batch.log"
echo "AEP_GPFS_ROOT=$AEP_GPFS_ROOT" | tee -a "$LOG_DIR/batch.log"
echo "SRC_ROOT=$SRC_ROOT (read-only)" | tee -a "$LOG_DIR/batch.log"
echo "DB_FULL=$DB_FULL" | tee -a "$LOG_DIR/batch.log"
echo "GPFS_FULL=$GPFS_FULL" | tee -a "$LOG_DIR/batch.log"
echo "LOG_DIR=$LOG_DIR incomplete=${#EXP_IDS[@]} exp_parallel=$EXP_PARALLEL concurrency=$CONCURRENCY force=$EVAL_FORCE sync_gpfs=$SYNC_GPFS" | tee -a "$LOG_DIR/batch.log"
printf 'RESUME_IDS=%s\n' "${EXP_IDS[*]}" | tee -a "$LOG_DIR/batch.log"

force_flag=()
[[ "$EVAL_FORCE" == "1" ]] && force_flag=(--force)

fail=0
running=0
for eid in "${EXP_IDS[@]}"; do
  while (( running >= EXP_PARALLEL )); do
    if wait -n; then
      :
    else
      fail=1
    fi
    running=$((running - 1))
  done
  safe="exp${eid}"
  echo "LAUNCH experiment_id=$eid" | tee -a "$LOG_DIR/batch.log"
  (
    run_sqlite_eval \
      --db "$DB_FULL" \
      --experiment-id "$eid" \
      --part all \
      "${force_flag[@]}" \
      --concurrency "$CONCURRENCY" \
      --log-level INFO \
      >>"${LOG_DIR}/official-full.${safe}.log" 2>&1
    sync_eval_db_to_gpfs "$DB_FULL"
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

sync_eval_db_to_gpfs "$DB_FULL"

if [[ "$fail" -eq 0 ]]; then
  echo "done OK: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
else
  echo "done WITH FAILURES: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
  exit 1
fi
