#!/usr/bin/env bash
# TB21 16-harness offline SQLite eval for the 26-metric matrix (V2_Matrix + submitted/redcode/etc).
#
# GPFS sqlite is unreliable — copy each DB to /tmp, eval locally, rsync back.
# Conservative parallelism to avoid OOM (dev machine previously crashed at 16×16).
#
# Usage:
#   bash scripts/run_tb21_v26_eval_parallel.sh
#   AEP_EVAL_DB_PARALLEL=2 AEP_EVAL_CONCURRENCY=8 bash scripts/run_tb21_v26_eval_parallel.sh
#
# Env:
#   AEP_EVAL_DB_PARALLEL   concurrent DB jobs (default 2)
#   AEP_EVAL_CONCURRENCY   per-DB async concurrency (default 8)
#   AEP_EVAL_FORCE=1       re-score all 26 metrics
#   AEP_EVAL_ENV_FILE      judge credentials

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPFS_ROOT="/mnt/shared-storage-gpfs2/agenteval/yuchenyue/tb21-eval"
WORK_DIR="${AEP_TB21_WORK_DIR:-/tmp/tb21_v26_work}"
LOG_DIR="${AEP_TB21_EVAL_LOG_DIR:-$ROOT/eval/outputs/tb21_v26_$(date +%Y%m%d_%H%M%S)}"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
EVAL_PY="$ROOT/eval/scripts/run_gpt56sol_official_sqlite_eval.py"
DB_PARALLEL="${AEP_EVAL_DB_PARALLEL:-2}"
CONCURRENCY="${AEP_EVAL_CONCURRENCY:-8}"
EVAL_FORCE="${AEP_EVAL_FORCE:-0}"

mkdir -p "$LOG_DIR" "$WORK_DIR"

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
  echo "error: OPENAI_API_KEY not set (source $ENV_FILE)" >&2
  exit 1
fi

mapfile -t JOBS < <(
  {
    for f in "$GPFS_ROOT"/glm-5.3/a2e-tb21-glm-5.3-*.db; do
      [[ -f "$f" ]] || continue
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      echo "$sz|$f"
    done
    for f in "$GPFS_ROOT"/gpt-5.6-sol/a2e-tb21-gpt-5.6-sol-*.db; do
      [[ -f "$f" ]] || continue
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      echo "$sz|$f"
    done
  } | sort -t'|' -k1,1n | cut -d'|' -f2-
)

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "error: no DBs under $GPFS_ROOT" >&2
  exit 1
fi

force_flag=()
[[ "$EVAL_FORCE" == "1" ]] && force_flag=(--force)

eval_one_db() {
  local src="$1"
  local name
  name=$(basename "$src")
  local local_db="$WORK_DIR/$name"
  local log="$LOG_DIR/${name%.db}.log"

  echo "COPY $name" | tee -a "$LOG_DIR/batch.log"
  cp -f "$src" "$local_db"

  echo "EVAL $name conc=$CONCURRENCY" | tee -a "$LOG_DIR/batch.log"
  (
    cd "$ROOT/server"
    uv run python "$EVAL_PY" \
      --db "$local_db" \
      --all-experiments \
      --part all \
      "${force_flag[@]}" \
      --concurrency "$CONCURRENCY" \
      --log-level INFO \
      >>"$log" 2>&1
  )

  sqlite3 "$local_db" "PRAGMA wal_checkpoint(TRUNCATE);" >/dev/null 2>&1 || true
  rsync -a --inplace "$local_db" "$src"
  rm -f "$local_db" "${local_db}-wal" "${local_db}-shm"
  echo "OK $name -> $src" | tee -a "$LOG_DIR/batch.log"
}

echo "TB21 v26 eval: ${#JOBS[@]} DB(s) db_parallel=$DB_PARALLEL concurrency=$CONCURRENCY force=$EVAL_FORCE" | tee "$LOG_DIR/batch.log"
echo "LOG_DIR=$LOG_DIR WORK_DIR=$WORK_DIR" | tee -a "$LOG_DIR/batch.log"

fail=0
running=0
for src in "${JOBS[@]}"; do
  while (( running >= DB_PARALLEL )); do
    if ! wait -n; then
      fail=1
    fi
    running=$((running - 1))
  done
  eval_one_db "$src" &
  running=$((running + 1))
done

while (( running > 0 )); do
  if ! wait -n; then
    fail=1
  fi
  running=$((running - 1))
done

if [[ "$fail" -eq 0 ]]; then
  echo "done OK: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
else
  echo "done WITH FAILURES: $LOG_DIR" | tee -a "$LOG_DIR/batch.log"
  exit 1
fi
