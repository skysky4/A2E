#!/usr/bin/env bash
# Run V2 full eval on canonical GLM-5.2 3×9×50 campaign and write back to local A2E DB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="${AEP_GLM52_DB:-/home/yuchenyue/AEP-glm52-ae2-with-correctness.db}"
BASE_URL="${A2E_BASE_URL:-http://127.0.0.1:6006}"
ENV_FILE="${AEP_EVAL_ENV_FILE:-/home/yuchenyue/AE2-prep-e2e-latest/.env}"
LOG_DIR="${AEP_GLM52_EVAL_LOG_DIR:-/home/yuchenyue/AEP/eval/outputs/glm52_v2_writeback_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export A2E_EVAL_LLM_PROVIDER="${A2E_EVAL_LLM_PROVIDER:-${AE2_EVAL_LLM_PROVIDER:-openai}}"
export A2E_EVAL_LLM_MODEL="${A2E_EVAL_LLM_MODEL:-${AE2_EVAL_LLM_MODEL:-deepseek-v4-pro}}"
# c=8 ~1.1s/it on this host; c=10+ hits LLM timeouts/rate limits. Override: AEP_EVAL_CONCURRENCY.
export AEP_EVAL_CONCURRENCY="${AEP_EVAL_CONCURRENCY:-8}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${AE2_EVAL_LLM_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${AE2_EVAL_LLM_API_KEY:-}}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY / AE2_EVAL_LLM_API_KEY not set (source $ENV_FILE)" >&2
  exit 1
fi

mapfile -t EXP_IDS < <(
  AEP_GLM52_DB="$DB_PATH" python3 - <<'PY'
import base64
import os
import sqlite3

db = os.environ["AEP_GLM52_DB"]
conn = sqlite3.connect(db)
rows = conn.execute(
    """
    SELECT e.id, e.name FROM experiments e
    WHERE name LIKE '%glm-5.2%'
      AND (SELECT COUNT(*) FROM experiment_runs er WHERE er.experiment_id=e.id)=50
    ORDER BY e.id
    """
).fetchall()
latest: dict[str, int] = {}
for eid, name in rows:
    key = name.split("-glm-5.2-")[0]
    latest[key] = eid
for eid in latest.values():
    api_id = base64.b64encode(f"Experiment:{eid}".encode()).decode()
    print(api_id)
PY
)

echo "GLM52 DB: $DB_PATH"
echo "Experiments: ${#EXP_IDS[@]}"
echo "Logs: $LOG_DIR"

cd "$ROOT/server"
SKIP_LIST="${SKIP_EXP_IDS:-${SKIP_EXP_ID:-}}"
for exp_id in "${EXP_IDS[@]}"; do
  if [[ -n "$SKIP_LIST" && ",${SKIP_LIST}," == *",${exp_id},"* ]]; then
    echo "SKIP (already done or elsewhere) $exp_id" | tee -a "$LOG_DIR/batch.log"
    continue
  fi
  safe_name="${exp_id//\//_}"
  log_file="$LOG_DIR/${safe_name}.log"
  echo "=== eval $exp_id ===" | tee -a "$LOG_DIR/batch.log"
  if uv run python ../eval/scripts/run_eval.py \
    --base-url "$BASE_URL" \
    --experiment-id "$exp_id" \
    --part all \
    --force \
    --concurrency "$AEP_EVAL_CONCURRENCY" \
    --log-level INFO \
    >>"$log_file" 2>&1; then
    echo "OK $exp_id" | tee -a "$LOG_DIR/batch.log"
  else
    echo "FAIL $exp_id (see $log_file)" | tee -a "$LOG_DIR/batch.log"
  fi
done

echo "done: $LOG_DIR/batch.log"
