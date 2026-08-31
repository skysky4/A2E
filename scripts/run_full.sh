#!/usr/bin/env bash
# Full-split official cell. Same evaluators / seed / budgets as run_n1.sh.
# Usage: bash scripts/run_full.sh <agent> <dataset> [extra args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
agent="${1:?agent}"
dataset="${2:?dataset}"
shift 2

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

case "$dataset" in
  deepsearchqa) evals="deepsearch_grader" ;;
  tau-bench|tau2|tau3|tau3bench|tau3-bench) evals="tau_grader" ;;
  gdpval-aa) evals="gdp_grader" ;;
  *) echo "unknown full-run dataset: $dataset" >&2; exit 2 ;;
esac

extra=()
case "$dataset" in
  tau-bench|tau2|tau3|tau3bench|tau3-bench) extra+=(--domain retail) ;;
esac

case "$dataset" in
  tau-bench|tau2|tau3|tau3bench|tau3-bench) wall=1200; turns=30; llm_to=180 ;;
  gdpval-aa) wall=1800; turns=8; llm_to=600; export A2E_MAX_TOKENS="${A2E_MAX_TOKENS:-16384}" ;;
  *) wall=720; turns=8; llm_to=180 ;;
esac
export A2E_LLM_TIMEOUT="${A2E_LLM_TIMEOUT:-$llm_to}"
export A2E_MAX_TURNS="${A2E_MAX_TURNS:-$turns}"
export A2E_MAX_STEPS="${A2E_MAX_STEPS:-$turns}"
export A2E_RUN_DEADLINE="${A2E_RUN_DEADLINE:-$((wall - 100))}"
export A2E_AGNO_DEADLINE="${A2E_AGNO_DEADLINE:-$A2E_RUN_DEADLINE}"

cd "$ROOT/task"
exec uv run --frozen python examples/run_experiment.py \
  --dataset "$dataset" \
  --agent "$agent" \
  --full \
  --evaluators "$evals" \
  --sample-seed 20260816 \
  "${extra[@]}" \
  "$@"
