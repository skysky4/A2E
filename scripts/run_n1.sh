#!/usr/bin/env bash
# One n=1 official cell: per-dataset evaluators, sample seed, and wall budget.
# Usage: bash scripts/run_n1.sh <agent> <dataset> [extra args...]
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
  deepsearchqa) evals="deepsearch_match,tool_recall" ;;
  tau-bench|tau2|tau3|tau3bench|tau3-bench|traject-bench) evals="tool_recall" ;;
  mmlu|gpqa|mmlu-pro|arc-challenge|truthfulqa|agieval|commonsenseqa|hellaswag|openbookqa) evals="mc_letter" ;;
  gsm8k|math) evals="numeric_match" ;;
  humaneval) evals="humaneval_pass" ;;
  persistbench) evals="substring" ;;
  bbh) evals="exact_match" ;;
  gdpval-aa) evals="llm_judge" ;;
  gdpval) echo "dataset gdpval was removed; use gdpval-aa (HF openai/gdpval)" >&2; exit 2 ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac

extra=()
case "$dataset" in
  tau-bench|tau2|tau3|tau3bench|tau3-bench) extra+=(--domain retail) ;;
esac

case "$dataset" in
  tau-bench|tau2|tau3|tau3bench|tau3-bench) wall=1200; turns=30 ;;
  gdpval-aa) wall=1800; turns=8; export A2E_MAX_TOKENS="${A2E_MAX_TOKENS:-16384}"; export A2E_LLM_TIMEOUT="${A2E_LLM_TIMEOUT:-600}" ;;
  *) wall=720; turns=8 ;;
esac
export A2E_MAX_TURNS="${A2E_MAX_TURNS:-$turns}"
export A2E_MAX_STEPS="${A2E_MAX_STEPS:-$turns}"
export A2E_RUN_DEADLINE="${A2E_RUN_DEADLINE:-$((wall - 100))}"
export A2E_AGNO_DEADLINE="${A2E_AGNO_DEADLINE:-$A2E_RUN_DEADLINE}"

cd "$ROOT/task"
cmd=(uv run --frozen python examples/run_experiment.py
  --dataset "$dataset"
  --agent "$agent"
  --n 1
  --evaluators "$evals"
  --sample-seed 20260816
  "${extra[@]}"
  "$@")
if command -v timeout >/dev/null 2>&1; then
  exec timeout "$wall" "${cmd[@]}"
fi
exec "${cmd[@]}"
