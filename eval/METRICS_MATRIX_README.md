# A2E Eval Metric Matrix (27 Metrics)

**English** | [中文](METRICS_MATRIX_README_zh.md)

This document describes the **current 27-metric evaluation matrix** used by A2E offline eval (`eval/scripts/run_eval.py` → `eval/core/deal_server.py`). It explains what each metric means, how it is implemented, which top-venue benchmarks inspired it, and **why it helps diagnose agent failures** instead of only reporting pass/fail.

> **Catalog source of truth:** `eval/metrics_catalog.json` (includes a formal `bibliography` with `cite_key` entries) and `eval/core/metric_groups.py`  
> **Unscored rule:** `label=unscored`, `score=null` — excluded from averages (missing evidence or N/A).

---

## 1. Why this matrix is diagnostic

A single **correctness** bit cannot tell you *why* an agent failed. The matrix splits failures into separable causes:

| Failure story | Metrics that expose it |
|---|---|
| Plan was wrong before any tool ran | `plan_*` |
| Tool args invalid vs environment rejected execution | `tool_invocation` vs `tool_execution_error_rate` |
| Stuck repeating the same tool | `repeated_tool_call_rate` |
| Succeeded but paid for empty-tool-call turns | `idle_turn_count` |
| Never handed in a final artifact | `submitted=empty` |
| Handed in something but verifier says wrong | `submitted=1` + `correctness=0` |
| Crashed / timed out without finishing | `task_completion` |
| Unsafe shell patterns in tool payloads | `redcode_risky_operation_count` |
| Answer not faithful to context | `hallucination` (safety) |

**Design principles**

1. **Outcome vs process:** `correctness` / `task_completion` / `submitted` are separate on purpose (SWE-bench / SPA-Bench split).
2. **Counts vs rates:** Several tool/safety metrics return **integer counts** (higher = worse), not 0–1 quality scores.
3. **LLM + CODE hybrid:** LLM judges semantic behavior; CODE locates evidence deterministically where possible (loops, exec failures, RedCode patterns, token sums).
4. **No fake zeros:** If evidence is missing, emit **unscored** — do not treat N/A as 0 or 1.

---

## 2. How to run

```bash
cd server
uv run python ../eval/scripts/run_eval.py \
  --base-url http://localhost:6006 \
  --experiment-id <Experiment:id> \
  --part all
```

Parts: `plan`, `tool`, `correct`, `efficiency`, `safety`, or `all` (27 metrics).

Direct SQLite writeback (no server): `eval/scripts/run_gpt56sol_official_sqlite_eval.py`.

---

## 3. Code map

| Area | Files |
|---|---|
| Groups / registry | `eval/core/metric_groups.py`, `eval/metrics_catalog.json` |
| Orchestration | `eval/core/deal_server.py`, `eval/scripts/run_eval.py` |
| Span enrichment | `eval/core/eval_common.py`, `eval/core/span_store.py` |
| Plan | `eval/process_values/plan_eval.py` |
| Tool | `eval/process_values/tool_eval.py` |
| Correct / submit | `eval/process_values/correct_eval.py`, `eval/process_values/delivery_eval.py` |
| Efficiency | `eval/result_values/efficiency_eval.py`, `eval/core/trajectory_token_usage.py` |
| Safety | `eval/result_values/safety_eval.py` |
| Tests | `eval/test/test_*.py` |

---

## 4. Metrics by group

### 4.1 Plan (5 metrics, LLM)

**Benchmark lineage:** [Agent Planning Benchmark (APB)](https://arxiv.org/abs/2408.03326) — plan grading and error taxonomy E1–E6.

| Metric | Plain meaning | Implementation | Diagnostic value |
|---|---|---|---|
| **plan_grade** | Overall plan quality (6 bands: perfect → failed) | LLM judge with APB rubric; scores 1.0, 0.8, …, 0.0 | Separates “plan fundamentally broken” from “execution noise” |
| **plan_goal_alignment** | Does the plan target the user’s actual goal? | LLM; inverse of APB **E1** goal misunderstanding | Catches wrong-problem plans that still look coherent |
| **plan_completeness** | Does the plan cover required steps without stopping early? | LLM; inverse of APB **E2** premature conclusion | Distinguishes incomplete planning from incomplete execution |
| **plan_constraint_adherence** | Does the plan respect explicit constraints? | LLM; inverse of APB **E3** constraint violation | Flags format/tool/time/method violations in the plan text |
| **plan_hallucination** | Does the plan invent tools or unsupported facts? | LLM; inverse of APB **E6** | Not answer faithfulness — only plan-level fabrication |

**Logic:** `_plan_context()` gathers instruction, tools, trajectory, final answer → `_text_judge()` (`plan_eval.py`).

---

### 4.2 Tool (6 metrics)

| Metric | Kind | Plain meaning | Implementation | Paper / lineage | Diagnostic value |
|---|---|---|---|---|---|
| **repeated_tool_call_rate** | LLM | How many **distinct tool names** were stuck in a dead loop? (0 = clean) | CODE locates repeated/near-repeated TOOL windows → one LLM confirms “no information gain” | Hu et al. (2026) RedundancyBench; Zhou et al. (2024) WebArena | Separates infinite retry from high `tool_call_count` with progress |
| **tool_invocation** | LLM | Are tool **arguments** valid vs schemas? | LLM compares each call to declared parameter schemas | Tool schema evaluation tradition (BFCL-style pre-exec) | “Wrong args” vs “right args, bad environment” |
| **tool_execution_error_rate** | CODE | Count of tool calls the **environment did not complete** | Classify each tool `result` → `runtime`, `timeout`, `harness_reject`, etc.; score = failure count | **BFCL** (ICML 2025) Executable Function Evaluation | TB21: bash “command not found”, duplicate-call rejects — args can be fine |
| **tool_call_count** | CODE | Total tool calls in trajectory | Count `tool_calls_full` / spans | — | Effort / thrashing magnitude (not quality) |
| **self_correction_rate** | CODE | After a tool error, did the agent retry successfully? | Find error indices; check later same-name success; ratio | Recovery / reflexion-style behavior | Errors that never get retried |
| **tool_recall** | CODE | Fraction of `expected_actions` observed | Set overlap; optional LLM fallback if GT empty | Action-sequence recall | **Often N/A** on TB21 (no expected_actions) |

**Unscored (common):** no TOOL spans (crewai/smolagents), or calls without `name`+`arguments` (loop) / without `result` (exec error).

---

### 4.3 Correct (3 metrics)

| Metric | Plain meaning | Implementation | Paper / lineage | Diagnostic value |
|---|---|---|---|---|
| **correctness** | Is the **final answer / verifier outcome** correct? | Adaptive: rule (MC/numeric/resolved) or LLM Phoenix-GT / Phoenix Correctness | Phoenix evaluators; SWE-bench **resolved** for patch benchmarks | Ground-truth task success |
| **task_completion** | Did the **harness finish** (`status==ok`)? | CODE reads `task_output.status` | Harness completion vs correctness split | Timeout/crash vs wrong answer |
| **submitted** | Did the agent emit a **candidate deliverable**? | CODE locators: `final_answer`, git patch, stop/submit tools, written task files | **SWE-bench** (ICLR 2024) submitted vs empty_patch; **GAIA** FINAL ANSWER; **WebArena** stop; **SPA-Bench** (ICLR 2025) completion vs success | **`submitted=0`**: stopped without handing in; **`submitted=1` + `correctness=0`**: handed in but wrong |

**Logic:** `delivery_eval.py` (`make_submitted`) — independent of verifier.

---

### 4.4 Efficiency (6 metrics)

| Metric | Kind | Plain meaning | Implementation | Paper / lineage | Diagnostic value |
|---|---|---|---|---|---|
| **conciseness** | LLM | Is the final answer unnecessarily verbose? | LLM on final answer text | — | Communication cost |
| **total_token_usage** | CODE | Token count (span sum preferred; harness or trajectory-fair fallback) | `_token_usage_from_task_output` → span sum → cl100k_base estimate | Phoenix telemetry scale | Cost drivers; N/A when spans missing |
| **cost** | CODE | USD cost if reported | Sum cost fields in spans / task_output | — | Often N/A on TB21 |
| **turn_count** | CODE | Conversation turns | Count turns in trajectory | — | Interaction depth |
| **idle_turn_count** | CODE | Agent steps with **empty tool calls** (message-only turns) | Count assistant messages with empty `tool_calls`; fallback `turns − tool_steps` | Hu et al. (2026) RedundancyBench Sec. 3; Appendix C | Billable dialogue / user-sim overhead vs tool-progress turns |
| **wall_time** | CODE | Wall-clock seconds | Harness or span timestamps | — | Slow runs vs many cheap turns |

---

### 4.5 Safety (7 metrics)

| Metric | Kind | Plain meaning | Implementation | Paper / lineage | Diagnostic value |
|---|---|---|---|---|---|
| **hallucination** | LLM | Is the **final answer** faithful to provided context? | LLM faithfulness judge | Phoenix / RAG faithfulness line | Confident wrong claims in the answer |
| **privacy_leakage** | LLM | Did the agent leak sensitive info? | LLM | Safety rubrics | Exfil in natural language |
| **unauthorized_action** | LLM | Out-of-scope actions (scope, not harm) | LLM | Scope violation | Wrong permissions / policy |
| **harmful_action** | LLM | Safety-harmful actions | LLM | Harm class | Distinct from scope |
| **failure_transparency** | LLM | Does the agent admit failure honestly? | LLM; N/A if no failure | Hides errors from user |
| **prompt_injection_resilience** | LLM | Resists injection in context | LLM; N/A without injection signal | Injection benchmarks | **N/A on TB21** (no injection setup) |
| **redcode_risky_operation_count** | CODE | Count of distinct **RedCode-Exec** risky scenarios in tool payloads | Regex scan of TOOL arguments (12 scenarios); task-necessary pip/chmod exempt | **RedCode-Exec** (NeurIPS 2024) | Dangerous shell patterns even when task “succeeds” |

---

## 5. What changed from the older 23-metric set

| Change | Detail |
|---|---|
| **Added** | `repeated_tool_call_rate`, `tool_execution_error_rate`, `submitted`, `redcode_risky_operation_count`, `idle_turn_count` |
| **Removed** | `tool_hallucination`, `os_harm_misbehavior` |
| **Group moves** | `conciseness` under efficiency; faithfulness only as `hallucination` under safety (no separate memory group) |
| **Total** | **27** deduped metrics (15 LLM + 12 CODE) |

---

## 6. Reading TB21 results (example)

From 1296 Terminal-Bench runs (16 harnesses × 81 tasks):

- **Mean correctness ~57%** — many failures are not crashes.
- **`submitted` ~92%** among scored runs — most agents hand *something* in; **~90 failures** are `empty` (never submitted).
- **`submitted=1` + wrong** — majority of incorrect runs (execution/quality, not “forgot to submit”).
- **`tool_execution_error_rate` ~6** — average ~6 tool exec failures per scored trajectory (bash errors, rejects).
- **`repeated_tool_call_rate` ~0.01** — rare formal dead loops; thrashing shows up in call count + exec errors instead.
- **Unscored hotspots:** crewai/smolagents missing TOOL span details; `tool_recall`, `cost`, `prompt_injection_resilience` designed N/A.

---

## 7. References

Formal entries live in `eval/metrics_catalog.json` → `bibliography` (referenced by `cite_key` in each metric’s `paper_refs`).

1. **Chang et al. (2024).** *Agent Planning Benchmark: Evaluating LLM Agents on Real-World Planning Tasks.* arXiv:2408.03326. — `plan_grade`, `plan_goal_alignment`, `plan_completeness`, `plan_constraint_adherence`, `plan_hallucination` (APB E1–E6).
2. **Jimenez et al. (2024).** *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770. — `submitted` vs empty_patch; `correctness` resolved.
3. **Mialon et al. (2024).** *GAIA: A Benchmark for General AI Assistants.* ICLR 2024. arXiv:2311.12983. — `FINAL ANSWER` / `final_answer` submit channel.
4. **Zhou et al. (2024).** *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024. arXiv:2307.13854. — stop/submit actions; repeating-equivalent-action baseline for loops.
5. **Chen et al. (2025).** *SPA-Bench: A Comprehensive Benchmark for SmartPhone Agent Evaluation.* ICLR 2025. arXiv:2410.15164. — completion (`submitted`) independent of success.
6. **Patil et al. (2025).** *Berkeley Function Calling Leaderboard (BFCL).* ICML 2025. — Executable Function Evaluation tradition (`tool_execution_error_rate`, `tool_invocation`).
7. **Hu et al. (2026).** *Redundant or Necessary? A Benchmark for Detecting Redundant Steps in Agent Trajectories.* arXiv:2605.29893. — `repeated_tool_call_rate` (Sec. 3 dead-loop definition); `idle_turn_count` (Sec. 3 empty-tool-call steps; Appendix C message-only turns).
8. **Wang et al. (2024).** *RedCode: Risky Code Execution and Generation Benchmark for Code Agents.* NeurIPS 2024. — risky-operation taxonomy (`redcode_risky_operation_count`).
9. **Arize AI (2024).** *Phoenix: Open-Source LLM Tracing and Evaluation.* — adaptive `correctness`; answer faithfulness (`hallucination`).

---

## 8. Tests

```bash
cd eval && python -m pytest test/test_submitted.py test/test_tool_execution_error_rate.py \
  test/test_repeated_tool_call_rate.py test/test_redcode_risky_operation_count.py \
  test/test_idle_turn_count.py -q
```

Each new CODE metric has dedicated unit tests; LLM metrics use stub judges in tests where present.
