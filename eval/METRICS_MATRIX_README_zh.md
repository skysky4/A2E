# A2E 评测指标矩阵（27 项）

[English](METRICS_MATRIX_README.md) | **中文**

本文档说明 A2E 当前使用的 **27 项评测矩阵**（入口：`eval/scripts/run_eval.py` → `eval/core/deal_server.py`）。对每一项说明：**通俗含义、代码怎么算、参考了哪些顶会工作、以及为什么能用来诊断 agent 失败**（而不只是报对错）。

> **权威目录：** `eval/metrics_catalog.json`（含正式 `bibliography` 与 `cite_key` 引用）、`eval/core/metric_groups.py`  
> **unscored 规则：** `label=unscored`、`score=null` — 不进均值（缺证据或不适用的样本）。

---

## 1. 为什么这套矩阵能「诊断」

单看 **correctness** 只知道「错了」，不知道 **错在哪一环**。矩阵把失败拆成可区分的因果：

| 失败故事 | 看哪些指标 |
|---|---|
| 计划阶段就偏了 | `plan_*` 五条 |
| 参数不对 vs 环境执行失败 | `tool_invocation` vs `tool_execution_error_rate` |
| 同一工具死循环 | `repeated_tool_call_rate` |
| 任务成功但空 tool call 轮次多 | `idle_turn_count` |
| 从没交出最终产物 | `submitted=empty` |
| 交了但答案/验证不过 | `submitted=1` + `correctness=0` |
| 超时/崩溃没跑完 | `task_completion` |
| 工具载荷里有危险 shell 模式 | `redcode_risky_operation_count` |
| 最终回答不忠实于上下文 | `hallucination`（safety 组） |

**设计原则**

1. **结果 vs 过程：** `correctness` / `task_completion` / `submitted` 故意分开（对齐 SWE-bench / SPA-Bench）。
2. **计数 vs 比率：** 部分 tool/safety 指标是 **整数次数**（越高越差），不是 0–1 质量分。
3. **LLM + CODE：** 语义类用 LLM；能定位证据的用 CODE（循环、执行失败、RedCode 模式、token 求和）。
4. **不造假分：** 缺证据就 **unscored**，不要把 N/A 当 0 或 1。

---

## 2. 怎么跑

```bash
cd server
uv run python ../eval/scripts/run_eval.py \
  --base-url http://localhost:6006 \
  --experiment-id <Experiment:id> \
  --part all
```

可选 part：`plan`、`tool`、`correct`、`efficiency`、`safety`，或 `all`（27 项全跑）。

离线 SQLite 写回（不启 server）：`eval/scripts/run_gpt56sol_official_sqlite_eval.py`。

---

## 3. 代码地图

| 模块 | 文件 |
|---|---|
| 分组 / 注册 | `eval/core/metric_groups.py`、`eval/metrics_catalog.json` |
| 编排 | `eval/core/deal_server.py`、`eval/scripts/run_eval.py` |
| Span  enrichment | `eval/core/eval_common.py`、`eval/core/span_store.py` |
| 规划 | `eval/process_values/plan_eval.py` |
| 工具 | `eval/process_values/tool_eval.py` |
| 正确 / 交付 | `eval/process_values/correct_eval.py`、`eval/process_values/delivery_eval.py` |
| 效率 | `eval/result_values/efficiency_eval.py`、`eval/core/trajectory_token_usage.py` |
| 安全 | `eval/result_values/safety_eval.py` |
| 测试 | `eval/test/test_*.py` |

---

## 4. 分组说明

### 4.1 Plan（规划，5 项，LLM）

** lineage：** [Agent Planning Benchmark (APB)](https://arxiv.org/abs/2408.03326) — 计划分级与 E1–E6 错误类型。

| 指标 | 通俗说法 | 实现 | 诊断价值 |
|---|---|---|---|
| **plan_grade** | 计划总体好不好（perfect → failed 六档） | LLM + APB  rubric；1.0/0.8/…/0.0 | 区分「计划本身烂」和「执行噪音」 |
| **plan_goal_alignment** | 计划是否对准用户真实目标？ | LLM；APB **E1** 目标理解反判 | 抓「题意理解错了但写得像样」 |
| **plan_completeness** | 该有的步骤有没有、会不会计划层面就收尾？ | LLM；APB **E2** 完整性/过早结束 | 计划不完整 vs 执行没做完 |
| **plan_constraint_adherence** | 有没有违反题目写明的约束？ | LLM；APB **E3** 约束违反 | 格式/工具/范围违规在计划里就能看见 |
| **plan_hallucination** | 计划里有没有编造工具或无根据内容？ | LLM；APB **E6** | 只评计划幻觉，不是最终答案忠实度 |

**逻辑：** `_plan_context()` 拼 instruction、工具、轨迹、final answer → `_text_judge()`（`plan_eval.py`）。

---

### 4.2 Tool（工具，6 项）

| 指标 | 类型 | 通俗说法 | 实现 | 顶会参考 | 诊断价值 |
|---|---|---|---|---|---|
| **repeated_tool_call_rate** | LLM | 有几种 tool **陷进死循环**？（0=没有） | CODE 找重复 TOOL 窗口 → LLM 确认「无信息增益」 | Hu et al. (2026) RedundancyBench；Zhou et al. (2024) WebArena | 和「调用很多次但有进展」区分开 |
| **tool_invocation** | LLM | 每次调用的 **参数** 是否符合 schema？ | LLM 对照参数 schema | BFCL 类「执行前」检查 | 「参数错」vs「参数对但环境报错」 |
| **tool_execution_error_rate** | CODE | **环境没跑完** 的 tool 次数 | 解析每个 `result` → runtime/timeout/harness_reject…；score=失败次数 | **BFCL** (ICML 2025) Executable Function Evaluation | TB21 常见：bash command not found、duplicate call 被拒 |
| **tool_call_count** | CODE | 一共调了多少次 tool | 数 `tool_calls_full` / spans | — | 工作量/刷屏，不是质量 |
| **self_correction_rate** | CODE | tool 报错后有没有成功重试？ | 找 error 下标；看后面同名 tool 是否成功；比例 | 恢复/反思类行为 | 错了就放弃 vs 会修 |
| **tool_recall** | CODE | 相对 `expected_actions` 召回了多少 | 集合交；GT 空时可 LLM 兜底 | 动作序列 recall | TB21 常 **N/A**（无 expected_actions） |

 

---

### 4.3 Correct（正确性 / 交付，3 项）

| 指标 | 通俗说法 | 实现 | 顶会参考 | 诊断价值 |
|---|---|---|---|---|
| **correctness** | **最终答案/验证器** 对不对？ | 自适应：规则(MC/数值/resolved) 或 LLM Phoenix-GT / Phoenix | Phoenix；SWE-bench **resolved** | 任务 ground truth |
| **task_completion** | **Harness 是否跑完**（status==ok）？ | CODE 读 `task_output.status` | 完成 vs 正确分离 | 超时/崩溃 vs 答错 |
| **submitted** | 有没有交出 **候选终产物**？ | CODE：final_answer、patch、stop/submit、写任务文件 | **SWE-bench** (ICLR 2024) submitted/empty_patch；**GAIA** FINAL ANSWER；**WebArena** stop；**SPA-Bench** (ICLR 2025) 完成≠成功 | **empty**：没交就停；**交了但错**：`submitted=1` + `correctness=0` |

**逻辑：** `delivery_eval.py`（`make_submitted`），与 verifier 无关。

---

### 4.4 Efficiency（效率，6 项）

| 指标 | 类型 | 通俗说法 | 实现 | 顶会参考 | 诊断价值 |
|---|---|---|---|---|---|
| **conciseness** | LLM | 最终回答是否啰嗦 | LLM 评 final answer | — | 沟通成本 |
| **total_token_usage** | CODE | token 总量 | 优先 span 求和；否则 harness 或 trajectory-fair 估计 | Phoenix 遥测量级 | 贵在哪；缺 span 则 unscored |
| **cost** | CODE | 美元花费 | span/task_output 里的 cost | — | TB21 常 N/A |
| **turn_count** | CODE | 对话轮数 | 数 turns | — | 交互深度 |
| **idle_turn_count** | CODE | **空 tool call** 的 agent 步数 | 数 assistant 空 `tool_calls`；fallback `turns − tool_steps` | Hu et al. (2026) RedundancyBench Sec. 3；Appendix C | 对话/user-sim 开销 vs 有 tool 进展的轮次 |
| **wall_time** | CODE | 墙钟秒数 | harness 或 span 时间戳 | — | 慢 vs 轮数多 |

---

### 4.5 Safety（安全，7 项）

| 指标 | 类型 | 通俗说法 | 实现 | 顶会参考 | 诊断价值 |
|---|---|---|---|---|---|
| **hallucination** | LLM | **最终回答** 是否忠实于给定上下文 | LLM faithfulness | Phoenix / RAG 忠实度 | 自信胡说 |
| **privacy_leakage** | LLM | 是否泄露敏感信息 | LLM | 安全 rubric | 自然语言泄密 |
| **unauthorized_action** | LLM | 是否越权/超范围（管范围不管伤害） | LLM | 策略违规 | |
| **harmful_action** | LLM | 是否有害行为 | LLM | 伤害类 | 与越权分开 |
| **failure_transparency** | LLM | 失败时是否说清楚 | LLM；无失败则 N/A | 瞒错 |
| **prompt_injection_resilience** | LLM | 是否抗 prompt 注入 | LLM；无注入信号则 N/A | 注入 benchmark | **TB21 全 N/A** |
| **redcode_risky_operation_count** | CODE | TOOL 载荷命中几种 **RedCode-Exec** 危险场景 | 12 类 regex；任务必需的 pip/chmod 豁免 | **RedCode-Exec** (NeurIPS 2024) | 任务「成功」但命令很危险 |

---

## 5. 相对旧版 23 项的变更

| 变更 | 说明 |
|---|---|
| **新增** | `repeated_tool_call_rate`、`tool_execution_error_rate`、`submitted`、`redcode_risky_operation_count`、`idle_turn_count` |
| **删除** | `tool_hallucination`、`os_harm_misbehavior` |
| **分组** | `conciseness` 归入 efficiency；忠实度只在 safety 的 `hallucination`（无 memory 组） |
| **合计** | **27** 项（LLM 15 + CODE 12） |

---

## 6. TB21 上怎么读（实例）

1296 条（16 harness × 81 task）：

- **correctness 均值 ~57%** — 大量失败不是 crash。
- **submitted 在可评样本上 ~92%** — 多数会交东西；约 **90 条 empty** 是「没交就停」。
- **submitted=1 但错** — incorrect 里的大头（执行/质量，不是忘记提交）。
- **tool_execution_error_rate ~6** — 平均每条可评轨迹约 6 次 tool 执行失败。
- **repeated_tool_call_rate ~0.01** —  formal 死循环很少；刷屏更多体现在 call count + exec error。

---

## 7. 参考文献

正式条目见 `eval/metrics_catalog.json` → `bibliography`（各指标的 `paper_refs` 通过 `cite_key` 引用）。

1. **Chang et al. (2024).** *Agent Planning Benchmark: Evaluating LLM Agents on Real-World Planning Tasks.* arXiv:2408.03326. — 规划五条（APB E1–E6）。
2. **Jimenez et al. (2024).** *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770. — `submitted` / empty_patch；`correctness` resolved。
3. **Mialon et al. (2024).** *GAIA: A Benchmark for General AI Assistants.* ICLR 2024. arXiv:2311.12983. — `FINAL ANSWER` / `final_answer` 提交通道。
4. **Zhou et al. (2024).** *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024. arXiv:2307.13854. — stop/submit；重复等价动作（loop 基线）。
5. **Chen et al. (2025).** *SPA-Bench: A Comprehensive Benchmark for SmartPhone Agent Evaluation.* ICLR 2025. arXiv:2410.15164. — 完成（`submitted`）与成功分离。
6. **Patil et al. (2025).** *Berkeley Function Calling Leaderboard (BFCL).* ICML 2025. — 可执行函数评测（`tool_execution_error_rate`、`tool_invocation`）。
7. **Hu et al. (2026).** *Redundant or Necessary? A Benchmark for Detecting Redundant Steps in Agent Trajectories.* arXiv:2605.29893. — `repeated_tool_call_rate`（Sec. 3 死循环）；`idle_turn_count`（Sec. 3 空 tool call；Appendix C 纯消息轮）。
8. **Wang et al. (2024).** *RedCode: Risky Code Execution and Generation Benchmark for Code Agents.* NeurIPS 2024. — 危险操作 taxonomy（`redcode_risky_operation_count`）。
9. **Arize AI (2024).** *Phoenix: Open-Source LLM Tracing and Evaluation.* — adaptive `correctness`；答案忠实度（`hallucination`）。

---

## 8. 测试

```bash
cd eval && python -m pytest test/test_submitted.py test/test_tool_execution_error_rate.py \
  test/test_repeated_tool_call_rate.py test/test_redcode_risky_operation_count.py \
  test/test_idle_turn_count.py -q
```

新增 CODE 指标均有单测；LLM 指标在测试中尽量用 stub judge。
