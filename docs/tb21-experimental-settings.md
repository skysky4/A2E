# Terminal-Bench 2.1：GPT-5.6-sol / GLM-5.3 × 9 Harness 实验设置

> 本文说明 2026-08-17 至 2026-08-24 期间完成的 Terminal-Bench 2.1（下称 TB2.1）评测设置。内容由最终 17 个 SQLite 数据库、Claude SDK campaign lock/merge report、运行脚本、模型 profile、TB2.1 `task.toml` 和实际 Python 环境交叉核对而来。模型服务地址与密钥不记录在文档中。

## 1. 实验设计概览

| 项目 | 设置 |
|---|---|
| Benchmark | Terminal-Bench 2.1 |
| TB2.1 源版本 | vendored commit `5c8eadf1f393183288fa08b8f73ca9a469cc5e00` |
| 模型 | `glm-5.3`、`gpt-5.6-sol` |
| Harness | Agno、AutoGen AgentChat、Claude SDK、CrewAI、Google ADK、LangGraph、LlamaIndex、OpenAI Agents、SmolAgents |
| 设计 | 2 个模型 × 9 个 harness，完全交叉 |
| 原始任务数 | 89 |
| 最终任务集 | 排除 `security` 类别后剩余的 81 题 |
| 重复次数 | 每个 model–harness–task 组合 1 次 |
| 每个 cell 的样本数 | 81 |
| 最终运行数 | 18 × 81 = 1,458 |
| 主指标 | 官方 held-out tests 产生的二元 `tb_resolved` |
| 每次 LLM 请求的输出上限 | 4,096 tokens |
| Agent 循环上限 | 10,000 turns/steps，作为基本不生效的保险上限 |
| Agent wall-clock 上限 | 逐题读取 `task.toml` 的 `[agent].timeout_sec` |
| Verifier wall-clock 上限 | 逐题读取 `task.toml` 的 `[verifier].timeout_sec` |
| 单次 `bash` 工具调用上限 | 300 秒 |
| Sandbox task 自动重试 | 0；基础设施无效结果通过定向补跑修复，而不是自动重放普通失败 |

九个 harness 的内部 Agent loop 与 prompt 包装不同，这是实验中的自变量。任务、两种工具、单次输出上限、逐题 wall-clock 预算和官方 verifier 保持一致。

## 2. 任务集与抽样

TB2.1 本地版本共有 89 个任务。本实验排除了以下 8 个 `security` 任务：

- `break-filter-js-from-html`
- `crack-7z-hash`
- `filter-js-from-html`
- `fix-code-vulnerability`
- `openssl-selfsigned-cert`
- `password-recovery`
- `sanitize-git-repo`
- `vulnerable-secret`

剩余 81 题全部进入每个 model–harness cell，因此这里的 seed 只影响任务顺序，不影响任务成员：

| 运行组 | Seed | 任务成员 |
|---|---:|---|
| 8 个 OpenAI Chat Completions harness | `20260817` | 全部 81 个非 security 任务 |
| Claude SDK | `20260822` | 同一组 81 个非 security 任务 |

任务难度构成为 easy 4 题、medium 49 题、hard 28 题。类别构成为：software-engineering 26、scientific-computing 8、system-administration 9、data-science 8、debugging 5、file-operations 5、data-processing 4、mathematics 4、model-training 4、machine-learning 3，以及 data-querying、games、optimization、personal-assistant、video-processing 各 1 题。

## 3. 时间与资源约束

### 3.1 Agent 和 verifier 的逐题时间预算

Agent 与 verifier 的超时都由任务自己的 `task.toml` 决定，不按模型或 harness 改变。分布如下，其中“任务数”表示采用该预算的任务数量。

| Agent timeout | 任务数 | Verifier timeout | 任务数 |
|---:|---:|---:|---:|
| 600 s | 1 | 360 s | 1 |
| 750 s | 1 | 600 s | 1 |
| 900 s | 43 | 900 s | 42 |
| 1,200 s | 4 | 1,200 s | 5 |
| 1,800 s | 15 | 1,800 s | 16 |
| 2,400 s | 2 | 2,400 s | 2 |
| 3,600 s | 13 | 3,600 s | 12 |
| 7,200 s | 1 | 7,200 s | 1 |
| 12,000 s | 1 | 12,000 s | 1 |

直接八-harness runner 的外层 per-example timeout 按以下公式计算：

```text
outer_timeout = max(agent_timeout + verifier_timeout) + 300 s
              = 12,000 + 12,000 + 300
              = 24,300 s
```

这里的 24,300 秒只是防止外层调度器过早终止 sandbox、验证和清理流程；Agent 仍在自己的逐题 `agent_timeout_sec` 到达时停止，verifier 仍在自己的 `verifier_timeout_sec` 到达时停止。

Claude SDK campaign 的 `execution.timeout_seconds` 为 `null`，即不再增加 campaign 级的统一 Trial timeout；逐题 Agent/verifier timeout 仍生效。campaign 取消进程时的 grace period 为 30 秒。

### 3.2 单次模型请求与循环预算

| 预算 | 设置 | 说明 |
|---|---:|---|
| `max_tokens` | 4,096 | 每次 LLM 请求的 completion 上限，不是整条 trajectory 的总 token 上限 |
| `max_turns` / `max_steps` | 10,000 | TB2.1 数据集 override；让逐题 wall-clock 成为实际终止条件 |
| 显式 LLM request timeout | 180 s | Agno、AutoGen、Claude SDK、LangGraph 显式传入；其余 harness 沿用对应 SDK/provider 的请求默认值 |
| Agno LLM retries | 2 | Agno client 内部，独立于 task/campaign 重试 |
| GLM compatibility gateway upstream timeout | 900 s | gateway 到上游模型服务的保护上限 |
| GLM compatibility gateway retries | 2 | 对 429/500/502/503/504 等可重试错误，指数退避起点 0.5 s |

不要把 180 秒的“单次 LLM HTTP 请求 timeout”与逐题 Agent wall-clock 混为一谈。一个 Agent task 可以发起多次 LLM 请求和多次工具调用，只要总执行时间不超过该题的 Agent timeout。

### 3.3 Sandbox 资源

每题使用其发布的 Docker image，资源同样读取 `task.toml`：

| 资源 | 分布/设置 |
|---|---|
| CPU | 1 core × 75 题；2 cores × 3 题；4 cores × 3 题 |
| 内存 | 2,048 MB × 62 题；4,096 MB × 12 题；8,192 MB × 7 题 |
| GPU | 所有 81 题均为 0 |
| Internet | 所有 81 题均允许 |
| Storage 声明 | 所有任务均为 10,240 MB |
| Environment build timeout 声明 | 所有任务均为 600 s |

评测直接启动发布好的 Docker image，没有在正式运行中重新 build image，因此 `build_timeout_sec=600` 没有成为 Agent 阶段的额外时间限制。Docker runner 实际传入了 CPU、内存、GPU、网络和工作目录设置；`storage_mb` 仅记录，没有强制为 Docker 根文件系统施加 quota，因为这一能力依赖宿主机 storage driver。

## 4. Agent 可见的统一任务接口

每个 harness 都收到同一条 TB2.1 task instruction 和同一个 `AgentBinding`。基础 system prompt 的核心约束为：

- Agent 是在真实 Linux Docker container 中工作的工程师。
- 先执行 `pwd` 和 `ls` 确认工作目录。
- 通过工具探索、构建、运行程序和编辑文件。
- 所有产物必须写到题目指定的准确路径。
- Agent 结束后才会运行不可见的 held-out tests。

Agent 可见两种 sandbox-backed 工具：

| 工具 | 行为 | 限制 |
|---|---|---|
| `bash(command)` | 在任务 Docker container 的 WORKDIR 中执行 `bash -lc` | 每次调用 300 s；返回 stdout 最后 8,000 字符、stderr 最后 4,000 字符和 exit code |
| `str_replace_editor` | 支持 `view`、`create`、唯一字符串 `str_replace`、按行 `insert` | 文件 view 最多返回最后 8,000 字符；`str_replace` 要求旧字符串唯一 |

各 SDK 会把这两个 OpenAI JSON Schema 工具转换成框架原生工具，因此 schema 语义一致，但 framework 自己的 tool wrapper、消息格式、循环控制和默认 prompt 仍是 harness treatment 的一部分。

## 5. 模型与协议设置

### 5.1 模型 profile

| 字段 | `glm-5.3` | `gpt-5.6-sol` |
|---|---|---|
| Provider 标识 | `zai` | `openai-compatible` |
| 上游协议 | OpenAI Chat Completions | OpenAI Chat Completions |
| Tools | enabled | enabled |
| Streaming capability | enabled | enabled |
| Vision | disabled | disabled |
| Structured output capability | disabled | enabled |
| Concurrency group | `zai-glm` | `gpt-5.6-sol` |
| Profile session ceiling | 基础 profile 为 32 | 基础 profile 为 25；Claude campaign 临时 profile 为 32 |
| Gateway interfaces | OpenAI Chat Completions、Anthropic Messages | OpenAI Chat Completions、Anthropic Messages |
| Middleware | `glm_tool_call_compat` | 无 |

模型 endpoint 的实际 URL 和 API key 通过环境变量注入，没有写入数据库或本文。

这里的 `Profile session ceiling` 是 campaign/gateway 层的模型会话池上限，不等同于 runner 同时调度的 task 数。直接八-harness 运行没有经过 Claude campaign 的共享模型池，因此第 7 节中 GPT cell 的 task 并发可以是 32，即使其基础 profile ceiling 为 25；多出的 task 会在请求模型时排队。`Streaming capability=enabled` 也只表示 profile 宣告支持 streaming，不代表本文已证明每个 SDK 的实际请求都启用了流式响应。

### 5.2 GLM tool-call compatibility

GLM 运行经过模型专属 compatibility middleware。它只做两类窄范围修复：

1. 历史 assistant message 同时包含 `tool_calls` 且 `content=null` 时，发送上游前改为 `content=""`。
2. tool arguments 呈现为一个或多个空对象后拼接一个非空对象（例如 `{}{...}`）时，只保留最后那个明确的非空 JSON object。

合法参数保持不变；含糊的多非空对象拼接或任意损坏 JSON 不做猜测式修复。该 middleware 是 GLM 条件的一部分，解释结果时应视为模型接入协议设置，而不是所有模型共享的处理。

### 5.3 Claude SDK 的协议转换

其余八个 harness 通过 OpenAI-compatible Chat Completions 使用 native function calling。Claude SDK harness 使用 Anthropic Messages API；A2E gateway 再把 Anthropic Messages 请求转换成两个模型各自的 OpenAI Chat Completions 上游请求。GLM 条件仍经过 `glm_tool_call_compat`。

因此 Claude SDK cell 同时改变了 Agent loop 和模型访问协议适配器。它适合被解释为整体 harness 结果，不是只控制消息协议的 ablation。

## 6. 九个 Harness 的具体配置

### 6.1 框架版本

| Harness | 运行时版本 |
|---|---|
| Agno | `agno 2.6.7` |
| AutoGen AgentChat | `autogen-agentchat 0.7.5`、`autogen-core 0.7.5`、`autogen-ext 0.7.5` |
| Claude SDK | `anthropic 0.101.0` |
| CrewAI | `crewai 1.15.16` |
| Google ADK | `google-adk 1.14.1` |
| LangGraph | `langgraph 1.1.10`、`langchain-openai 1.2.1` |
| LlamaIndex | `llama-index-core 0.14.22`、`llama-index-llms-openai-like 0.7.2` |
| OpenAI Agents | `openai-agents 0.17.2` |
| SmolAgents | `smolagents 1.24.0` |

主任务环境还包含 `openai 2.36.0`、`opentelemetry-api 1.42.1`、`a2e-client 2.3.1`；AutoGen 因 protobuf 依赖冲突使用隔离环境，其中 `openai 2.37.0`。执行 Python 为 3.12.3。

### 6.2 Harness loop 与 decoding 差异

| Harness | 模型 client / loop | 关键设置或差异 |
|---|---|---|
| Agno | `OpenAILike` + `Agent.arun` | `max_tokens=4096`、request timeout 180 s、client retries 2、`tool_call_limit=10000`；Agent 内部 deadline 比官方逐题 deadline 提前 0.5 s，以便保存 partial trace；附加原生 function-call 提示 |
| AutoGen AgentChat | `OpenAIChatCompletionClient` + `AssistantAgent` | timeout 180 s、`max_tool_iterations=10000`；尝试传入 `max_tokens=4096`，若 SDK 签名不支持则回退；使用隔离 uv 环境 |
| Claude SDK | `AsyncAnthropic.messages.create` 手写 tool loop | timeout 180 s、`max_tokens=4096`、最多 10,000 turns；原生 Anthropic `tool_use/tool_result`；经 A2E Anthropic→OpenAI gateway |
| CrewAI | `crewai.LLM` / LiteLLM + `Agent`/`Crew` | `openai/<model>`、`max_tokens=4096`、`max_iter=10000`、`verbose=False`；role/goal/backstory 包装；附加“必须先调用工具”的 task hint；未显式覆盖 request timeout |
| Google ADK | `LiteLlm` + `InMemoryRunner` | `openai/<model>`、`max_tokens=4096`；禁用 LiteLLM aiohttp transport；丢弃不兼容的 `prompt_cache_retention` 参数；附加原生工具提示；未显式覆盖 request timeout |
| LangGraph | `ChatOpenAI` + 自定义状态图 | `max_tokens=4096`、timeout 180 s、最多 10,000 turns；保留 text 和 native tool call 两条路径 |
| LlamaIndex | `OpenAILike` + `FunctionAgent` | `is_chat_model=True`、`is_function_calling_model=True`、`max_tokens=4096`、`max_iterations=10000`；唯一显式设置 `temperature=1.0` 的 harness；未显式覆盖 request timeout |
| OpenAI Agents | `AsyncOpenAI` + `OpenAIChatCompletionsModel` + `Runner` | `ModelSettings(max_tokens=4096)`、`max_turns=10000`；未显式覆盖 request timeout |
| SmolAgents | `OpenAIServerModel` + `ToolCallingAgent` | `max_tokens=4096`、`max_steps=10000`；框架默认 `tool_choice="required"`；保留框架自己的 ReAct/system template，只把 TB2.1 policy 作为 `instructions` 注入；未显式覆盖 request timeout |

除 LlamaIndex 外，其余 harness 没有在 A2E wrapper 中显式设置 temperature，因而沿用各 SDK/provider 默认值。由此不能声称九个 harness 的最终 prompt 字节级一致，或 decoding 默认值完全一致；实验控制的是任务、工具能力、主要 token/wall-clock 上限和 verifier。

## 7. 并发与执行拓扑

### 7.1 八个 OpenAI-compatible harness

八个非 Claude harness 逐 cell 顺序启动；同一 cell 内并发执行不同任务。每个 cell 使用独立 A2E server 生命周期和独立 SQLite 数据库。

| Harness | GPT-5.6-sol 基础运行并发 | GLM-5.3 基础运行并发 |
|---|---:|---:|
| Agno | 32 | 32 |
| AutoGen AgentChat | 32 | 25 |
| CrewAI | 32 | 25 |
| Google ADK | 25 | 25 |
| LangGraph | 32 | 25 |
| LlamaIndex | 32 | 25 |
| OpenAI Agents | 32 | 25 |
| SmolAgents | 32 | 25 |

并发用于提高吞吐，不是实验因子。数据库中的 `run_duration_seconds` 会混合模型请求、工具执行、容器工作、排队和 verifier 时间，不能当作纯模型 inference latency。

### 7.2 Claude SDK campaign

Claude SDK 的两个模型在同一个 162-Trial campaign 中运行：

| 资源 | 配置上限 | 实测 high-water mark |
|---|---:|---:|
| Trial processes（global） | 64 | 64 |
| Active model cells | 2 | 2 个 cell 同时可调度 |
| Sandboxes | 32 | 32 |
| Aggregate model sessions | 32 | 32 |
| GLM pool | 32 | 16 |
| GPT pool | 32 | 17 |
| Graders | 32 | 16 |
| Uploads | 16 | 1 |
| Queue capacity | 64 | — |

该 campaign 的自动 retry 为 0、`timeout_seconds=null`、取消 grace 为 30 s，artifact retention 策略为 `failures`。

## 8. Verifier 流程与评分口径

Agent 停止后，仍在同一个 live container 中执行以下步骤：

1. 从宿主机注入受信任且固定版本的 `uv` / `uvx 0.9.5`。
2. 把官方 held-out `tests/` 复制到 container 的 `/tests`；Agent 运行时不可见这些文件。
3. 将官方 `test.sh` 中的在线 uv bootstrap 替换成本地可用性检查。
4. 创建并清空 `/logs/verifier/reward.txt` 与 `/logs/verifier/ctrf.json` 的旧状态。
5. 在题目 WORKDIR 中执行 `bash /tests/test.sh`，使用该题的 verifier timeout。
6. verifier 依赖使用预热的 Docker named volumes，并设置 `UV_OFFLINE=1`。
7. 读取官方 `reward.txt`，同时解析 CTRF test report。

一个运行只有同时满足以下条件才记为 resolved（1 分）：

- `reward.txt` 的值严格为 `1`；
- CTRF 存在且可解析；
- CTRF 收集到的测试数大于 0；
- CTRF 中失败测试数为 0。

否则，在 verifier 有效完成的前提下记为 0。每个 cell 的主分数为：

```text
cell_score = sum(tb_resolved) / 81
```

分母固定为 81。Agent 的 `status`、error/timeout、turn 数、tool-call 数和耗时只是诊断字段，不会覆盖官方 verifier 的二元结果。例如 Agent 虽然到时退出，但它此前写入的产物仍可能通过 verifier；反之 Agent 返回 `ok` 也不代表通过测试。

## 9. 基础设施无效结果的补跑与合并

普通解题失败、模型错误、合法 Agent timeout 和有效 verifier failure 都保留为 0，不做“直到成功”为止的重试。只有缺失/损坏 CTRF、verifier artifact 缺失或明确的基础设施失败会定向补跑。

替换键固定为：

```text
(model, harness, task_id, repetition)
```

最终数据库对每个键只保留一条记录。

| 运行组 | 定向替换数 | 分布 |
|---|---:|---|
| GLM 八-harness | 14 | Agno 1、AutoGen 2、CrewAI 2、Google ADK 2、LangGraph 1、LlamaIndex 2、OpenAI Agents 2、SmolAgents 2 |
| GPT 八-harness | 68 | Agno 9、AutoGen 1、CrewAI 0、Google ADK 2、LangGraph 13、LlamaIndex 27、OpenAI Agents 15、SmolAgents 1 |
| Claude SDK | 24 | GLM 10、GPT 14 |

GLM 八-harness 的缺失 CTRF 补跑按 harness 顺序执行，单 harness 内任务并发为 4。Claude SDK 补跑总上限为 16；实际 GLM 10 题和 GPT 14 题分别以不超过各自任务数的并发运行。Claude 的 24 条 replacement 中有两条最终成为合法 Agent timeout（`gcode-to-text` 900 s、`make-mips-interpreter` 1,800 s），它们有完整 verifier 结果并保留 0 分。

因此最终的 1,458 条结果是“基础运行 + 仅针对基础设施无效项的确定性 replacement merge”，而不是单次不间断 campaign 的原始输出。

## 10. Trace、token 与持久化

- 九个 harness 都通过 OpenInference instrumentation 产生 OpenTelemetry spans。
- A2E server 将 experiments、runs、annotations、traces 和 spans 写入 SQLite。
- task-level prompt/completion token 应从 final run trace 中 `span_kind=LLM` 的直接 usage counter 求和；不要同时加父 span 的累计值，否则会重复计算。
- `experiment_runs.prompt_token_count` 和 `completion_token_count` 在这批数据库中普遍为空，不应把空值当成 0。
- CrewAI 与 Claude SDK 的 LLM-span token telemetry 不完整；部分 GLM harness 也存在缺失或 trace/run 脱离。因此 token/cost 分析必须报告 available-case 分母，不能用 81 作为所有 cell 的默认 token 分母。
- 这些 telemetry 缺口不影响 held-out-test 的 `tb_resolved`。

## 11. 复现边界与应归档内容

已知 campaign lock 显示 Claude SDK 基于 Git commit `8489fb0904cf7c4142daf30018d874d97dfe3f10` 的 dirty worktree；部分后续 CrewAI/SmolAgents campaign lock 基于 `79a6238c0a1e8916d2ae9eeeb74c3c220858f0d2` 的 dirty worktree。八个直接运行数据库记录了模型、harness、seed 和任务 ID，但没有完整的源码 patch digest。因此不应把单个 Git commit 描述为全部 18 个 cell 的精确代码快照。

为完整复现，建议同时归档：

- 整个 A2E source tree 的 tarball 或 clean commit + dirty patch；
- 两个 model profile 及其 digest；
- `task/uv.lock`、AutoGen 隔离环境 lock 和 Python 版本；
- 81 个 task ID 及每题 `task.toml`；
- Docker image 的不可变 digest，而不仅是 tag；
- campaign `config.json`、`lock.json`、Trial attempt artifacts；
- GLM middleware 版本与配置；
- 基础数据库、补跑数据库和 merge report 的 SHA-256；
- 模型服务端的精确模型 revision、采样参数与 provider 侧默认值。

当前记录没有完整保存宿主机 CPU/RAM 规格、全部 Docker image digest、模型服务端 revision，以及所有 SDK/provider 默认 decoding 参数。论文中应把这些列为复现限制，而不是从当前机器状态反推。

## 12. 配置来源索引

- TB2.1 loader、资源与逐题 timeout：[`task/datasets/terminal_bench_2_1/.../loader.py`](../task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/loader.py)
- 统一 prompt 与工具执行：[`binding.py`](../task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/binding.py)
- 官方 verifier 与 resolved 规则：[`grader.py`](../task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/grader.py)
- TB2.1 的 10,000-turn override：[`task/runners/.../registry.py`](../task/runners/src/ageneval/task/runners/registry.py)
- 直接 runner 的 outer timeout 与 sandbox retry=0：[`task/examples/run_experiment.py`](../task/examples/run_experiment.py)
- 模型 profiles：[`glm-5.3.yaml`](../task/models/glm-5.3.yaml)、[`gpt-5.6-sol.yaml`](../task/models/gpt-5.6-sol.yaml)
- GLM middleware 说明：[`glm-openai-compat-proxy.md`](glm-openai-compat-proxy.md)
- Claude campaign 的最终并发记录：[`campaign-plan.json`](../.a2e-tb21-claude-sdk/full-c64-s32-m32-20260822-200553/campaign-plan.json)、[`concurrency-report.json`](../.a2e-tb21-claude-sdk/full-c64-s32-m32-20260822-200553/concurrency-report.json)
- Claude replacement merge：[`a2e.merge-report.json`](../.a2e-tb21-claude-sdk/merged-claude-sdk-final-20260824/a2e.merge-report.json)

## 附录 A：81 题逐题 Agent / Verifier / 资源设置

下表直接由每题 `task.toml` 与 Dockerfile 最后一个 `WORKDIR` 生成。Storage 均为 10,240 MB、GPU 均为 0、Internet 均允许，故不在逐题表中重复。

| Task | 类别 | 难度 | Agent timeout | Verifier timeout | CPU | 内存 (MB) | WORKDIR |
|---|---|---|---:|---:|---:|---:|---|
| `adaptive-rejection-sampler` | scientific-computing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `bn-fit-modify` | scientific-computing | hard | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `build-cython-ext` | debugging | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `build-pmars` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `build-pov-ray` | software-engineering | medium | 12,000 s | 12,000 s | 1 | 2,048 | `/app` |
| `caffe-cifar-10` | machine-learning | medium | 3,600 s | 1,200 s | 4 | 8,192 | `/app` |
| `cancel-async-tasks` | software-engineering | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `chess-best-move` | games | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `circuit-fibsqrt` | software-engineering | hard | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `cobol-modernization` | software-engineering | easy | 900 s | 900 s | 1 | 2,048 | `/app` |
| `code-from-image` | software-engineering | medium | 1,200 s | 1,200 s | 1 | 2,048 | `/app` |
| `compile-compcert` | system-administration | medium | 2,400 s | 2,400 s | 2 | 4,096 | `/app` |
| `configure-git-webserver` | system-administration | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `constraints-scheduling` | personal-assistant | medium | 1,200 s | 1,200 s | 1 | 2,048 | `/app` |
| `count-dataset-tokens` | model-training | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `custom-memory-heap-crash` | debugging | medium | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `db-wal-recovery` | file-operations | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `distribution-search` | machine-learning | medium | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `dna-assembly` | scientific-computing | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `dna-insert` | scientific-computing | medium | 1,800 s | 1,800 s | 1 | 4,096 | `/app` |
| `extract-elf` | file-operations | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `extract-moves-from-video` | file-operations | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `feal-differential-cryptanalysis` | mathematics | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `feal-linear-cryptanalysis` | mathematics | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `financial-document-processor` | data-processing | medium | 1,200 s | 1,200 s | 1 | 4,096 | `/app` |
| `fix-git` | software-engineering | easy | 900 s | 900 s | 1 | 2,048 | `/app/personal-site` |
| `fix-ocaml-gc` | software-engineering | hard | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `gcode-to-text` | file-operations | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `git-leak-recovery` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `git-multibranch` | system-administration | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `gpt2-codegolf` | software-engineering | hard | 900 s | 900 s | 1 | 8,192 | `/app` |
| `headless-terminal` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `hf-model-inference` | data-science | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `install-windows-3.11` | system-administration | hard | 3,600 s | 3,600 s | 2 | 4,096 | `/app` |
| `kv-store-grpc` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `large-scale-text-editing` | file-operations | medium | 1,200 s | 1,200 s | 1 | 2,048 | `/app` |
| `largest-eigenval` | mathematics | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `llm-inference-batching-scheduler` | machine-learning | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `log-summary-date-ranges` | data-processing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `mailman` | system-administration | medium | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `make-doom-for-mips` | software-engineering | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `make-mips-interpreter` | software-engineering | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `mcmc-sampling-stan` | data-science | hard | 1,800 s | 1,800 s | 4 | 8,192 | `/app` |
| `merge-diff-arc-agi-task` | debugging | medium | 900 s | 900 s | 1 | 4,096 | `/app` |
| `model-extraction-relu-logits` | mathematics | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `modernize-scientific-stack` | scientific-computing | medium | 600 s | 600 s | 1 | 2,048 | `/app` |
| `mteb-leaderboard` | data-science | medium | 3,600 s | 3,600 s | 1 | 8,192 | `/app` |
| `mteb-retrieve` | data-science | medium | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `multi-source-data-merger` | data-processing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `nginx-request-logging` | system-administration | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `overfull-hbox` | debugging | easy | 750 s | 360 s | 2 | 4,096 | `/app` |
| `path-tracing` | software-engineering | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `path-tracing-reverse` | software-engineering | hard | 1,800 s | 1,800 s | 1 | 2,048 | `/app` |
| `polyglot-c-py` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `polyglot-rust-c` | software-engineering | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `portfolio-optimization` | optimization | medium | 3,600 s | 3,600 s | 1 | 4,096 | `/app` |
| `protein-assembly` | scientific-computing | hard | 1,800 s | 1,800 s | 1 | 4,096 | `/app` |
| `prove-plus-comm` | software-engineering | easy | 900 s | 900 s | 1 | 2,048 | `/workspace` |
| `pypi-server` | software-engineering | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `pytorch-model-cli` | model-training | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `pytorch-model-recovery` | model-training | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `qemu-alpine-ssh` | system-administration | medium | 900 s | 900 s | 1 | 4,096 | `/app` |
| `qemu-startup` | system-administration | medium | 900 s | 900 s | 1 | 4,096 | `/app` |
| `query-optimize` | data-science | medium | 900 s | 1,800 s | 1 | 2,048 | `/app` |
| `raman-fitting` | scientific-computing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `regex-chess` | software-engineering | hard | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `regex-log` | data-processing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `reshard-c4-data` | data-science | medium | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `rstan-to-pystan` | data-science | medium | 1,800 s | 1,800 s | 4 | 8,192 | `/app` |
| `sam-cell-seg` | data-science | hard | 7,200 s | 7,200 s | 1 | 4,096 | `/app` |
| `schemelike-metacircular-eval` | software-engineering | medium | 2,400 s | 2,400 s | 1 | 2,048 | `/app` |
| `sparql-university` | data-querying | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
| `sqlite-db-truncate` | debugging | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `sqlite-with-gcov` | system-administration | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `torch-pipeline-parallelism` | software-engineering | hard | 900 s | 900 s | 1 | 8,192 | `/app` |
| `torch-tensor-parallelism` | software-engineering | hard | 900 s | 900 s | 1 | 8,192 | `/app` |
| `train-fasttext` | model-training | hard | 3,600 s | 3,600 s | 1 | 4,096 | `/app` |
| `tune-mjcf` | scientific-computing | medium | 900 s | 900 s | 1 | 2,048 | `/app` |
| `video-processing` | video-processing | hard | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `winning-avg-corewars` | software-engineering | medium | 3,600 s | 3,600 s | 1 | 2,048 | `/app` |
| `write-compressor` | software-engineering | hard | 900 s | 900 s | 1 | 2,048 | `/app` |
