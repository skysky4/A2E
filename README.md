<p align="center">
  <img src="docs/A2E_logo.jpg" alt="A2E" width="900"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.07346"><img src="https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://github.com/datamllab/A2E"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://colab.research.google.com/github/stevewithjobs/AEP/blob/yuchenyue/notebooks/a2e_quickstart.ipynb"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/></a>
  <a href="https://huggingface.co/papers/2608.07346"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
</p>

<p align="center">
  <a href="#-updates">🎉 Updates</a> •
  <a href="#1-quick-start">🚀 Quick start</a> •
  <a href="#2-build-experiments">🧪 Build experiments</a> •
  <a href="#3-capture-trajectories">📡 Capture trajectories</a> •
  <a href="#4-score-results">📊 Score results</a> •
  <a href="#5-view-results">👀 View results</a>
</p>

<p align="center">
  <em>Evaluate any agent on any dataset, with full trajectory visibility.</em>
</p>

<p align="center">
  <strong>English</strong> | <a href="README_zh.md">中文</a>
</p>

---

**A<sup>2</sup>E** (Agent Auditing Engine) is an open-source platform for auditing agent harnesses end to end. It helps you:

- **Build experiments** — pair any benchmark with any agent harness and run them through one CLI
- **Capture trajectories** — auto-instrument LLM and tool calls into standardized traces
- **Score results** — score both the process and the final outcome with multidimensional metrics
- **View results** — browse datasets, experiments, and trace trees in a local UI

The loop is simple: **build → capture → score → view**. A local server stores runs, traces, and scores for every step.

A<sup>2</sup>E is harness and vendor agnostic, with out-of-the-box support for popular agent SDKs ([OpenAI Agents SDK](#harness-openai-agents), [Claude Agent SDK](#harness-claude-sdk), [LangGraph](#harness-langgraph), [Google ADK](#harness-google-adk), [AutoGen AgentChat](#harness-autogen-agentchat), [CrewAI](#harness-crewai), [LlamaIndex](#harness-llama-index), [Agno](#harness-agno), [smolagents](#harness-smolagents)) reached through any OpenAI-compatible or Anthropic endpoint. Each harness is auto-instrumented through OpenInference, so your agent code stays untouched — see the [supported harness table](#3-capture-trajectories) for the full list and instrumentor paths.

## 🎉 Updates

- **2026-08-29** — Added DeepSearchQA (`deepsearchqa`), official cell runners (`scripts/run_n1.sh`, `scripts/run_full.sh`), and GDPval-AA (`gdpval-aa`).
- **2026-08-12** — 🤗 Paper page live on [Hugging Face](https://huggingface.co/papers/2608.07346).
- **2026-08-11** — 📓 [Colab quickstart notebook](https://colab.research.google.com/github/stevewithjobs/AEP/blob/yuchenyue/notebooks/a2e_quickstart.ipynb) published — run A<sup>2</sup>E end to end in the browser.
- **2026-08-10** — ✨✨ Full codebase of A<sup>2</sup>E released.
- **2026-08-07** — 📄 A<sup>2</sup>E preprint posted on [arXiv](https://arxiv.org/abs/2608.07346).

## 1. Quick start

**Dependency**

```bash
bash scripts/start.sh
```

**API key**

```bash
cp .env.example .env
# fill in the values below
```

Then open http://localhost:6006.

## 2. Build experiments

Pick a dataset and an agent harness, then run:

```bash
# Terminal 1 — keep the server up
bash scripts/start.sh                 # → http://localhost:6006

# Terminal 2 — demo cell (τ-bench × agno)
bash task/run_experiment.sh

# Official n=1 / full-split cells (evaluators, seed, and per-dataset budgets)
bash scripts/run_n1.sh <agent> <dataset>
bash scripts/run_full.sh <agent> <dataset>
# e.g. bash scripts/run_n1.sh agno tau-bench
#      bash scripts/run_n1.sh langgraph deepsearchqa
#      bash scripts/run_full.sh llama-index gdpval-aa
```

Or follow the interactive walkthrough: `bash example/run_examples.sh`.

### CLI

Run custom experiments from `task/`:

```bash
cd task
uv run --frozen python examples/run_experiment.py \
  --dataset <dataset> \
  --agent   <framework>      # default: agno
  --model   <model>          # default: A2E_MODEL from .env
  --evaluators <a,b,...>     # default: dataset preset
  --n       <count>          # default: 40
  --full                     # use the official split instead of --n
  --sample-seed 20260816 \
  --domain  retail           # tau-bench / tau2 / tau3
  --max-turns 30 \
  --max-tokens 4096 \
  --llm-timeout 180 \
  --run-deadline 1100
```

| Flag | Purpose |
|------|---------|
| `--list` | Print all datasets / agent harnesses / evaluators |
| `--dataset` | Dataset name (required) |
| `--agent` | Agent harness (default `agno`) |
| `--model` | Override `A2E_MODEL` for this run |
| `--evaluators` | Comma-separated graders (default: this dataset's official grader) |
| `--n` | Sample size (absolute count, random without replacement) |
| `--full` | Run the official split (ignore `--n`) |
| `--sample-seed` | Reproducible sample (official cells use `20260816`) |
| `--domain` | `retail` / `airline` (tau-bench / tau2 / tau3) |
| `--max-turns` | Override official `max_turns` (prints the resolved value) |
| `--max-tokens` | Override official `max_tokens` |
| `--llm-timeout` | Override official per-request LLM timeout (seconds) |
| `--run-deadline` | Override official whole-agent wall (seconds) |

```bash
cd task
uv run --frozen python examples/run_experiment.py --list
```

### Official cell budgets

Canonical per-benchmark numbers live on each `DATASETS` entry as `official_settings` in
`task/runners/src/ageneval/task/runners/registry.py`. The CLI prints the resolved
values at start-up. Precedence: `--max-turns` / `--max-tokens` / `--llm-timeout` /
`--run-deadline` → env (`A2E_MAX_TURNS`, `A2E_MAX_TOKENS`, `A2E_LLM_TIMEOUT`,
`A2E_RUN_DEADLINE`) → `official_settings`. `scripts/run_n1.sh` and
`scripts/run_full.sh` set the same official caps via env (you can still pass CLI flags after the dataset name):

| Dataset | `max_turns` | `max_tokens` | LLM timeout | wall / deadline |
|---------|-------------|--------------|-------------|-----------------|
| `tau-bench` / `tau2` / `tau3` | 30 | 4096 | 180s | 1200s / 1100s |
| `deepsearchqa` | 8 | 4096 | 180s | 720s / 620s |
| `gdpval-aa` | 250 | 16384 | 600s | 7200s / 7000s |

Set `A2E_MODEL`, `OPENAI_API_KEY`, and `OPENAI_API_BASE` in `.env`. Optional GDPval locals: `A2E_GDPVAL_FILES_DIR`, `A2E_GDPVAL_PARQUET`.

### Benchmarks

Each benchmark includes a built-in evaluator preset. Use `--evaluators` to override
it; use `--list` to inspect every available benchmark, harness, and evaluator.

| Benchmark              | Kind    | Built-in default evaluators                                  | Sandbox |
| ---------------------- | ------- | ------------------------------------------------------------ | ------- |
| `tau-bench`          | Tool    | `tau_grader` (Sierra `calculate_reward` pass^1)          | /       |
| `tau2`               | Tool    | `tau_grader` (Sierra `calculate_reward` pass^1)          | /       |
| `tau3`               | Tool    | `tau_grader` (Sierra `calculate_reward` pass^1)          | /       |
| `traject-bench`      | Tool    | `tool_recall`, `llm_judge`                               | /       |
| `deepsearchqa`       | Tool    | `deepsearch_grader`                                      | HF [`google/deepsearchqa`](https://huggingface.co/datasets/google/deepsearchqa) |
| `mmlu`               | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `gsm8k`              | QA      | `numeric_match`, `llm_judge`                             | /       |
| `humaneval`          | QA      | `humaneval_pass`                                         | /       |
| `persistbench`       | QA      | `substring`, `llm_judge`                                 | /       |
| `gdpval-aa`          | Tool    | `gdp_grader` (files + web/code/finish; Elo off-run) | HF [`openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) |
| `gpqa`               | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `mmlu-pro`           | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `arc-challenge`      | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `truthfulqa`         | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `agieval`            | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `commonsenseqa`      | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `hellaswag`          | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `openbookqa`         | QA      | `mc_letter`, `llm_judge`                                 | /       |
| `bbh`                | QA      | `exact_match`, `llm_judge`                               | /       |
| `math`               | QA      | `numeric_match`, `llm_judge`                             | /       |
| `swe-bench-lite`     | Sandbox | `swe_resolved`, `swe_fail_to_pass`, `swe_pass_to_pass` | ✅      |
| `swe-bench-verified` | Sandbox | `swe_resolved`, `swe_fail_to_pass`, `swe_pass_to_pass` | ✅      |
| `swe-bench-pro`      | Sandbox | `swe_resolved`, `swe_fail_to_pass`, `swe_pass_to_pass` | ✅      |
| `terminal-bench-2`   | Sandbox | `tb_resolved`                                              | ✅      |
| `terminal-bench-2.1` | Sandbox | `tb_resolved`                                              | ✅      |

### Agent Harnesses

All supported agent harnesses are auto-instrumented. Select one with `--agent`:

| `--agent`           | Harness           |
| --------------------- | ----------------- |
| `agno`              | Agno              |
| `smolagents`        | smolagents        |
| `llama-index`       | LlamaIndex        |
| `langgraph`         | LangGraph         |
| `crewai`            | CrewAI            |
| `google-adk`        | Google ADK        |
| `autogen-agentchat` | AutoGen AgentChat |
| `claude-sdk`        | Claude Agent SDK  |
| `openai-agents`     | OpenAI Agents SDK |

Implementation and OpenInference instrumentor paths are listed under
[Capture trajectories](#3-capture-trajectories).

### Sandbox experiments

Sandbox benchmarks require Docker and pull images of 1–3 GB each. For a repeatable
SWE-bench run, pass a cached instance ID to the helper script:

```bash
bash task/run_sandbox_experiment.sh <cached_instance_id>
```

## 3. Capture trajectories

While an experiment runs, **Monitor** auto-instruments the agent: LLM calls, tool
calls, and related spans are written to the server as OpenTelemetry / OpenInference
traces. You do not need a separate capture step for supported harnesses.

Supported auto-instrumented agent harnesses:

| `--agent`                                                 | Harness package                   | OpenInference instrumentor                                                  |
| ----------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------- |
| <a id="harness-agno"></a>`agno`                           | `task/agents/agno`              | `monitor/instrumentation/openinference-instrumentation-agno`              |
| <a id="harness-smolagents"></a>`smolagents`               | `task/agents/smolagents`        | `monitor/instrumentation/openinference-instrumentation-smolagents`        |
| <a id="harness-llama-index"></a>`llama-index`             | `task/agents/llama_index`       | `monitor/instrumentation/openinference-instrumentation-llama-index`       |
| <a id="harness-langgraph"></a>`langgraph`                 | `task/agents/langgraph`         | `monitor/instrumentation/openinference-instrumentation-langchain`         |
| <a id="harness-crewai"></a>`crewai`                       | `task/agents/crewai`            | `monitor/instrumentation/openinference-instrumentation-crewai`            |
| <a id="harness-google-adk"></a>`google-adk`               | `task/agents/google_adk`        | `monitor/instrumentation/openinference-instrumentation-google-adk`        |
| <a id="harness-autogen-agentchat"></a>`autogen-agentchat` | `task/agents/autogen_agentchat` | `monitor/instrumentation/openinference-instrumentation-autogen-agentchat` |
| <a id="harness-claude-sdk"></a>`claude-sdk`               | `task/agents/claude_sdk`        | `monitor/instrumentation/openinference-instrumentation-anthropic`         |
| <a id="harness-openai-agents"></a>`openai-agents`         | `task/agents/openai_agents`     | `monitor/instrumentation/openinference-instrumentation-openai-agents`     |

Shared OpenInference plumbing also lives under `monitor/openinference-instrumentation`
and `monitor/openinference-semantic-conventions`. Wiring is selected via
`framework_for_agent()` in `task/runners/.../registry.py`.

Captured trajectories show up in the **Trace** view at http://localhost:6006: open a
sample and switch to Trace for its span tree of LLM and tool calls.

## 4. Score results

After trajectories land on the server, score the process and the outcome with the
unified evaluation pipeline in `eval/`. Lightweight inline scoring can still ride
along with `--evaluators` on the experiment CLI; use `eval/` for deeper metric
groups across planning, tool usage, memory, correctness, efficiency, and safety.

Full details: [`eval/README.md`](eval/README.md).

### Run all metrics

```bash
# Terminal 1 — keep the server up
bash scripts/start.sh                 # → http://localhost:6006

# Terminal 2
cd server
uv run python ../eval/scripts/run_eval.py \
  --base-url http://localhost:6006 \
  --experiment-id <experiment_id> \
  --part all
```

### Run one metric group

| Part           | What it scores                       |
| -------------- | ------------------------------------ |
| `plan`       | Planning quality and decision-making |
| `skill`      | Execution skill (e.g. conciseness)   |
| `memory`     | Memory / faithfulness                |
| `tool`       | Tool selection and execution         |
| `correct`    | Final task correctness               |
| `efficiency` | Tokens, cost, turns, latency         |
| `safety`     | Safety-related behaviors             |

```bash
cd server
uv run python ../eval/scripts/run_eval.py \
  --base-url http://localhost:6006 \
  --experiment-id <experiment_id> \
  --part plan
```

Supported parts:

```text
plan, skill, memory, tool, correct, efficiency, safety
```

## 5. View results

The React viewer is served by `a2e serve` at http://localhost:6006 (built by `scripts/start.sh`). Swipe between **Task / Trace / Eval** for the same sample; open Trace for the span tree.

### Development (HMR)

`scripts/start.sh` builds the production UI. For Vite HMR, start the API and the UI separately:

```bash
# Terminal 1 — API only (after deps are installed)
cd server && uv run a2e serve       # http://127.0.0.1:6006

# Terminal 2
cd ui && pnpm install && pnpm dev   # http://127.0.0.1:5173  (proxies /v1 → :6006)
```

Or use `a2e serve --dev` for Vite HMR via the server templates.

## Project layout

```
AEP/
├── scripts/             # Install + server start; official n=1 / full-split runners
├── task/                # Build experiments (datasets, agents, runners)
├── monitor/             # Capture trajectories (auto-instrumentation)
├── eval/                # Score results (process and outcomes)
├── server/              # Store runs, traces, and scores
├── ui/                  # View results
└── example/             # Interactive walkthrough
```

Those directories are not layers stacked on top of each other — they are four independent processes that meet at the server. The diagram below shows what each one sends across:

<p align="center">
  <img src="docs/pipeline.png" alt="A2E pipeline" width="720"/>
</p>

- **`task/` + `monitor/` — ①** `task/` pairs a benchmark with an agent harness and runs the pair in a sandbox. `monitor/` wraps that run, auto-instrumenting every LLM and tool call, and writes the run plus its traces to the server. The harness stays unmodified: nothing in your agent code has to know A<sup>2</sup>E exists.
- **`server/` — the hub** One local store holding three kinds of records: `data` (datasets and samples), `trace` (execution trees), and `eval` (scores). Every other component talks only to this store, never directly to each other, so a run can be scored or re-scored long after it finished.
- **`eval/` — ② ③** Pulls traces back out and scores them on two axes: *trace eval* judges the process (did the agent call the right tools, in a sensible order?) while *outcome eval* judges the final answer. Scores are written back to the server against the same trace IDs.
- **`ui/` — ④** Reads it all back and renders datasets, experiments, trace trees, and scores at http://localhost:6006.

Because everything routes through the server, each stage is independently runnable — you can capture traces today, add an evaluator next week, and score the old runs without re-running a single agent.

## Acknowledgements

A<sup>2</sup>E depends on and draws from the following open-source projects:

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Phoenix](https://github.com/Arize-ai/phoenix)

## License

A<sup>2</sup>E is licensed under the [MIT License](LICENSE).

## Citation

If you find A<sup>2</sup>E useful in your research, please cite:

```bibtex
@misc{wang2026a2eendtoendagent,
      title={$A^2E$ : An End-to-End Agent Auditing Engine}, 
      author={Haoning Wang and Mingxun Zhang and Chenyue Yu and Yingjun Shang and Xia Hu and Guanchu Wang and Na Zou},
      year={2026},
      eprint={2608.07346},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.07346}, 
}
```
