# ageneval-task-terminal-bench-2-1

Terminal-Bench 2.1 dataset adapter for AEP sandboxed terminal-task evaluation.

**Source:** [`harbor-framework/terminal-bench-2-1`](https://github.com/harbor-framework/terminal-bench-2-1)
(Apache-2.0), pinned to commit
`5c8eadf1f393183288fa08b8f73ca9a469cc5e00`. The complete set of 89 task
definitions is vendored under
`src/ageneval/task/datasets/terminal_bench_2_1/vendor/tasks/`.
See `vendor/SOURCE.md`.

## Shape

This is a sandbox dataset (`kind="sandbox"`). Each task points to its published
Docker image:

1. `SandboxScoringRunner` pulls and starts the image.
2. The agent works through the sandbox-backed `bash` and
   `str_replace_editor` tools.
3. After the agent exits, `score_terminal_bench_2_1` injects trusted, pinned
   `uv`/`uvx` binaries into the live container. The agent never sees them.
4. The scorer copies the held-out `tests/` directory, replaces its online uv
   installer with a local availability check, and runs the official tests from
   pre-warmed dependency caches.
5. A run is resolved only when `reward.txt` and a non-empty CTRF report both
   confirm that every test passed. Missing CTRF is a `verifier_error`, not a
   graded task failure.
6. Before the container is cleaned, the exact `ctrf.json` bytes are atomically
   copied into the Trial attempt's `verifier/ctrf.json`. Its path, byte size,
   SHA-256, full parsed JSON, and summary counts are retained in the Trial
   output; Campaign uploads therefore keep the complete parsed report in the
   existing ExperimentRun JSON without requiring a Server schema change.

The 89 task definitions are stored locally. Docker images are pulled only when a
selected task is run.

The adapter enforces the per-task `task.toml` runtime settings: `[agent]`
`timeout_sec`, `[verifier]` `timeout_sec`, Docker CPU/memory/GPU and internet
limits, `[environment.env]` for the whole container, and `[verifier.env]` for
the held-out verifier process only. `storage_mb` remains recorded in metadata:
Docker root-filesystem quotas are storage-driver-specific and cannot be applied
portably without daemon configuration.

For the vendored 89-task release, `mcp_servers`, `environment.env`,
`verifier.env`, and `solution.env` are all empty; all tasks request zero GPUs
and allow internet access. Published Docker images are used directly, so
`build_timeout_sec` is not exercised. Agent implementations receive the
per-task `agent.timeout_sec`; A2E's global agent deadline is only a fallback for
datasets that do not define a task timeout.

## Run

Prepare the pinned uv 0.9.5 binaries and every verifier dependency set once:

```bash
python scripts/prewarm_tb21_verifier_cache.py
```

If the existing Docker cache volumes are already warm, only extract the trusted
binaries:

```bash
python scripts/prewarm_tb21_verifier_cache.py --binaries-only
```

The binaries are stored under the ignored host cache
`.a2e-cache/tb21-verifier/uv-0.9.5/`; they are copied into each task container
only after its agent run finishes.

```bash
cd /root/ageneval/AEP/task
AEP_TB21_TASK=fix-git \
uv run --frozen python examples/run_experiment.py \
  --dataset terminal-bench-2.1 \
  --agent agno \
  --model qwen-max \
  --n 1
```

The benchmark's `tb_resolved` grader is selected automatically.

Without `AEP_TB21_TASK`, the loader prefers an already cached task image and
then falls back to task-name order.

## Environment

| Variable | Purpose |
|---|---|
| `AEP_TB21_TASK` | Pin one Terminal-Bench 2.1 task ID |
| `A2E_TB21_TASK` | Backward-compatible alias for `AEP_TB21_TASK` |
| `A2E_TB2_SCORE_PROXY` | Proxy URL used by in-container verifier setup |
| `A2E_TB2_DOCKER_GW` | Docker bridge gateway used for proxy rewriting |
| `A2E_TB21_UV_BIN_DIR` | Override the trusted host directory containing uv/uvx 0.9.5 |
