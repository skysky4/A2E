# ageneval-task-tau-bench

τ-bench dataset adapter for A2E.

Provides:
- `load_tau_bench_tasks(domain, n)` — return ``TaskInput`` records.
- `get_tool_schemas(domain)` — OpenAI-compatible function specs for the
  ``retail`` / ``airline`` domain (used by both single- and multi-agent agents).

## Sources

Official outcome metric is ``tau_grader`` (alias ``tau_reward``): Sierra
``calculate_reward`` (pass^1) — final DB hash vs gold-action replay, plus
required NL outputs. Implementation:
``src/ageneval/task/datasets/tau_bench/grader.py``.

If the optional dependency ``tau-bench`` (upstream sierra-research package)
is installed, this adapter uses its scenarios. Otherwise it falls back to a
**small vendored sample set** so smoke tests work offline.

To enable upstream tasks:

```bash
uv pip install --extra upstream ageneval-task-tau-bench
```
