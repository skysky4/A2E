# GLM OpenAI compatibility proxy

`ageneval-model-gateway` owns the GLM compatibility middleware.
`scripts/glm_openai_compat_proxy.py` remains a thin, stdlib-only CLI wrapper for the
`glm-5.3` Chat Completions compatibility issues observed in A2E tool-calling
runs. It does not store or log request bodies, prompts, or credentials.

The proxy applies two model-scoped transformations:

1. On requests, an assistant message with `tool_calls` and `content: null` is
   sent upstream with `content: ""`.
2. On JSON and SSE responses, a tool argument string shaped like
   `{}{...non-empty object...}` is reduced to the single non-empty object.

Valid arguments are unchanged. Ambiguous concatenated objects and arbitrary
malformed JSON are counted but never guessed at.

## Start

Preserve the real API base URL separately, then point agents at the local
proxy:

```bash
export GLM_COMPAT_UPSTREAM_BASE_URL="$OPENAI_API_BASE"
python scripts/glm_openai_compat_proxy.py --port 8011

# In another shell:
export OPENAI_API_BASE=http://127.0.0.1:8011/v1
```

The client continues to send its normal `Authorization` header. The proxy
forwards it without logging it.

Campaign runs normally do not start this script manually. A model profile with
`middleware: [glm_tool_call_compat]` starts one loopback instance for the
Campaign, health-checks it, and shuts it down with the Controller.

Health and counters are available locally:

```bash
curl http://127.0.0.1:8011/healthz
curl http://127.0.0.1:8011/metrics
```

Use `--model` repeatedly to opt additional model IDs into normalization. All
other models pass through without body changes.

## Test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  task/.venv/bin/python -m pytest -q tests/test_glm_openai_compat_proxy.py
task/.venv/bin/python -m ruff check \
  scripts/glm_openai_compat_proxy.py tests/test_glm_openai_compat_proxy.py
```

The end-to-end test opens temporary loopback sockets, so it may need local
network permission in a sandboxed development environment.

## Full Terminal-Bench 2.1 run

Run the eight OpenAI-compatible agents sequentially with GLM-5.3. Each agent
uses task concurrency 32; keeping the agents sequential caps total benchmark
concurrency at 32 rather than 256:

```bash
bash scripts/run_tb21_glm53_8_agents.sh
```

The default is the 81-task non-security split. Include all 89 tasks explicitly:

```bash
bash scripts/run_tb21_glm53_8_agents.sh --include-security
```

The script shares one compatibility proxy but starts and stops A2E separately
for each agent. Every agent therefore gets an independent `<agent>/a2e.db`,
server log, runner log, and proxy-metrics snapshot under
`.a2e-tb2.1-glm-5.3-result/`. Use `--dry-run` to inspect the commands. Set
`TB21_RUN_ROOT` to an existing run directory to resume; agents with a `DONE`
marker are skipped.

## Scope

This proxy implements OpenAI Chat Completions transport. It does not translate
Anthropic `/v1/messages`, so the Claude SDK needs a separate protocol adapter.
