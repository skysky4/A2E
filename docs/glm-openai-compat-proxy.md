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

## Terminal-Bench 2.1 Campaigns

Use the unified Campaign wrapper for TB2.1 runs. Model gateway middleware now
applies GLM tool-call compatibility inside each Campaign, while the wrapper
owns the A2E Server and SQLite lifecycle:

```bash
scripts/run_campaign.sh \
  --config task/campaigns/tb21-crewai-smolagents-gpt56-glm53.yaml
```

Concurrency is controlled by the Campaign YAML rather than by a dedicated
model or harness shell script. Resume the immutable task selection with:

```bash
scripts/run_campaign.sh \
  --resume task/runs/campaign-c19e7e74248264ab
```

## Scope

This proxy implements OpenAI Chat Completions transport. It does not translate
Anthropic `/v1/messages`, so the Claude SDK needs a separate protocol adapter.
