# A2E Campaign runner

The Campaign runner executes a deterministic `model × benchmark × harness`
matrix while the A2E Server remains a result and trace sink. Scheduling,
retries, process isolation, Docker limits, grading, and recovery all live in
the `task/` workspace.

Each benchmark selects its dataset-owned primary grader automatically; campaign
YAML does not need to repeat that grader configuration.

## Configure models

Model profiles live in `task/models/`. They contain only environment-variable
names, never credentials. For the included GLM profile:

```bash
export GLM_API_BASE=https://your-openai-compatible-endpoint/v1
export GLM_API_KEY=...
```

Each gateway-managed model starts one dynamic loopback port for the Campaign
and shares it across its Cell workers. The profile distinguishes the real
upstream protocol from the interfaces exposed to Harnesses:

```yaml
upstream_protocol: openai_chat_completions
gateway:
  interfaces:
    - openai_chat_completions
    - anthropic_messages
```

OpenAI-style Harnesses use `<gateway>/v1/chat/completions`; Claude SDK uses
the same port at `<gateway>/v1/messages`. For the included models:

```bash
export GLM_API_BASE=https://your-glm-endpoint/v1
export GLM_API_KEY=...
export GPT56_API_BASE=https://your-gpt56-endpoint/v1
export OPENAI_API_KEY=...
```

The model ids remain `glm-5.3` and `gpt-5.6-sol` for every compatible Harness;
no Harness-specific model copy is needed. The facade translates Anthropic
messages, native tools, stop reasons, usage, errors, and streaming events, and
applies model-specific middleware such as GLM tool-call repair on the same
port. Profiles without a `gateway` block connect directly from the Harness.

## Run and recover

```bash
cd /path/to/A2E

scripts/run_campaign.sh \
  --config task/campaigns/example.yaml --dry-run
scripts/run_campaign.sh \
  --config task/campaigns/example.yaml

scripts/run_campaign.sh \
  --resume task/runs/<campaign-id>
scripts/run_campaign.sh \
  --resume task/runs/<campaign-id> --rerun-failed
scripts/run_campaign.sh \
  --regrade task/runs/<campaign-id> --grader exact_match
```

The wrapper loads the repository `.env`, starts an isolated A2E Server backed
by `.a2e-campaigns/<campaign-id>/a2e.db`, waits for readiness, runs the
Campaign, and stops the Server on exit. Pass `--database`, `--http-port`, or
`--grpc-port` to override those defaults. A dry run does not start the Server.

The first command materializes the exact sample selection and matrix in
`runs/<campaign-id>/config.json` and `lock.json` without contacting the A2E
Server or model provider. A resume reloads the locked task IDs and refuses
modified Campaign or Model Profile content.

Each Trial stores atomic `config.json`, `lock.json`, current `result.json`, and
per-attempt results. Completed Server uploads are never rerun. A locally
completed Trial with a failed upload is uploaded again without invoking the
model. Corrupt result files are quarantined and the Trial is scheduled again.

Large Trial results do not travel inline over JSONL IPC. The child atomically
writes `attempts/<attempt>/process-result.json` and emits only a small
`result_ready` notification containing its byte size and SHA-256 digest. The
Controller verifies the digest and Trial identity before accepting it. This
keeps lifecycle IPC bounded even when an agent produces megabytes of tool
output or verifier data.

Terminal-Bench 2.1 preserves the exact verifier bytes before its container is
cleaned at `attempts/<attempt>/verifier/ctrf.json`. The result records its
relative path, size, and SHA-256; the complete parsed CTRF object is also part
of `ExperimentRun.output`, alongside the normalized terminal-bench evaluation.
The local raw file remains the byte-for-byte audit artifact.

## Concurrency

The Controller owns a bounded global Trial queue. `n_active_cells` selects the
Cells participating in each deterministic round-robin scheduling window; it
does not create a shared Cell execution process. Every Trial attempt runs in a
fresh OS process, so synchronous Docker commands and third-party SDK tools in
one Trial cannot block another Trial's event loop.

The Controller is the sole Permit Broker. Trial processes emit lifecycle requests over
IPC; they never mutate semaphores or the permit ledger directly. Controller
semaphores independently cap:

- complete Trials;
- model sessions by Model Profile concurrency group;
- live sandboxes;
- grader execution;
- Server uploads.

The global Trial permit spans `START` through the complete retry/backoff loop
and is released by `END` or `CANCEL`. Model, sandbox, and grader permits cover
only their lifecycle phases. Uploads use a separate Controller pool. Since
semaphore operations and lease accounting happen in one event loop, Worker
death cannot occur between acquiring a permit and recording it. `CANCEL` and
`END` are backstop releases for every lease owned by a Trial.

On Ctrl-C the Controller stops dispatching and terminates each Trial process
group after `cancellation_grace_seconds`. Docker containers carry Campaign,
Cell, Trial, and attempt labels, so recovery only removes resources owned by
that Campaign. Campaign `result.json` is updated while running and records both
permit high-water marks and real Trial-process/Docker-command activity.

The `autogen-agentchat` adapter keeps its standalone environment because of its
protobuf pin. Install it once with
`uv sync --project task/agents/autogen_agentchat --frozen`. The Controller
starts AutoGen Trial processes with that interpreter while retaining exact
global/model/sandbox/grader permits through duplex lifecycle IPC.

## Server mapping

One selected benchmark sample is uploaded once and shared by all its Cells.
Each Cell creates one Experiment. Trial results use the existing ExperimentRun
endpoint, and normalized grader results use the existing
ExperimentEvaluation upsert endpoint. No Server schema or scheduling API is
required.

## Legacy runner

`examples/run_experiment.py` also accepts `--model-profile`; its benchmark
grader is selected automatically. Explicit `--model`, `--api-base`, and
`--api-key` values override the selected profile. Its examples also run in
independent Trial processes, including non-sandbox datasets whose synchronous
tools could otherwise serialize the a2e-client event loop.
