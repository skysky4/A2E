"""Generic A2E experiment runner — pick any (dataset, agent, evaluators).

This is the "free-selection" CLI:

    uv run --frozen python examples/run_experiment.py --list
    uv run --frozen python examples/run_experiment.py \\
        --dataset mmlu --agent langgraph --evaluators mc_letter,llm_judge

A2E UI will show the result under Datasets + Experiments at
http://localhost:6006 .
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from ageneval.task.core import SandboxScoringRunner, setup_instrumentation
from ageneval.task.runners import (
    AGENTS,
    DATASETS,
    DEFAULT_SAMPLE_SIZE,
    EVALUATORS,
    build_experiment_metadata,
    build_run_identity,
    framework_for_agent,
    list_registries,
    make_llm_judge,
    sample_dataset,
)

logger = logging.getLogger(__name__)


def _build_examples(tasks):
    rows = []
    for t in tasks:
        meta = {"task_id": t.task_id, **dict(t.metadata)}
        # Sandbox datasets carry their per-task sandbox spec on TaskInput.sandbox;
        # round-trip it through metadata so the sandbox task_fn can rebuild it.
        if t.sandbox is not None:
            meta["sandbox"] = dict(t.sandbox)
        rows.append(
            {
                "input": {"instruction": t.instruction, "initial_state": dict(t.initial_state)},
                "output": {
                    "expected_outputs": list(t.expected_outputs),
                    "expected_actions": list(t.expected_actions),
                },
                "metadata": meta,
            }
        )
    return rows


def _make_task_fn(agent, ds_entry: dict | None = None):
    from ageneval.task.core import TaskInput

    ds_entry = ds_entry or {}
    is_sandbox = ds_entry.get("kind") == "sandbox"
    # For sandbox datasets the agent runs inside a per-task container managed by
    # SandboxScoringRunner, which also grades the result while the container is
    # alive (A2E evaluators run after task_fn returns — too late).
    runner = agent
    if is_sandbox:
        runner = SandboxScoringRunner(
            inner=agent,
            score_fn=ds_entry["score"],
            setup_fn=ds_entry.get("setup"),
        )

    async def task_fn(input: dict, metadata: dict) -> dict:
        task_input = TaskInput(
            task_id=metadata.get("task_id", "?"),
            instruction=input.get("instruction", ""),
            initial_state=input.get("initial_state", {}),
            metadata=metadata if is_sandbox else {},
            sandbox=metadata.get("sandbox") if is_sandbox else None,
        )
        trace = await runner.run(task_input)
        out = {
            "final_answer": trace.final_answer or "",
            "tool_calls": [tc.name for tc in trace.tool_calls],
            "status": trace.status,
            "turns": trace.turns,
            "trace_id": trace.trace_id,
            "error": trace.error,
        }
        if is_sandbox:
            raw = dict(trace.raw)
            out["resolved"] = bool(raw.get("resolved"))
            out["swe_status"] = raw.get("status")
            out["swe_f2p_passed"] = raw.get("f2p_passed")
            out["swe_f2p_total"] = raw.get("f2p_total")
            out["swe_p2p_passed"] = raw.get("p2p_passed")
            out["swe_p2p_total"] = raw.get("p2p_total")
            out["model_patch"] = (raw.get("model_patch") or "")[:4000]
            # Terminal-Bench graders expose reward, verifier status, and test
            # counts. Preserve them in experiment-run output so the final DB
            # can be audited without relying on process logs.
            out["tb_status"] = raw.get("status")
            for key in (
                "tb_reward",
                "tb_tests_total",
                "tb_tests_passed",
                "tb_tests_failed",
                "tb_verifier_files",
                "tb_verifier_exit",
                "tb_verifier_stdout_tail",
                "tb_verifier_stderr_tail",
                "tb_verifier_phase",
                "tb_uv_injected",
                "tb_uv_version",
                "tb_bootstrap_rewritten",
                "tb_reward_read_error",
                "tb_ctrf_error",
                "tb_ctrf",
                "tb_ctrf_artifact",
                "tb_ctrf_artifact_error",
                "score_error",
            ):
                out[key] = raw.get(key)
        return out

    return task_fn


def _make_process_task_fn(
    *,
    dataset_name: str,
    harness: str,
    bind_kwargs: dict[str, Any],
    resolved_model: Any,
    explicit_agent_kwargs: dict[str, str],
    project_name: str,
    endpoint: str | None,
    timeout_seconds: int,
    run_id: str,
    run_root: Path,
):
    """Build an a2e-client task callback backed by one process per example."""
    from ageneval.task.orchestrator.process import TrialProcessRunner
    from ageneval.task.orchestrator.schema import LifecycleEvent

    script = Path(__file__).resolve().with_name("run_isolated_trial.py")

    async def task_fn(input: dict, metadata: dict) -> dict:
        from opentelemetry.propagate import inject

        task_id = str(metadata.get("task_id") or "unknown")
        trial_id = f"legacy-{task_id}-{uuid.uuid4().hex[:12]}"
        trace_context: dict[str, str] = {}
        inject(trace_context)

        async def lifecycle(_event: LifecycleEvent, _attempt: int | None) -> None:
            return None

        runner = TrialProcessRunner(
            script=script,
            cancellation_grace_seconds=30,
            lifecycle=lifecycle,
        )
        result = await runner.run(
            payload={
                "cell": {
                    "campaign_id": run_id,
                    "cell_id": f"legacy-{run_id}",
                    "benchmark": dataset_name,
                    "harness": harness,
                    "retry": {"max_retries": 0},
                },
                "trial": {
                    "trial_id": trial_id,
                    "task_id": task_id,
                    "repetition": 1,
                },
                "task": {
                    "task_id": task_id,
                    "instruction": input.get("instruction", ""),
                    "initial_state": input.get("initial_state", {}),
                    "expected_actions": [],
                    "expected_outputs": [],
                    "metadata": metadata,
                    "sandbox": metadata.get("sandbox"),
                },
                "benchmark": {
                    "id": dataset_name,
                    "domain": bind_kwargs.get("domain"),
                    "graders": [],
                },
                "profile": resolved_model.profile.public_dict(),
                "base_url": resolved_model.base_url,
                "project_name": project_name,
                "otel_endpoint": endpoint,
                "attempt": 1,
                "timeout_seconds": timeout_seconds,
                "agent_kwargs": explicit_agent_kwargs,
                "trace_context": trace_context,
            },
            resolved_model=resolved_model,
            python=sys.executable,
            pythonpath="",
            stderr_path=(
                run_root / "trials" / trial_id / "attempts" / "1" / "stderr.log"
            ),
            result_path=(
                run_root
                / "trials"
                / trial_id
                / "attempts"
                / "1"
                / "process-result.json"
            ),
            timeout_seconds=timeout_seconds + 30,
            extra_env={
                "A2E_TRIAL_LEGACY_API_KEY": (
                    resolved_model.api_key.get_secret_value()
                    if "api_key" in explicit_agent_kwargs
                    else ""
                ),
                "A2E_TRIAL_LEGACY_API_BASE": (
                    resolved_model.base_url or ""
                    if "api_base" in explicit_agent_kwargs
                    else ""
                ),
            },
        )
        output = dict(result.output)
        output.setdefault("trace_id", result.trace_id)
        output.setdefault("error", result.error)
        return output

    return task_fn


def _build_evaluator_list(names: list[str], judge_llm: Any | None):
    """Return a list of evaluator callables for a2e run_experiment.

    A2E introspects each callable's signature to know which kwargs
    (``output``, ``expected``, ``input``, ``metadata``, ``example``) to bind,
    so we MUST pass functions that already declare those parameters — no
    ``**kw`` wrappers.
    """
    evaluators: list = []
    for name in names:
        n = name.strip()
        if not n:
            continue
        if n == "llm_judge":
            if judge_llm is None:
                logger.warning("llm_judge requested but no LLM configured; skipped")
                continue
            evaluators.append(make_llm_judge(judge_llm))
            continue
        fn = EVALUATORS.get(n)
        if fn is None:
            raise ValueError(f"Unknown evaluator: {n}. Available: {sorted(EVALUATORS)}")
        evaluators.append(fn)
    return evaluators


def _build_experiment_metadata(*, agent_name: str, agent: Any, sdk: str) -> dict[str, Any]:
    return build_experiment_metadata(agent_name=agent_name, agent=agent, sdk=sdk)


def _sandbox_outer_timeout(tasks: list[Any], buffer_seconds: int = 300) -> int:
    """Cover one task's agent + verifier budgets without premature replay."""
    budgets = [
        float(task.metadata.get("agent_timeout_sec") or 1800)
        + float(task.metadata.get("verifier_timeout_sec") or 1800)
        for task in tasks
    ]
    return int(max(budgets, default=3600) + buffer_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available datasets/agents/evaluators and exit")
    parser.add_argument("--dataset", default=None, help=f"one of {sorted(DATASETS)}")
    parser.add_argument("--agent", default="agno", help=f"one of {sorted(AGENTS)}")
    parser.add_argument(
        "--evaluators",
        default="exact_match,substring",
        help="comma-separated names; pass empty string (\"\") for task-only (no scoring)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            f"number of cases to sample (default: {DEFAULT_SAMPLE_SIZE}); "
            "fails if the dataset has fewer candidates"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="seed for reproducible random sampling; omitted generates and records one",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        help=(
            "task category to exclude before sampling; repeat for multiple "
            "categories (currently supported by terminal-bench-2.1)"
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help=(
            "exact task ID to select; repeat for multiple tasks "
            "(currently supported by terminal-bench-2.1)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="maximum number of examples executed concurrently (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "outer per-example timeout in seconds; defaults to the largest "
            "task.toml agent+verifier budget plus 300 seconds for sandbox "
            "datasets, and 60 seconds otherwise"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional run id; omitted generates a unique timestamped id",
    )
    parser.add_argument("--domain", default=None, help="τ-bench-style 'retail' / 'airline' (for tau-bench / tau2 / tau3)")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-profile",
        default=None,
        help="model profile id from task/models or an explicit YAML path",
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None, help="A2E OTLP endpoint override")
    parser.add_argument("--project-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    if args.list:
        print(json.dumps(list_registries(), indent=2, ensure_ascii=False))
        return 0
    if not args.dataset:
        parser.error("--dataset is required (or pass --list)")
    if args.dataset == "gdpval":
        parser.error("dataset 'gdpval' was removed; use --dataset gdpval-aa (HF openai/gdpval)")
    if args.dataset not in DATASETS:
        parser.error(f"unknown dataset: {args.dataset}. Available: {sorted(DATASETS)}")
    if args.agent not in AGENTS:
        parser.error(f"unknown agent: {args.agent}. Available: {sorted(AGENTS)}")
    if args.n <= 0:
        parser.error("--n must be a positive integer")
    if args.concurrency <= 0:
        parser.error("--concurrency must be a positive integer")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be a positive integer")

    # Resolve the unified profile before constructing any framework SDK. Legacy
    # flags remain authoritative and are folded into the runtime-only model.
    model_runtime = None
    runtime_model = None
    if args.model_profile:
        from ageneval.model.gateway import ModelRuntime, load_model_profile, resolve_model

        profile_path = Path(args.model_profile)
        if not profile_path.is_absolute() and not profile_path.exists():
            profile_path = Path(__file__).resolve().parents[1] / "models" / f"{args.model_profile}.yaml"
        profile = load_model_profile(profile_path)
        profile_env = dict(os.environ)
        if args.api_base and profile.connection.base_url_env:
            profile_env[profile.connection.base_url_env] = args.api_base
        if args.api_key:
            profile_env[profile.connection.api_key_env] = args.api_key
        resolved = resolve_model(profile, profile_env)
        if args.model:
            resolved = resolved.model_copy(
                update={"profile": resolved.profile.model_copy(update={"model": args.model})}
            )
        model_runtime = ModelRuntime(resolved)
        runtime_model = model_runtime.start()
        atexit.register(model_runtime.close)
        args.model = args.model or runtime_model.profile.model
        args.api_base = runtime_model.base_url
        args.api_key = runtime_model.api_key.get_secret_value()

    # 1. Load the full candidate split, then select this run's exact sample.
    # Passing n=None avoids every loader's legacy first-N truncation. Explicit
    # sandbox pins (A2E_SWE_INSTANCE / A2E_SWE_PRO_INSTANCE / A2E_TB2_TASK /
    # AEP_TB21_TASK) still win inside the registry wrappers.
    ds_entry = DATASETS[args.dataset]
    load_kwargs: dict[str, Any] = {"n": None}
    bind_kwargs: dict[str, Any] = {}
    if args.dataset in ("tau-bench", "tau2", "tau3", "tau3bench", "tau3-bench"):
        # Live tools exist for retail/airline only. Default retail so a
        # telecom-heavy vendor dump is never paired with the retail wiki.
        domain = args.domain or "retail"
        load_kwargs["domain"] = domain
        bind_kwargs["domain"] = domain
    # Sandbox loaders reorder toward locally-cached docker images when ``n`` is
    # set. Passing n=None then randomly sampling (as we do for HF QA) would
    # pick an uncached image and docker pull through a dead proxy.
    if ds_entry.get("kind") == "sandbox":
        load_kwargs["n"] = args.n
    if args.exclude_category:
        if args.dataset != "terminal-bench-2.1":
            parser.error("--exclude-category is currently supported only by terminal-bench-2.1")
        load_kwargs["exclude_categories"] = args.exclude_category
    if args.task_id:
        if args.dataset != "terminal-bench-2.1":
            parser.error("--task-id is currently supported only by terminal-bench-2.1")
        load_kwargs["task_ids"] = args.task_id
    dataset = ds_entry["load"](**load_kwargs)
    dataset, selection = sample_dataset(
        dataset,
        n=args.n,
        seed=args.sample_seed,
    )
    binding = ds_entry["bind"](**bind_kwargs)

    # 2. Build the agent. Importing an SDK here is safe; instrumentation is
    # installed before the first framework call below.
    agent_entry = AGENTS[args.agent]
    if not agent_entry["supports_any_binding"] and args.dataset not in ("tau-bench",):
        print(
            f"⚠ agent '{args.agent}' is currently only τ-bench-compatible; "
            "results may be degraded.",
            file=sys.stderr,
        )
    agent_kwargs: dict[str, Any] = {"binding": binding}
    if args.model:
        agent_kwargs["model"] = args.model
    if args.api_base:
        agent_kwargs["api_base"] = args.api_base
    if args.api_key:
        agent_kwargs["api_key"] = args.api_key
    # Sandbox datasets (SWE-bench) need many turns; apply dataset-recommended
    # overrides (each builder ignores kwargs it doesn't accept).
    for _k, _v in (ds_entry.get("agent_overrides") or {}).items():
        agent_kwargs.setdefault(_k, _v)
    agent = agent_entry["build"](**agent_kwargs)

    # 3. Give this invocation its own dataset, experiment, and trace project.
    actual_model = (
        getattr(agent, "model", None)
        or getattr(agent, "_model_name", None)
        or args.model
        or "default-model"
    )
    if runtime_model is None:
        from ageneval.model.gateway import ModelProfile, ResolvedModel

        inherited_key = (
            args.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or "inherited-by-agent-builder"
        )
        runtime_model = ResolvedModel(
            profile=ModelProfile.model_validate(
                {
                    "id": f"legacy-{args.agent}",
                    "provider": "legacy",
                    "model": str(actual_model),
                    "upstream_protocol": "openai_chat_completions",
                    "connection": {"api_key_env": "A2E_TRIAL_MODEL_API_KEY"},
                    "concurrency": {
                        "group": f"legacy-{args.agent}",
                        "max_sessions": args.concurrency,
                    },
                }
            ),
            base_url=args.api_base,
            api_key=inherited_key,
        )
    identity = build_run_identity(
        dataset_name=dataset.name,
        dataset_key=args.dataset,
        agent_name=args.agent,
        model=str(actual_model),
        run_id=args.run_id,
    )
    framework = framework_for_agent(args.agent)
    project_name = args.project_name or identity.project_name
    provider = setup_instrumentation(
        project_name=project_name,
        endpoint=args.endpoint,
        framework=framework,
    )

    # 4. Upload dataset to A2E
    from a2e.client import Client  # type: ignore

    client = Client()
    examples = _build_examples(dataset.tasks)
    ds_name = identity.dataset_name
    a2e_dataset = client.datasets.create_dataset(
        name=ds_name,
        examples=examples,
        dataset_description=(
            f"A2E {dataset.name} run {identity.run_id} "
            f"({len(examples)} of {selection.available_n} examples)"
        ),
    )
    print(f"✓ uploaded isolated dataset '{ds_name}'")
    print(
        f"  sampling: {selection.strategy}; seed={selection.seed}; "
        f"task_ids={list(selection.task_ids)}"
    )

    # 5. Build evaluators (incl. optional LLM judge)
    judge_llm = None
    if "llm_judge" in args.evaluators:
        try:
            from a2e.evals.llm import LLM  # type: ignore

            judge_kwargs: dict[str, Any] = {
                "provider": "openai",
                "model": args.model or os.environ.get("A2E_LANGGRAPH_MODEL") or "gpt-4o-mini",
            }
            if args.api_base or os.environ.get("OPENAI_API_BASE"):
                judge_kwargs["base_url"] = args.api_base or os.environ["OPENAI_API_BASE"]
            if args.api_key or os.environ.get("OPENAI_API_KEY"):
                judge_kwargs["api_key"] = args.api_key or os.environ["OPENAI_API_KEY"]
            judge_llm = LLM(**judge_kwargs)
        except Exception as exc:
            logger.warning("LLM judge construction failed: %s", exc)
    evaluators = _build_evaluator_list(args.evaluators.split(","), judge_llm)
    if not evaluators:
        evaluators = None  # a2e-client skips evaluate_experiment when evaluators is None

    # 6. Run experiment
    from a2e.client.experiments import async_run_experiment  # type: ignore

    # Sandbox datasets (SWE-bench) run real docker containers. The per-task
    # lifecycle removes its own container, but a hard-killed run can leak one.
    # Sweep orphans before (clean slate, even if the previous run was killed)
    # and after (clean up once when done) — gated by A2E_SANDBOX_CLEANUP.
    sweep = None
    if ds_entry.get("kind") == "sandbox" and os.environ.get("A2E_SANDBOX_CLEANUP", "1") != "0":
        from ageneval.task.sandbox import sweep_sandbox_containers as sweep
        _pre = sweep(
            campaign_id=identity.run_id,
            include_image_orphans=False,
        )
        if _pre:
            print(f"🧹 pre-run sweep removed {len(_pre)} leftover sandbox container(s)")

    eval_label = args.evaluators if evaluators is not None else "(task-only)"
    print(
        f"▶ run {identity.run_id}: {args.dataset} x {args.agent} x "
        f"{actual_model} x [{eval_label}] over {len(examples)} examples"
    )
    is_sandbox = ds_entry.get("kind") == "sandbox"
    outer_timeout = (
        args.timeout
        if args.timeout is not None
        else (_sandbox_outer_timeout(dataset.tasks) if is_sandbox else 60)
    )
    print(
        f"  execution: concurrency={args.concurrency}; "
        f"outer_timeout={outer_timeout}s; retries={0 if is_sandbox else 3}"
    )
    explicit_agent_kwargs = {"model": str(actual_model)}
    if args.api_base:
        explicit_agent_kwargs["api_base"] = args.api_base
    if args.api_key or model_runtime is not None:
        explicit_agent_kwargs["api_key"] = "from-process-environment"
    legacy_run_root = (
        Path(os.environ.get("A2E_LEGACY_RUN_ROOT", ".a2e-legacy-runs"))
        / identity.run_id
    ).resolve()
    task_fn = _make_process_task_fn(
        dataset_name=args.dataset,
        harness=args.agent,
        bind_kwargs=bind_kwargs,
        resolved_model=runtime_model,
        explicit_agent_kwargs=explicit_agent_kwargs,
        project_name=project_name,
        endpoint=args.endpoint,
        timeout_seconds=outer_timeout,
        run_id=identity.run_id,
        run_root=legacy_run_root,
    )
    run_kwargs: dict[str, Any] = dict(
        dataset=a2e_dataset,
        task=task_fn,
        evaluators=evaluators,
        experiment_name=identity.experiment_name,
        experiment_description=(
            f"A2E CLI run {identity.run_id}: {args.dataset} x "
            f"{args.agent} x {actual_model}"
        ),
        experiment_metadata={
            **_build_experiment_metadata(
                agent_name=args.agent,
                agent=agent,
                sdk=framework,
            ),
            "run_id": identity.run_id,
            "dataset": args.dataset,
            "sampling_strategy": selection.strategy,
            "sample_seed": selection.seed,
            "requested_n": selection.requested_n,
            "available_n": selection.available_n,
            "selected_n": selection.selected_n,
            "sample_task_ids": list(selection.task_ids),
        },
        # SandboxScoringRunner applies the task.toml agent timeout and the
        # grader applies verifier_timeout_sec. The outer budget covers both.
        timeout=outer_timeout,
        # Replaying a stateful sandbox task can duplicate containers and API
        # calls. Inner layers already convert task failures into TaskTrace.
        retries=0 if is_sandbox else 3,
    )
    try:
        ran = asyncio.run(
            async_run_experiment(**run_kwargs, concurrency=args.concurrency)
        )
    finally:
        if sweep is not None:
            _post = sweep(
                campaign_id=identity.run_id,
                include_image_orphans=False,
            )
            if _post:
                print(f"🧹 cleaned up {len(_post)} sandbox container(s) after run")
    print()
    print("✓ experiment finished")
    print(f"  run_id: {identity.run_id}")
    if isinstance(ran, dict):
        print(f"  experiment_id: {ran.get('experiment_id', '')}")
        print(f"  dataset_version_id: {ran.get('dataset_version_id', '')}")
    print(f"  open http://localhost:6006/datasets — '{ds_name}'")
    if evaluators is None:
        print("  evaluators: (none — run eval/ separately)")
    else:
        print(f"  evaluators: {[e.__name__ for e in evaluators]}")

    provider.force_flush(timeout_millis=8000)
    if model_runtime is not None:
        model_runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
