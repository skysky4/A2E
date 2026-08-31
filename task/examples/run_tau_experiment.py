"""End-to-end τ-bench experiment visible in A2E UI.

What this script does
---------------------
1. Sets up OpenInference instrumentation (langchain).
2. Uploads τ-bench tasks to A2E as a *Dataset* (visible in UI under
   "Datasets").
3. Wraps the multi-agent `LangGraphTauAgent` as a A2E *experiment task*.
4. Runs the benchmark-owned Sierra grader.
5. Calls `run_experiment` so the task trace and official score land in A2E.

Usage
-----
    # 1. start a2e in another shell
    uv run a2e serve

    # 2. then:
    OPENAI_API_KEY=...  OPENAI_API_BASE=http://.../v1/  \
      A2E_LANGGRAPH_MODEL=deepseek-v4-pro \
      uv run --frozen python examples/run_tau_experiment.py --domain retail

Open http://localhost:6006 → Datasets / Experiments tab.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from ageneval.task.agents.langgraph import LangGraphTauAgent
from ageneval.task.core import platform_evaluator, run_grader, setup_instrumentation
from ageneval.task.datasets.tau_bench import load_tau_bench_tasks
from ageneval.task.runners import (
    DEFAULT_SAMPLE_SIZE,
    build_experiment_metadata,
    build_run_identity,
    grader_for_dataset,
    sample_dataset,
    wrap_agent_for_dataset,
)

logger = logging.getLogger(__name__)


def _build_examples(tasks):
    """Convert TaskInput records into a2e-client dataset rows."""
    rows = []
    for t in tasks:
        rows.append(
            {
                "input": {"instruction": t.instruction, "initial_state": dict(t.initial_state)},
                "output": {
                    "expected_outputs": list(t.expected_outputs),
                    "expected_actions": list(t.expected_actions),
                },
                "metadata": {"task_id": t.task_id, **dict(t.metadata)},
            }
        )
    return rows


def _make_task_fn(agent: LangGraphTauAgent):
    """Build an A2E task that runs the official session and grader once."""
    from ageneval.task.core import TaskInput

    runner = wrap_agent_for_dataset("tau-bench", agent)
    grader = grader_for_dataset("tau-bench")

    def task_fn(input: dict, expected: dict, metadata: dict) -> dict:
        task_input = TaskInput(
            task_id=metadata.get("task_id", "?"),
            instruction=input.get("instruction", ""),
            initial_state=input.get("initial_state", {}),
            expected_outputs=expected.get("expected_outputs") or (),
            expected_actions=expected.get("expected_actions") or (),
            metadata=metadata,
        )
        trace = asyncio.run(runner.run(task_input))
        output = {
            "final_answer": trace.final_answer or "",
            "tool_calls": [tc.name for tc in trace.tool_calls],
            "tool_call_records": [
                {
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                    "result": tc.result,
                    "error": tc.error,
                }
                for tc in trace.tool_calls
            ],
            "status": trace.status,
            "turns": trace.turns,
            "trace_id": trace.trace_id,
            "error": trace.error,
            **dict(trace.raw),
        }
        report = asyncio.run(
            run_grader(
                grader,
                output=output,
                expected=expected,
                input={
                    "instruction": task_input.instruction,
                    "initial_state": task_input.initial_state,
                },
                metadata=metadata,
                example=task_input,
            )
        )
        output["grade_report"] = report.as_dict()
        return output

    return task_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="retail", choices=["retail", "airline"])
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--endpoint", default=None, help="OTLP endpoint override")
    parser.add_argument("--model", default=None, help="overrides A2E_LANGGRAPH_MODEL")
    parser.add_argument("--api-base", default=None, help="overrides OPENAI_API_BASE")
    parser.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
    if args.n <= 0:
        parser.error("--n must be a positive integer")

    # 1. Load the complete domain and draw this run's exact random sample.
    tasks = load_tau_bench_tasks(domain=args.domain, n=None)
    tasks, selection = sample_dataset(
        tasks,
        n=args.n,
        seed=args.sample_seed,
    )
    examples = _build_examples(tasks.tasks)

    # 2. Build a unique identity before instrumentation or persistence.
    agent = LangGraphTauAgent(
        domain=args.domain,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    actual_model = str(getattr(agent, "_model_name", None) or args.model or "default-model")
    identity = build_run_identity(
        dataset_name=tasks.name,
        dataset_key=f"tau-bench-{args.domain}",
        agent_name="langgraph",
        model=actual_model,
        run_id=args.run_id,
    )

    # 3. Instrumentation (so the agent's spans land in a2e).
    provider = setup_instrumentation(
        project_name=args.experiment_name or identity.project_name,
        endpoint=args.endpoint,
        framework="langchain",
    )

    # 4. Upload as a new A2E dataset. Persistence errors must propagate; they
    # must never be converted into reuse of an older dataset version.
    from a2e.client import Client  # type: ignore

    client = Client()
    dataset_name = identity.dataset_name
    dataset = client.datasets.create_dataset(
        name=dataset_name,
        examples=examples,
        dataset_description=(
            f"tau-bench {args.domain} run {identity.run_id} "
            f"({len(examples)} of {selection.available_n} examples)"
        ),
    )
    print(f"✓ uploaded isolated dataset '{dataset_name}' with {len(examples)} examples")
    print(f"  sampling: random; seed={selection.seed}; task_ids={list(selection.task_ids)}")

    # 5. Build the A2E experiment task.
    task_fn = _make_task_fn(agent)

    # 6. The task embeds the benchmark grade; this adapter persists it through
    # the platform's existing experiment annotation boundary.
    grader = grader_for_dataset("tau-bench")
    platform_graders = [platform_evaluator(grader)]

    # 7. Run the experiment.
    from a2e.client.experiments import run_experiment  # type: ignore

    print(f"▶ running experiment over {len(examples)} examples …")
    ran = run_experiment(
        dataset=dataset,
        task=task_fn,
        evaluators=platform_graders,
        experiment_name=args.experiment_name or identity.experiment_name,
        experiment_description=(
            f"tau-bench {args.domain} run {identity.run_id} via langgraph"
        ),
        experiment_metadata={
            **build_experiment_metadata(
                agent_name="langgraph",
                agent=agent,
                sdk="langchain",
            ),
            "run_id": identity.run_id,
            "dataset": "tau-bench",
            "sampling_strategy": selection.strategy,
            "sample_seed": selection.seed,
            "requested_n": selection.requested_n,
            "available_n": selection.available_n,
            "selected_n": selection.selected_n,
            "sample_task_ids": list(selection.task_ids),
        },
    )

    print()
    print("✓ experiment finished")
    print(f"  run_id: {identity.run_id}")
    print("  open http://localhost:6006/datasets to see the τ-bench dataset")
    print("  open http://localhost:6006/experiments to see this run's per-example scores")
    if hasattr(ran, "summary"):
        try:
            print(json.dumps(ran.summary, ensure_ascii=False, indent=2, default=str)[:1500])
        except Exception:
            pass

    provider.force_flush(timeout_millis=8000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
