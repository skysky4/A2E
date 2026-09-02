#!/usr/bin/env python3
"""Prove that synchronous Trial work really overlaps across OS processes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.orchestrator.concurrency import PermitBroker, RuntimeMetrics
from ageneval.task.orchestrator.process import TrialProcessRunner
from ageneval.task.orchestrator.schema import LifecycleEvent


async def probe(concurrency: int, blocking_seconds: float, output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    profile = ModelProfile.model_validate(
        {
            "id": "concurrency-probe",
            "provider": "local",
            "model": "concurrency-probe",
            "upstream_protocol": "openai_chat_completions",
            "connection": {"api_key_env": "A2E_PROBE_KEY"},
            "concurrency": {"group": "probe", "max_sessions": concurrency},
        }
    )
    resolved = ResolvedModel(profile=profile, api_key="unused-probe-secret")
    broker = PermitBroker({"model:probe": concurrency})
    runtime = RuntimeMetrics()
    with tempfile.TemporaryDirectory(prefix="a2e-concurrency-probe-") as temporary:
        temp = Path(temporary)
        release = temp / "release"

        async def run_one(index: int) -> None:
            owner = f"probe:{index}"

            async def lifecycle(event: LifecycleEvent, _attempt: int | None) -> None:
                if event == LifecycleEvent.AGENT_START:
                    await broker.acquire(owner, "model:probe")
                elif event == LifecycleEvent.AGENT_END:
                    broker.release(owner, "model:probe")

            def activity(name: str, state: str) -> None:
                runtime.activity(owner, name, state)
                counter = runtime.snapshot()["activities"].get("probe:blocking", {})
                if counter.get("active") == concurrency and not release.exists():
                    release.touch()

            def process_event(state: str, pid: int) -> None:
                runtime.process_event(state, pid)
                if state != "start":
                    runtime.release_activities(owner)

            runner = TrialProcessRunner(
                script=root / "task/examples/run_trial_process_probe.py",
                cancellation_grace_seconds=2,
                lifecycle=lifecycle,
                activity=activity,
                process_event=process_event,
            )
            await runner.run(
                payload={
                    "cell": {"campaign_id": "probe", "cell_id": "probe-cell"},
                    "trial": {
                        "trial_id": f"probe-{index}",
                        "task_id": f"probe-{index}",
                        "repetition": 1,
                    },
                    "attempt": 1,
                    "release_path": str(release),
                    "blocking_seconds": blocking_seconds,
                    "probe_timeout": 30,
                },
                resolved_model=resolved,
                python=sys.executable,
                pythonpath="",
                stderr_path=temp / f"probe-{index}.stderr.log",
                result_path=temp / f"probe-{index}.result.json",
                timeout_seconds=40,
            )

        started = time.perf_counter()
        await asyncio.gather(*(run_one(index) for index in range(concurrency)))
        elapsed = time.perf_counter() - started
    permit = broker.snapshot()["model:probe"]
    metrics = runtime.snapshot()
    processes = metrics["trial_processes"]
    blocking = metrics["activities"].get("probe:blocking", {})
    errors = []
    if processes["high_water"] != concurrency:
        errors.append("Trial process high-water did not reach configured concurrency")
    if blocking.get("high_water") != concurrency:
        errors.append("synchronous blocking work did not overlap at configured concurrency")
    if processes["active"] != 0 or blocking.get("active") != 0 or permit["active"] != 0:
        errors.append("process, activity, or permit counters did not return to zero")
    if permit["high_water"] != concurrency:
        errors.append("model permit did not reach configured concurrency")
    report = {
        "status": "passed" if not errors else "failed",
        "concurrency": concurrency,
        "blocking_seconds": blocking_seconds,
        "elapsed_seconds": elapsed,
        "permits": permit,
        "runtime": metrics,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--blocking-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.blocking_seconds <= 0:
        parser.error("concurrency and blocking seconds must be positive")
    report = asyncio.run(
        probe(args.concurrency, args.blocking_seconds, args.output.resolve())
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
