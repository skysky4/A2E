from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from ageneval.model.gateway import ModelProfile, ResolvedModel
from ageneval.task.orchestrator.controller import CampaignController
from ageneval.task.orchestrator.matrix import CampaignPlan, CellSpec, TrialSpec
from ageneval.task.orchestrator.schema import CampaignConfig, LifecycleEvent, TrialResult
from ageneval.task.orchestrator.state import RunDirectory


def test_controller_schedules_independent_trial_processes(monkeypatch, tmp_path: Path) -> None:
    profile = ModelProfile.model_validate(
        {
            "id": "model",
            "provider": "test",
            "model": "model",
            "upstream_protocol": "openai_chat_completions",
            "connection": {"api_key_env": "TEST_KEY"},
            "concurrency": {"group": "model", "max_sessions": 4},
        }
    )
    config = CampaignConfig.model_validate(
        {
            "name": "process-scheduler",
            "models": ["model"],
            "benchmarks": [{"id": "mmlu"}],
            "harnesses": ["langgraph"],
            "execution": {
                "n_concurrent_trials": 4,
                "n_active_cells": 1,
                "n_concurrent_sandboxes": 4,
                "n_concurrent_model_sessions": 4,
                "n_concurrent_graders": 4,
                "n_concurrent_uploads": 4,
                "queue_capacity": 4,
            },
        }
    )
    cell = CellSpec("cell", "model", "mmlu", "langgraph", profile.digest())
    trials = tuple(TrialSpec(f"trial-{i}", "cell", f"task-{i}", 1) for i in range(4))
    run_directory = RunDirectory(tmp_path / "campaign")
    run_directory.initialize(config=config.model_dump(mode="json"), lock={})
    controller = CampaignController(
        config=config,
        config_path=None,
        run_directory=run_directory,
        model_directory=tmp_path,
        repo_root=Path(__file__).resolve().parents[4],
    )
    controller.plan = CampaignPlan("campaign", (cell,), trials)
    controller.profiles = {"model": profile}
    controller.tasks = {
        "mmlu": [
            {
                "task_id": trial.task_id,
                "instruction": "answer",
                "initial_state": {},
                "expected_actions": [],
                "expected_outputs": [],
                "metadata": {},
                "sandbox": None,
            }
            for trial in trials
        ]
    }
    controller.benchmarks = {"mmlu": {"id": "mmlu", "graders": []}}
    resolved = ResolvedModel(profile=profile, api_key="secret")
    entered = 0
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_run(self, *, payload, **_kwargs):
        nonlocal entered
        if self.process_event:
            self.process_event("start", 1000 + entered)
        await self.lifecycle(LifecycleEvent.AGENT_START, payload["attempt"])
        entered += 1
        if entered == 4:
            all_entered.set()
        await asyncio.wait_for(release.wait(), timeout=2)
        await self.lifecycle(LifecycleEvent.AGENT_END, payload["attempt"])
        if self.process_event:
            self.process_event("end", 1000 + entered)
        now = datetime.now(timezone.utc)
        return TrialResult(
            trial_id=payload["trial"]["trial_id"],
            cell_id="cell",
            task_id=payload["trial"]["task_id"],
            repetition=1,
            attempt=1,
            status="local_complete",
            started_at=now,
            ended_at=now,
        )

    async def fake_upload(_sink, _state, result):
        return result.model_copy(update={"status": "completed", "uploaded": True})

    monkeypatch.setattr(
        "ageneval.task.orchestrator.process.TrialProcessRunner.run", fake_run
    )
    monkeypatch.setattr(controller, "_upload_one", fake_upload)

    async def scenario() -> None:
        running = asyncio.create_task(
            controller._run_trial_processes(
                sink=object(),
                server_state={"cells": {"cell": {"project_name": "test"}}},
                resolved_models={"model": resolved},
                rerun_failed=False,
            )
        )
        await asyncio.wait_for(all_entered.wait(), timeout=2)
        release.set()
        await running

    asyncio.run(scenario())
    assert controller.concurrency_stats["global"]["high_water"] == 4
    assert controller.concurrency_stats["model:model"]["high_water"] == 4
    assert controller.runtime_stats["trial_processes"]["high_water"] == 4
    assert controller.runtime_stats["trial_processes"]["active"] == 0
    assert all(
        run_directory.load_trial_result(trial.trial_id).status == "completed"
        for trial in trials
    )
