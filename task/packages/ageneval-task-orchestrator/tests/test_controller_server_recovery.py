from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ageneval.task.orchestrator.controller import CampaignController


def test_stale_server_ids_are_discarded_before_resume() -> None:
    updates: list[dict] = []
    controller = CampaignController.__new__(CampaignController)
    controller.plan = SimpleNamespace(
        campaign_id="campaign-1",
        cells=[SimpleNamespace(cell_id="cell-1", benchmark="benchmark-1")],
    )
    controller.tasks = {"benchmark-1": [{"task_id": "task-1"}]}
    controller.lock = {}
    controller.run_directory = SimpleNamespace(update_lock=updates.append)

    class Sink:
        async def locked_dataset_exists(self, **_kwargs) -> bool:
            return False

        async def locked_experiment_exists(self, **_kwargs) -> bool:
            raise AssertionError("experiment validation must stop after a stale dataset")

    state = {
        "datasets": {
            "benchmark-1": {
                "id": "dataset-old",
                "version_id": "version-old",
                "examples": {"task-1": "example-old"},
            }
        },
        "cells": {
            "cell-1": {
                "experiment_id": "experiment-old",
                "project_name": "project-old",
            }
        },
    }

    recovered = asyncio.run(controller._validated_server_state(Sink(), state))

    assert recovered == {"datasets": {}, "cells": {}}
    assert controller.lock["server"] == recovered
    assert updates == [{"server": recovered}]
