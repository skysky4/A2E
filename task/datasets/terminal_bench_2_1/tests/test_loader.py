from __future__ import annotations

from pathlib import Path

from ageneval.task.core.dataset import TaskInput
from ageneval.task.datasets.terminal_bench_2_1 import loader


def test_exclude_categories_is_case_insensitive_and_precedes_limit(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = {
        "excluded": TaskInput(
            task_id="excluded",
            instruction="",
            metadata={"category": "Security"},
        ),
        "included": TaskInput(
            task_id="included",
            instruction="",
            metadata={"category": "software-engineering"},
        ),
    }
    monkeypatch.setattr(loader, "_tasks_dir", lambda: tmp_path)
    monkeypatch.setattr(loader, "list_task_names", lambda: list(tasks))
    monkeypatch.setattr(loader, "_safe_build", lambda path: tasks[path.name])
    monkeypatch.setattr(loader, "_local_images", set)

    dataset = loader.load_terminal_bench_2_1_tasks(
        n=1,
        exclude_categories=[" SECURITY "],
    )

    assert [task.task_id for task in dataset.tasks] == ["included"]
