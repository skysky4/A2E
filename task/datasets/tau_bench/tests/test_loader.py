from __future__ import annotations

import pytest
from ageneval.task.datasets.tau_bench.loader import load_tau_bench_tasks


def test_load_explicit_task_ids_in_requested_order() -> None:
    dataset = load_tau_bench_tasks(
        domain="retail",
        source="vendor",
        task_ids=("retail-0003", "retail-0001", "retail-0002"),
    )

    assert [task.task_id for task in dataset] == [
        "retail-0003",
        "retail-0001",
        "retail-0002",
    ]


def test_load_explicit_task_ids_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="retail-9999"):
        load_tau_bench_tasks(
            domain="retail",
            source="vendor",
            task_ids=("retail-9999",),
        )
