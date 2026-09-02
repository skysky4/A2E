from __future__ import annotations

from ageneval.task.core.grading import GraderSpec
from ageneval.task.runners.registry import DATASETS, grader_for_dataset


def test_every_dataset_resolves_its_owned_grader() -> None:
    for dataset, entry in DATASETS.items():
        spec = grader_for_dataset(dataset)
        assert isinstance(spec, GraderSpec)
        assert spec.id
        assert spec.source
        assert spec.version
        assert "score" not in entry
        assert (spec.mode == "inline") == (entry["kind"] == "sandbox")


def test_dynamic_graders_preserve_provenance() -> None:
    assert grader_for_dataset("gdpval-aa").factory is not None
    assert grader_for_dataset("math").official is False
    assert grader_for_dataset("persistbench").official is False
    assert grader_for_dataset("traject-bench").official is False
    assert grader_for_dataset("tau2").official is False
    assert grader_for_dataset("tau3").official is False


def test_sandbox_graders_define_live_result_summarizers() -> None:
    for dataset in (
        "swe-bench-lite",
        "swe-bench-verified",
        "swe-bench-pro",
        "terminal-bench-2",
        "terminal-bench-2.1",
    ):
        assert grader_for_dataset(dataset).summarize is not None
