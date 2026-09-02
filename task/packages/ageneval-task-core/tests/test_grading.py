from __future__ import annotations

import asyncio

import pytest

from ageneval.task.core.grading import (
    GradeReport,
    GraderSpec,
    normalize_grade,
    platform_evaluator,
    run_grader,
)


def test_normalize_scalar_preserves_provenance() -> None:
    spec = GraderSpec(
        id="answer",
        grade=lambda: 1,
        source="paper",
        version="1",
        metadata={"limitations": ["text-only"]},
    )
    report = normalize_grade(1.0, spec)
    assert report.score == 1.0
    assert report.passed is True
    assert report.source == "paper"
    assert report.version == "1"
    assert report.metadata["limitations"] == ["text-only"]


def test_run_grader_binds_only_declared_context() -> None:
    def grade(output: dict, expected: dict) -> float:
        return float(output["answer"] == expected["answer"])

    report = asyncio.run(
        run_grader(
            GraderSpec(id="answer", grade=grade),
            output={"answer": "A"},
            expected={"answer": "A"},
            input={"instruction": "question"},
            metadata={"benchmark": "demo"},
        )
    )
    assert report.score == 1.0


def test_factory_requires_runtime() -> None:
    spec = GraderSpec(id="judge", factory=lambda runtime: runtime)
    with pytest.raises(ValueError, match="requires a grading runtime"):
        spec.resolve()


def test_platform_adapter_has_stable_name_and_report_shape() -> None:
    evaluator = platform_evaluator(
        GraderSpec(
            id="canonical_score",
            grade=lambda output, expected: output["value"] == expected["value"],
            official=False,
            source="local",
        )
    )
    value = asyncio.run(
        evaluator(
            {"value": 3},
            {"value": 3},
            {},
            {},
        )
    )
    assert evaluator.__qualname__ == "canonical_score"
    assert value["score"] == 1.0
    assert value["metadata"]["official"] is False


def test_existing_grade_report_is_not_rewritten() -> None:
    original = GradeReport(score=0.5, official=False, source="custom")
    spec = GraderSpec(id="answer", grade=lambda: original)
    assert normalize_grade(original, spec) is original


def test_inline_summarizer_converts_live_report() -> None:
    spec = GraderSpec(
        id="resolved",
        grade=lambda *_args: {"resolved": True},
        summarize=lambda output: bool(output["resolved"]),
        mode="inline",
    )
    report = spec.summarize_inline({"resolved": True})
    assert report.score == 1.0
    assert report.passed is True
