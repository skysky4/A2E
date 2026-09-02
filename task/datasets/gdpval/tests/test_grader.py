from __future__ import annotations

from ageneval.task.datasets.gdpval.grader import (
    GRADER_METADATA,
    make_gdp_grader,
)


class _FakeJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate_text(self, *, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_gdp_grader_scores_rubric_with_stable_name() -> None:
    judge = _FakeJudge("SCORE=1; EXPLANATION=The deliverable meets the rubric.")
    grader = make_gdp_grader(judge)
    result = grader(
        {"final_answer": "A professional report."},
        {"expected_outputs": ["Include a report."]},
        {"instruction": "Write a report."},
    )
    assert grader.__qualname__ == "gdp_grader"
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert "Include a report." in judge.prompt
    assert result["official"] is False


def test_gdp_grader_reports_runtime_failure() -> None:
    class _FailingJudge:
        def generate_text(self, *, prompt: str) -> str:
            raise RuntimeError("offline")

    result = make_gdp_grader(_FailingJudge())(
        {"final_answer": ""},
        {"expected_outputs": ["rubric"]},
        {"instruction": "task"},
    )
    assert result["score"] == 0.0
    assert result["label"] == "error"
    assert "offline" in result["explanation"]
    assert GRADER_METADATA["limitations"]
