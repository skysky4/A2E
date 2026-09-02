from __future__ import annotations

from ageneval.task.datasets.deepsearchqa.grader import (
    GRADER_METADATA,
    deepsearch_grader,
)


def test_single_answer_uses_case_insensitive_containment() -> None:
    expected = {"expected_outputs": ["Joint hypermobility syndrome"]}
    input_value = {"initial_state": {"answer_type": "Single Answer"}}
    assert deepsearch_grader(
        {"final_answer": "The diagnosis is joint hypermobility syndrome."},
        expected,
        input_value,
    ) == 1.0
    assert deepsearch_grader(
        {"final_answer": "A different diagnosis."},
        expected,
        input_value,
    ) == 0.0


def test_set_answer_uses_gold_item_recall() -> None:
    score = deepsearch_grader(
        {"final_answer": "apple, banana"},
        {"expected_outputs": ["apple, banana, cherry"]},
        {"initial_state": {"answer_type": "Set Answer"}},
    )
    assert score == 2 / 3


def test_deterministic_grader_metadata_is_unofficial() -> None:
    assert GRADER_METADATA["official"] is False
    assert "autorater" in " ".join(GRADER_METADATA["limitations"]).lower()
