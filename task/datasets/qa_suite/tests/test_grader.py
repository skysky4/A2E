from fractions import Fraction

import pytest

from ageneval.task.datasets.qa_suite import (
    BENCHMARKS,
    grade_qa_suite,
    normalize_math_number,
)

MC_BENCHMARKS = (
    "gpqa",
    "mmlu-pro",
    "arc-challenge",
    "truthfulqa",
    "agieval",
    "commonsenseqa",
    "hellaswag",
    "openbookqa",
)


@pytest.mark.parametrize("benchmark", MC_BENCHMARKS)
def test_all_multiple_choice_benchmarks_use_exact_letters(benchmark: str) -> None:
    result = grade_qa_suite(
        {"final_answer": '{"final_answer": "E"}'},
        {"expected_outputs": ["E"]},
        metadata={"benchmark": benchmark},
    )

    assert result["score"] == 1.0
    assert result["metrics"] == {"answer_letter_exact": 1.0}
    assert result["metadata"]["benchmark"] == benchmark


def test_bbh_uses_normalized_freeform_exact_match_from_input_metadata() -> None:
    result = grade_qa_suite(
        {"final_answer": "  TRUE. "},
        {"expected_outputs": ["true"]},
        input={"metadata": {"benchmark": "bbh"}},
    )

    assert result["score"] == 1.0
    assert result["metrics"] == {"freeform_exact": 1.0}


def test_math_parses_boxed_latex_fraction() -> None:
    result = grade_qa_suite(
        {"final_answer": r"$\boxed{\frac{3}{4}}$"},
        {"expected_outputs": ["0.75"]},
        metadata={"benchmark": "math"},
    )

    assert normalize_math_number(r"\boxed{\frac{3}{4}}") == Fraction(3, 4)
    assert result["score"] == 1.0
    assert result["metrics"] == {"numeric_exact": 1.0}


def test_dispatch_covers_exactly_the_configured_ten_keys() -> None:
    assert set(BENCHMARKS) == {*MC_BENCHMARKS, "bbh", "math"}

    result = grade_qa_suite("A", ["A"], metadata={"benchmark": "unknown"})
    assert result["score"] is None
    assert result["passed"] is None
    assert result["error"]
