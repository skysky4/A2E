from fractions import Fraction

from ageneval.task.datasets.gsm8k import (
    extract_final_numeric_answer,
    grade_gsm8k,
    normalize_numeric_answer,
)


def test_extracts_explicit_final_number_instead_of_intermediate_work() -> None:
    output = "First I found 12, then multiplied by 100.\n#### $1,200.00"

    assert extract_final_numeric_answer(output) == "1,200.00"
    assert normalize_numeric_answer(output) == Fraction(1200)


def test_gsm8k_normalizes_json_decimals_and_fractions() -> None:
    result = grade_gsm8k(
        {"final_answer": '{"final_answer": "The final answer is 3/2."}'},
        {"expected_outputs": ["1.500"]},
    )

    assert result["score"] == 1.0
    assert result["metrics"]["numeric_exact"] == 1.0
    assert result["official"] is True


def test_gsm8k_reports_unparseable_answer_as_incorrect() -> None:
    result = grade_gsm8k("no numeric answer", ["42"])

    assert result["score"] == 0.0
    assert result["metadata"]["predicted"] is None
