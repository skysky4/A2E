from ageneval.task.datasets.mmlu import grade_mmlu


def test_mmlu_scores_exact_json_answer_letter() -> None:
    result = grade_mmlu(
        {"final_answer": '{"final_answer": "b"}'},
        {"expected_outputs": ["B"]},
    )

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["official"] is True


def test_mmlu_rejects_non_exact_or_wrong_letters() -> None:
    verbose = grade_mmlu({"final_answer": "I choose B"}, {"expected_outputs": ["B"]})
    wrong = grade_mmlu("C", ["B"])

    assert verbose["score"] == 0.0
    assert wrong["score"] == 0.0
