from ageneval.task.datasets.persistbench import grade_persistbench


def test_persistbench_scores_available_reference_unofficially() -> None:
    result = grade_persistbench(
        {"final_answer": "Your dog's name is Rocco."},
        {"expected_outputs": ["Rocco"]},
        metadata={"source": "vendor"},
    )

    assert result["score"] == 1.0
    assert result["metrics"] == {"reference_answer_match": 1.0}
    assert result["official"] is False


def test_persistbench_reports_missing_reference_as_unsupported() -> None:
    result = grade_persistbench(
        {"final_answer": "A plausible answer"},
        {"expected_outputs": []},
        metadata={"source": "upstream-full"},
    )

    assert result["score"] is None
    assert result["passed"] is None
    assert result["label"] == "unsupported"
    assert result["metadata"]["reference_available"] is False
    assert result["official"] is False
