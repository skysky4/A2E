from ageneval.task.datasets.traject_bench import grade_traject_bench


def test_traject_bench_combines_tool_and_answer_correctness() -> None:
    result = grade_traject_bench(
        {
            "final_answer": "The result is 901.",
            "tool_call_records": [
                {
                    "name": "calculate",
                    "arguments": {"expression": "47 * 19 + 8"},
                }
            ],
        },
        {
            "expected_actions": [
                {
                    "name": "calculate",
                    "arguments": {"expression": "47 * 19 + 8"},
                }
            ],
            "expected_outputs": ["901"],
        },
        metadata={"source": "vendor"},
    )

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["metrics"]["tool_action_correctness"] == 1.0
    assert result["metrics"]["answer_match"] == 1.0
    assert result["official"] is False


def test_traject_bench_averages_available_components_without_claiming_official() -> None:
    result = grade_traject_bench(
        {"final_answer": "wrong", "tool_calls": ["get_weather"]},
        {
            "expected_actions": [{"name": "get_weather"}],
            "expected_outputs": ["21"],
        },
    )

    assert result["score"] == 0.5
    assert result["passed"] is False
    assert result["official"] is False


def test_traject_bench_reports_no_references_as_unsupported() -> None:
    result = grade_traject_bench({}, {"expected_actions": [], "expected_outputs": []})

    assert result["score"] is None
    assert result["label"] == "unsupported"
    assert result["official"] is False
