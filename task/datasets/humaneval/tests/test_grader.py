from ageneval.task.datasets.humaneval.grader import GRADER, grade_humaneval


def test_grader_spec_and_execution_wrapper() -> None:
    report = grade_humaneval(
        {"final_answer": "    return a + b"},
        input={
            "initial_state": {
                "prompt": "def add(a, b):\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
                "entry_point": "add",
            }
        },
    )

    assert GRADER.id == "humaneval_pass"
    assert GRADER.mode == "posthoc"
    assert GRADER.grade is grade_humaneval
    assert GRADER.official is True
    assert report.score == 1.0
    assert report.metrics["humaneval_pass"] == 1.0
