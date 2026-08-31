"""DeepSearchQA official Appendix A grader."""

from __future__ import annotations

from ageneval.task.datasets.deepsearchqa.grader import deepsearch_grader
from ageneval.task.runners.registry import _eval_deepsearch_grader, _eval_deepsearch_match


def test_single_answer_containment():
    out = {"final_answer": "The diagnosis is Joint hypermobility syndrome."}
    exp = {"expected_outputs": ["Joint hypermobility syndrome"]}
    inp = {"initial_state": {"answer_type": "Single Answer"}}
    assert deepsearch_grader(out, exp, inp) == 1.0
    assert deepsearch_grader({"final_answer": "nope"}, exp, inp) == 0.0


def test_set_answer_recall():
    out = {"final_answer": "apple, banana"}
    exp = {"expected_outputs": ["apple, banana, cherry"]}
    inp = {"initial_state": {"answer_type": "Set Answer"}}
    assert abs(deepsearch_grader(out, exp, inp) - (2 / 3)) < 1e-9


def test_registry_aliases_match():
    out = {"final_answer": "Hypermobility"}
    exp = {"expected_outputs": ["Hypermobility"]}
    inp = {"initial_state": {"answer_type": "Single Answer"}}
    assert _eval_deepsearch_grader(out, exp, inp) == 1.0
    assert _eval_deepsearch_match(out, exp, inp) == 1.0
    assert _eval_deepsearch_grader.__name__ == "deepsearch_grader"
    assert _eval_deepsearch_match.__name__ == "deepsearch_match"
