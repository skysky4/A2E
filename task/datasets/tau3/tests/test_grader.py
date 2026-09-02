from __future__ import annotations

from ageneval.task.datasets.tau3.grader import GRADER_METADATA, tau3_grader
from ageneval.task.datasets.tau_bench.grader import tau_grader
from ageneval.task.datasets.tau_bench.reward import data_hash
from ageneval.task.datasets.tau_bench.runtime import load_domain_data


def test_tau3_delegates_to_shared_grader_and_marks_limitations() -> None:
    output = {
        "tau_data_hash": data_hash(load_domain_data("retail")),
        "tau_domain": "retail",
    }
    expected = {"expected_actions": [], "expected_outputs": []}
    assert tau3_grader(output, expected) == tau_grader(output, expected)
    assert GRADER_METADATA["official"] is False
    assert "voice" in " ".join(GRADER_METADATA["limitations"])
