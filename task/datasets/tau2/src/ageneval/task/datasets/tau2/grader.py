"""τ2 compatibility grader using the shared τ-bench state-hash policy.

This is not the complete official τ2 evaluation: A2E's current adapter does
not model τ2 dual-control interactions. It scores the supported retail/airline
text-tool subset with the shared Sierra database/outcome checker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ageneval.task.core.grading import GraderSpec
from ageneval.task.datasets.tau_bench.grader import tau_grader

GRADER_METADATA = {
    "id": "tau2_grader",
    "official": False,
    "source": "tau-bench calculate_reward compatibility mode",
    "version": "text-tools-v1",
    "limitations": ("dual-control is not implemented",),
}


def tau2_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Delegate the supported text-tool subset to the shared τ grader."""
    return tau_grader(output, expected, input, domain)


grade = tau2_grader
GRADER = GraderSpec(
    id=GRADER_METADATA["id"],
    grade=tau2_grader,
    official=GRADER_METADATA["official"],
    source=GRADER_METADATA["source"],
    version=GRADER_METADATA["version"],
    metadata={"limitations": GRADER_METADATA["limitations"]},
)

__all__ = ["GRADER", "GRADER_METADATA", "grade", "tau2_grader"]
