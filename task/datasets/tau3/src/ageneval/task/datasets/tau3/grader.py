"""τ3 text-mode grader using the shared τ-bench state-hash policy.

This is not the complete official τ3 evaluation. The current A2E adapter omits
the benchmark's voice/full-duplex modality and grades only supported text-tool
tasks with the shared Sierra database/outcome checker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ageneval.task.core.grading import GraderSpec
from ageneval.task.datasets.tau_bench.grader import tau_grader

GRADER_METADATA = {
    "id": "tau3_grader",
    "official": False,
    "source": "tau-bench calculate_reward compatibility mode",
    "version": "text-tools-v1",
    "limitations": ("voice/full-duplex evaluation is not implemented",),
}


def tau3_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Delegate the supported text-tool subset to the shared τ grader."""
    return tau_grader(output, expected, input, domain)


grade = tau3_grader
GRADER = GraderSpec(
    id=GRADER_METADATA["id"],
    grade=tau3_grader,
    official=GRADER_METADATA["official"],
    source=GRADER_METADATA["source"],
    version=GRADER_METADATA["version"],
    metadata={"limitations": GRADER_METADATA["limitations"]},
)

__all__ = ["GRADER", "GRADER_METADATA", "grade", "tau3_grader"]
