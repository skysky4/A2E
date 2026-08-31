"""Official τ-bench grader backed by Sierra ``calculate_reward``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ageneval.task.core.grading import GraderSpec
from ageneval.task.datasets.tau_bench.reward import official_tau_reward

GRADER_METADATA = {
    "id": "tau_grader",
    "aliases": ("tau_reward",),
    "official": True,
    "source": "sierra-research/tau-bench calculate_reward",
    "version": "pass^1",
}


def tau_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Score a τ-bench trajectory with the official binary outcome metric."""
    return official_tau_reward(
        output=output,
        expected=expected,
        input=input,
        domain=domain,
    )


grade = tau_grader
GRADER = GraderSpec(
    id=GRADER_METADATA["id"],
    grade=tau_grader,
    official=GRADER_METADATA["official"],
    source=GRADER_METADATA["source"],
    version=GRADER_METADATA["version"],
    aliases=GRADER_METADATA["aliases"],
)

__all__ = ["GRADER", "GRADER_METADATA", "grade", "tau_grader"]
