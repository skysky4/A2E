"""Official τ-bench / τ2 / τ3 grader: Sierra ``calculate_reward`` pass^1.

This is the only default outcome metric for the τ family. It is the Sierra
policy (DB SHA-256 vs gold-action replay + required NL outputs), not A2E
``tool_recall``.

τ2 dual-control and τ3 voice are not implemented here; those datasets reuse
this same Sierra hash grader on the live retail / airline text tools.
"""

from __future__ import annotations

from typing import Any, Mapping

from ageneval.task.datasets.tau_bench.reward import official_tau_reward


def tau_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Registry name: ``tau_grader`` (alias: ``tau_reward``)."""
    return official_tau_reward(
        output=output, expected=expected, input=input, domain=domain
    )
