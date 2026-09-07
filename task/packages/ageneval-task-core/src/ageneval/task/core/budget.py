"""Shared LLM / loop budget so every harness calls the API with the same caps.

Fair comparison requires the same ``max_tokens``, turn/step budget, and
request timeout on every agent. Dataset ``agent_overrides`` (τ=30,
DeepSearchQA=8, sandbox=40) still apply equally to all harnesses.
"""

from __future__ import annotations

import os


def max_tokens() -> int:
    return int(os.environ.get("A2E_MAX_TOKENS", "4096"))


def max_turns() -> int:
    return int(os.environ.get("A2E_MAX_TURNS", os.environ.get("A2E_MAX_STEPS", "8")))


def max_steps() -> int:
    return int(os.environ.get("A2E_MAX_STEPS", os.environ.get("A2E_MAX_TURNS", "8")))


def llm_timeout() -> float:
    return float(os.environ.get("A2E_LLM_TIMEOUT", "180"))


def run_deadline() -> float:
    """Whole-agent wall clock. Same value for every harness on a dataset.

    ``run_n1.sh`` / ``run_full.sh`` set ``A2E_RUN_DEADLINE``
    (τ=1100, DeepSearchQA=620, GDPval=7000) so one harness cannot stop
    early while others keep going.
    """
    return float(
        os.environ.get(
            "A2E_RUN_DEADLINE",
            os.environ.get("A2E_AGNO_DEADLINE", "1800"),
        )
    )


def max_retries() -> int:
    return int(os.environ.get("A2E_LLM_MAX_RETRIES", "2"))


def tool_result_chars() -> int:
    return int(os.environ.get("A2E_TOOL_RESULT_CHARS", "2500"))


def remaining_deadline(start: float) -> float:
    """Seconds left on the shared wall. Always >= 1 so wait_for is valid."""
    import time

    return max(1.0, run_deadline() - (time.perf_counter() - start))
