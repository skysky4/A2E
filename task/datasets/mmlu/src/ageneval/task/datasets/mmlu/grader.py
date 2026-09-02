"""Deterministic, benchmark-local MMLU grading.

The public ``grade_mmlu`` signature intentionally matches the context supplied
by the future core grader adapter while keeping this package independent of it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ageneval.task.core.grading import GraderSpec

_ANSWER_RE = re.compile(r"^\s*[\(\[]?([A-D])[\)\].:]?\s*$", re.IGNORECASE)


def _final_answer(output: Any) -> str:
    if isinstance(output, Mapping):
        output = output.get("final_answer", output.get("answer", ""))
    text = str(output or "").strip()
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            text = str(value.get("final_answer", value.get("answer", text))).strip()
    return text


def _reference(expected: Any) -> str:
    if isinstance(expected, Mapping):
        expected = expected.get("expected_outputs", expected.get("answer", ""))
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        expected = expected[0] if expected else ""
    return str(expected or "").strip()


def _answer_letter(value: str) -> str | None:
    match = _ANSWER_RE.fullmatch(value)
    return match.group(1).upper() if match else None


def grade_mmlu(
    output: Any,
    expected: Any,
    input: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one MMLU response by exact normalized answer-letter equality."""
    del input, metadata
    predicted = _answer_letter(_final_answer(output))
    reference = _answer_letter(_reference(expected))
    score = float(predicted is not None and reference is not None and predicted == reference)
    return {
        "score": score,
        "passed": bool(score),
        "metrics": {"answer_letter_exact": score},
        "metadata": {"predicted": predicted, "reference": reference},
        "explanation": "Exact MMLU answer-letter match.",
        "official": True,
        "source": "cais/mmlu",
        "version": "local-letter-exact-v1",
    }


grade = grade_mmlu
GRADER = GraderSpec(
    id="mc_letter",
    grade=grade_mmlu,
    official=True,
    source="cais/mmlu",
    version="local-letter-exact-v1",
)

__all__ = ["GRADER", "grade", "grade_mmlu"]
