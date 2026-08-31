"""Deterministic GSM8K final-answer extraction and grading."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from ageneval.task.core.grading import GraderSpec

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
)
_EXPLICIT_RE = re.compile(
    r"(?:####|final\s+answer\s*(?:is|:)?|answer\s*(?:is|:))\s*(.+)",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("final_answer", value.get("answer", ""))
    text = str(value or "").strip()
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, Mapping):
            text = str(decoded.get("final_answer", decoded.get("answer", text))).strip()
    return text


def _reference(expected: Any) -> str:
    if isinstance(expected, Mapping):
        expected = expected.get("expected_outputs", expected.get("answer", ""))
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        expected = expected[0] if expected else ""
    return _text(expected)


def extract_final_numeric_answer(value: Any) -> str | None:
    """Extract the last numeric token, preferring an explicit answer marker."""
    text = _text(value)
    explicit = list(_EXPLICIT_RE.finditer(text))
    search_text = explicit[-1].group(1) if explicit else text
    numbers = _NUMBER_RE.findall(search_text)
    if not numbers and explicit:
        numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(" ", "") if numbers else None


def normalize_numeric_answer(value: Any) -> Fraction | None:
    """Normalize commas, decimals, fractions, signs, and percent suffixes."""
    token = extract_final_numeric_answer(value)
    if token is None:
        return None
    token = token.replace(",", "").removesuffix("%")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            denominator_value = Decimal(denominator)
            if denominator_value == 0:
                return None
            return Fraction(Decimal(numerator)) / Fraction(denominator_value)
        return Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def grade_gsm8k(
    output: Any,
    expected: Any,
    input: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one GSM8K response by exact normalized final-number equality."""
    del input, metadata
    predicted = normalize_numeric_answer(output)
    reference = normalize_numeric_answer(_reference(expected))
    score = float(predicted is not None and reference is not None and predicted == reference)
    return {
        "score": score,
        "passed": bool(score),
        "metrics": {"numeric_exact": score},
        "metadata": {
            "predicted": str(predicted) if predicted is not None else None,
            "reference": str(reference) if reference is not None else None,
        },
        "explanation": "Exact match after GSM8K final-number normalization.",
        "official": True,
        "source": "openai/gsm8k",
        "version": "local-numeric-exact-v1",
    }


grade = grade_gsm8k
GRADER = GraderSpec(
    id="numeric_match",
    grade=grade_gsm8k,
    official=True,
    source="openai/gsm8k",
    version="local-numeric-exact-v1",
)

__all__ = [
    "GRADER",
    "extract_final_numeric_answer",
    "grade",
    "grade_gsm8k",
    "normalize_numeric_answer",
]
