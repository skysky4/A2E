"""Benchmark-aware deterministic graders for the ten QA Suite datasets."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from ageneval.task.core.grading import GraderSpec
from ageneval.task.datasets.qa_suite.benchmarks import BENCHMARKS

_MC_KEYS = frozenset(
    {
        "gpqa",
        "mmlu-pro",
        "arc-challenge",
        "truthfulqa",
        "agieval",
        "commonsenseqa",
        "hellaswag",
        "openbookqa",
    }
)
_LETTER_RE = re.compile(r"^\s*[\(\[]?([A-P])[\)\].:]?\s*$", re.IGNORECASE)
_LATEX_FRAC_RE = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_SLASH_FRAC_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
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


def _benchmark_key(
    input: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None
) -> str | None:
    candidates: list[Mapping[str, Any]] = []
    if metadata:
        candidates.append(metadata)
    if input:
        nested = input.get("metadata")
        if isinstance(nested, Mapping):
            candidates.append(nested)
        candidates.append(input)
    for candidate in candidates:
        value = candidate.get("benchmark") or candidate.get("dataset")
        if value:
            return str(value).strip().lower()
    return None


def _grade_letter(output: Any, expected: Any) -> tuple[float, dict[str, Any]]:
    prediction_match = _LETTER_RE.fullmatch(_text(output))
    reference_match = _LETTER_RE.fullmatch(_reference(expected))
    predicted = prediction_match.group(1).upper() if prediction_match else None
    reference = reference_match.group(1).upper() if reference_match else None
    score = float(predicted is not None and reference is not None and predicted == reference)
    return score, {"predicted": predicted, "reference": reference}


def normalize_bbh_answer(value: Any) -> str:
    """Apply conservative BBH exact-match normalization."""
    text = re.sub(r"\s+", " ", _text(value)).strip().casefold()
    return text[:-1].rstrip() if text.endswith((".", ";")) else text


def _grade_bbh(output: Any, expected: Any) -> tuple[float, dict[str, Any]]:
    predicted = normalize_bbh_answer(output)
    reference = normalize_bbh_answer(_reference(expected))
    score = float(bool(predicted) and bool(reference) and predicted == reference)
    return score, {"predicted": predicted or None, "reference": reference or None}


def _boxed_contents(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return text


def normalize_math_number(value: Any) -> Fraction | None:
    """Parse a MATH numeric answer, including boxed and fractional forms."""
    text = _boxed_contents(_text(value)).replace("$", "").replace(",", "").strip()
    latex_fractions = list(_LATEX_FRAC_RE.finditer(text))
    slash_fractions = list(_SLASH_FRAC_RE.finditer(text))
    try:
        if latex_fractions:
            match = latex_fractions[-1]
            denominator = Decimal(match.group(2))
            return None if denominator == 0 else Fraction(Decimal(match.group(1))) / Fraction(denominator)
        if slash_fractions:
            match = slash_fractions[-1]
            denominator = Decimal(match.group(2))
            return None if denominator == 0 else Fraction(Decimal(match.group(1))) / Fraction(denominator)
        numbers = _NUMBER_RE.findall(text)
        return Fraction(Decimal(numbers[-1])) if numbers else None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _grade_math(output: Any, expected: Any) -> tuple[float, dict[str, Any]]:
    predicted = normalize_math_number(output)
    reference = normalize_math_number(_reference(expected))
    score = float(predicted is not None and reference is not None and predicted == reference)
    return score, {
        "predicted": str(predicted) if predicted is not None else None,
        "reference": str(reference) if reference is not None else None,
    }


def grade_qa_suite(
    output: Any,
    expected: Any,
    input: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch to MC, BBH free-form, or MATH numeric grading by metadata."""
    benchmark = _benchmark_key(input, metadata)
    if benchmark not in BENCHMARKS:
        return {
            "score": None,
            "passed": None,
            "metrics": {},
            "metadata": {"benchmark": benchmark},
            "explanation": "QA Suite grading requires a known benchmark metadata key.",
            "error": f"unsupported QA Suite benchmark: {benchmark!r}",
            "official": True,
            "source": "qa-suite",
            "version": "local-dispatch-v1",
        }

    if benchmark in _MC_KEYS:
        score, details = _grade_letter(output, expected)
        metric = "answer_letter_exact"
    elif benchmark == "bbh":
        score, details = _grade_bbh(output, expected)
        metric = "freeform_exact"
    else:
        score, details = _grade_math(output, expected)
        metric = "numeric_exact"

    return {
        "score": score,
        "passed": bool(score),
        "metrics": {metric: score},
        "metadata": {"benchmark": benchmark, **details},
        "explanation": f"QA Suite {benchmark} {metric} grading.",
        "official": benchmark != "math",
        "source": BENCHMARKS[benchmark].hf_id,
        "version": "local-dispatch-v1",
    }


grade = grade_qa_suite


def grader_for_benchmark(benchmark: str) -> GraderSpec:
    """Return the benchmark-owned strategy for one QA Suite registry key."""
    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown QA Suite benchmark: {benchmark}")
    if benchmark in _MC_KEYS:
        grader_id = "mc_letter"
        official = True
    elif benchmark == "bbh":
        grader_id = "exact_match"
        official = True
    else:
        grader_id = "numeric_match"
        # The local MATH parser does not implement full symbolic equivalence.
        official = False

    def selected_grader(
        output: Any,
        expected: Any,
        input: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return grade_qa_suite(
            output,
            expected,
            input,
            {"benchmark": benchmark, **dict(metadata or {})},
        )

    selected_grader.__name__ = grader_id
    selected_grader.__qualname__ = grader_id
    return GraderSpec(
        id=grader_id,
        grade=selected_grader,
        official=official,
        source=BENCHMARKS[benchmark].hf_id,
        version="local-dispatch-v1",
    )

__all__ = [
    "grade",
    "grade_qa_suite",
    "grader_for_benchmark",
    "normalize_bbh_answer",
    "normalize_math_number",
]
