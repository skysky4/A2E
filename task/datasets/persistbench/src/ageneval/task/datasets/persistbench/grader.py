"""Best-effort local PersistBench grading.

This is explicitly unofficial (``official=False``).  Upstream PersistBench rows
do not contain reference answers and require a judge; only rows that do carry a
local reference can be deterministically scored here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ageneval.task.core.grading import GraderSpec


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


def _references(expected: Any) -> list[str]:
    if isinstance(expected, Mapping):
        expected = expected.get("expected_outputs", expected.get("answer", ()))
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        values = expected
    else:
        values = (expected,)
    return [str(value).strip() for value in values if str(value or "").strip()]


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip(" \t\r\n.;").casefold()


def grade_persistbench(
    output: Any,
    expected: Any,
    input: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use a reference answer when present; otherwise report unsupported."""
    del input
    references = _references(expected)
    source = str((metadata or {}).get("source") or "PersistBench/PersistBench")
    if not references:
        return {
            "score": None,
            "passed": None,
            "label": "unsupported",
            "metrics": {},
            "metadata": {"status": "unsupported", "reference_available": False},
            "explanation": (
                "No reference answer is available; official PersistBench evaluation "
                "requires a judge and cannot be reproduced locally."
            ),
            "official": False,
            "source": source,
            "version": "unofficial-reference-v1",
        }

    predicted = _normalize(output)
    normalized_references = [_normalize(reference) for reference in references]
    matched = next(
        (
            reference
            for reference in normalized_references
            if predicted and reference and (predicted == reference or reference in predicted)
        ),
        None,
    )
    score = float(matched is not None)
    return {
        "score": score,
        "passed": bool(score),
        "metrics": {"reference_answer_match": score},
        "metadata": {
            "reference_available": True,
            "matched_reference": matched,
        },
        "explanation": "Unofficial best-effort match against an available reference answer.",
        "official": False,
        "source": source,
        "version": "unofficial-reference-v1",
    }


grade = grade_persistbench
GRADER = GraderSpec(
    id="persistbench_grader",
    grade=grade_persistbench,
    official=False,
    source="PersistBench/PersistBench",
    version="unofficial-reference-v1",
    aliases=("substring",),
)

__all__ = ["GRADER", "grade", "grade_persistbench"]
