"""Deterministic DeepSearchQA Appendix-A-style answer grading.

The paper also describes a Gemini autorater. This local implementation covers
the deterministic single-answer containment and set-answer recall checks only,
so its provenance is explicitly marked unofficial.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ageneval.task.core.grading import GraderSpec

_SET_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\n)\s*", re.IGNORECASE)

GRADER_METADATA = {
    "id": "deepsearch_grader",
    "aliases": ("deepsearch_match",),
    "official": False,
    "source": "DeepSearchQA Appendix A deterministic stand-in",
    "version": "containment-recall-v1",
    "limitations": ("Gemini autorater is not included",),
}


def _final_answer(output: Mapping[str, Any] | None) -> str:
    raw = (output or {}).get("final_answer", "")
    if not raw:
        return ""
    text = str(raw).strip()
    if text.startswith("{") and "final_answer" in text:
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return text
        if isinstance(decoded, Mapping) and "final_answer" in decoded:
            return str(decoded["final_answer"]).strip()
    return text


def _deepsearch_items(text: str) -> list[str]:
    parts = [
        re.sub(r"\s+", " ", part).strip(" .;:")
        for part in _SET_SPLIT_RE.split(text or "")
    ]
    return [part.lower() for part in parts if part]


def deepsearch_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
) -> float:
    """Score single answers by containment and set answers by item recall."""
    answer = _final_answer(output)
    gold = str(((expected or {}).get("expected_outputs") or [""])[0] or "")
    if not answer or not gold:
        return 0.0
    state = (input or {}).get("initial_state") or {}
    answer_type = (
        str(state.get("answer_type") or "Single Answer")
        if isinstance(state, Mapping)
        else "Single Answer"
    )
    if answer_type != "Set Answer":
        normalized_gold = gold.strip().lower()
        normalized_answer = answer.strip().lower()
        return float(
            normalized_gold == normalized_answer
            or normalized_gold in normalized_answer
        )
    gold_items = _deepsearch_items(gold)
    if not gold_items:
        return 0.0
    answer_blob = answer.lower()
    predicted_items = set(_deepsearch_items(answer))
    hits = sum(
        1
        for item in gold_items
        if item in predicted_items or item in answer_blob
    )
    return hits / len(gold_items)


grade = deepsearch_grader
GRADER = GraderSpec(
    id=GRADER_METADATA["id"],
    grade=deepsearch_grader,
    official=GRADER_METADATA["official"],
    source=GRADER_METADATA["source"],
    version=GRADER_METADATA["version"],
    aliases=GRADER_METADATA["aliases"],
    metadata={"limitations": GRADER_METADATA["limitations"]},
)

__all__ = ["GRADER", "GRADER_METADATA", "deepsearch_grader", "grade"]
