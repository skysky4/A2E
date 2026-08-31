"""Official DeepSearchQA grader (paper Appendix A style).

Single Answer: 1 iff the gold string is contained in the reply (case-insensitive).
Set Answer: item recall of gold items found in the reply.

The paper also describes a ``gemini-2.5-flash`` autorater. That judge is not
wired here; this module is the deterministic in-repo official stand-in.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

_SET_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\n)\s*", re.IGNORECASE)


def _final_answer(output: Mapping[str, Any] | None) -> str:
    raw = (output or {}).get("final_answer", "")
    if not raw:
        return ""
    text = str(raw).strip()
    if text.startswith("{") and "final_answer" in text:
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(obj, dict) and "final_answer" in obj:
            return str(obj["final_answer"]).strip()
    return text


def _deepsearch_items(text: str) -> list[str]:
    parts = [re.sub(r"\s+", " ", p).strip(" .;:") for p in _SET_SPLIT_RE.split(text or "")]
    return [p.lower() for p in parts if p]


def deepsearch_grader(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
) -> float:
    """Registry name: ``deepsearch_grader`` (alias: ``deepsearch_match``)."""
    answer = _final_answer(output)
    gold = str(((expected or {}).get("expected_outputs") or [""])[0] or "")
    if not answer or not gold:
        return 0.0
    state = (input or {}).get("initial_state") or {}
    answer_type = str(state.get("answer_type") or "Single Answer")
    if answer_type != "Set Answer":
        g = gold.strip().lower()
        a = answer.strip().lower()
        return float(g == a or g in a)
    golds = _deepsearch_items(gold)
    if not golds:
        return 0.0
    blob = answer.lower()
    pred = set(_deepsearch_items(answer))
    hits = sum(1 for item in golds if item in pred or item in blob)
    return hits / len(golds)
