"""GDPval in-run rubric judge.

The published GDPval leaderboard uses file-aware pairwise Elo against human
deliverables. A2E's online cell instead asks its configured grading model to
score the generated deliverable against ``expected_outputs[0]``. This narrower
score is explicitly marked unofficial.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ageneval.task.core.grading import GraderSpec

GRADER_METADATA = {
    "id": "gdp_grader",
    "aliases": ("llm_judge",),
    "official": False,
    "source": "openai/gdpval rubric judge",
    "version": "in-run-binary-v1",
    "limitations": ("file-aware pairwise Elo is not executed",),
}

_PROMPT = (
    "You are an evaluator. Decide whether the agent's deliverable satisfies "
    "the user's request and grading rubric.\n"
    "Return EXACTLY one line: SCORE=<0 or 1>; EXPLANATION=<one sentence>\n\n"
    "User instruction: {instruction}\n"
    "Agent deliverable: {answer}\n"
    "Grading rubric: {expected}\n"
)


def _provenance() -> dict[str, Any]:
    return {
        "official": GRADER_METADATA["official"],
        "source": GRADER_METADATA["source"],
        "version": GRADER_METADATA["version"],
        "metadata": {"limitations": GRADER_METADATA["limitations"]},
    }


def make_gdp_grader(llm: Any) -> Callable[..., Any]:
    """Build the dataset-owned GDPval rubric grader for a model runtime."""

    def gdp_grader(output: dict, expected: dict, input: dict) -> dict[str, Any]:
        expected_outputs = (expected or {}).get("expected_outputs") or [""]
        prompt = _PROMPT.format(
            instruction=(input or {}).get("instruction", ""),
            answer=(output or {}).get("final_answer", "") or "(no deliverable)",
            expected=expected_outputs[0],
        )
        try:
            text = llm.generate_text(prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            return {
                "score": 0.0,
                "passed": False,
                "label": "error",
                "explanation": str(exc)[:200],
                **_provenance(),
            }
        score_match = re.search(r"SCORE\s*=\s*([01](?:\.\d+)?)", text or "")
        explanation_match = re.search(
            r"EXPLANATION\s*=\s*(.+?)(?:\n|$)",
            text or "",
        )
        score = float(score_match.group(1)) if score_match else 0.0
        return {
            "score": score,
            "passed": score >= 0.5,
            "label": "correct" if score >= 0.5 else "incorrect",
            "explanation": (
                explanation_match.group(1)
                if explanation_match
                else (text or "")
            )[:500],
            **_provenance(),
        }

    gdp_grader.__name__ = "gdp_grader"
    gdp_grader.__qualname__ = "gdp_grader"
    return gdp_grader


GRADER = GraderSpec(
    id=GRADER_METADATA["id"],
    factory=make_gdp_grader,
    official=GRADER_METADATA["official"],
    source=GRADER_METADATA["source"],
    version=GRADER_METADATA["version"],
    aliases=GRADER_METADATA["aliases"],
    metadata={"limitations": GRADER_METADATA["limitations"]},
)

__all__ = ["GRADER", "GRADER_METADATA", "make_gdp_grader"]
