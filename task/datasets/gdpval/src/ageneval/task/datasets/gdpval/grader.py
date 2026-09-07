"""GDPval in-run grader.

OpenAI's published GDPval leaderboard metric is a file-aware pairwise Elo
against human deliverables (human = 1000). That tournament is not executed
during an A2E cell.

The in-run official-cell grader is an LLM-as-judge of the agent deliverable
against the task rubric (``expected_outputs[0]``). It must see the files
submitted via ``finish``, not only the chat text. Registry name: ``gdp_grader``
(alias: ``llm_judge`` when this dataset is selected).
"""

from __future__ import annotations

import re
from typing import Any, Callable


def make_gdp_grader(llm: Any) -> Callable[..., Any]:
    """Build the named GDPval grader (file-aware rubric judge, not Elo)."""

    def fn(output: dict, expected: dict, input: dict) -> Any:
        submitted = list((output or {}).get("gdp_submitted") or [])
        meta = list((output or {}).get("gdp_submitted_meta") or [])
        n_att = (output or {}).get("gdp_n_attachments")
        finish = str((output or {}).get("gdp_finish_summary") or "")
        answer = (output or {}).get("final_answer") or finish or "(no text)"
        file_lines: list[str] = []
        for row in meta:
            if isinstance(row, dict):
                name = row.get("name") or row.get("path") or "?"
                nbytes = row.get("bytes")
                extra = f" ({nbytes} bytes)" if nbytes is not None else ""
                file_lines.append(f"- {name}{extra}")
            elif row:
                file_lines.append(f"- {row}")
        if not file_lines:
            file_lines = [f"- {n}" for n in submitted] or [
                "(no files submitted via finish)"
            ]
        prompt = (
            "You are the official in-run GDPval-AA rubric judge. "
            "OpenAI pairwise Elo (human=1000) is off-run and must not be computed here.\n"
            "The agent works in a computer-use sandbox. Deliverables are files, "
            "not only chat text.\n"
            "If real files were submitted, judge those filenames plus the "
            "text/summary against the rubric. Do not mark incorrect solely "
            "because a spreadsheet or document was not pasted into the chat.\n"
            "Return EXACTLY one line: SCORE=<0 or 1>; EXPLANATION=<one sentence>\n\n"
            f"User instruction: {input.get('instruction', '')}\n"
            f"Reference attachments loaded: {n_att}\n"
            f"Submitted files:\n{chr(10).join(file_lines)}\n"
            f"Finish summary: {finish or '(none)'}\n"
            f"Agent text: {answer}\n"
            f"Expected rubric: {((expected or {}).get('expected_outputs') or [''])[0]}\n"
        )
        try:
            text = llm.generate_text(prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            return {"score": 0.0, "label": "error", "explanation": str(exc)[:200]}
        m_score = re.search(r"SCORE\s*=\s*([01](?:\.\d+)?)", text or "")
        m_expl = re.search(r"EXPLANATION\s*=\s*(.+?)(?:\n|$)", text or "")
        score = float(m_score.group(1)) if m_score else 0.0
        return {
            "score": score,
            "label": "correct" if score >= 0.5 else "incorrect",
            "explanation": (m_expl.group(1) if m_expl else (text or ""))[:500],
        }

    fn.__name__ = "gdp_grader"
    fn.__qualname__ = "gdp_grader"
    return fn
