"""GDPval in-run grader.

OpenAI's published GDPval leaderboard metric is a file-aware pairwise Elo
against human deliverables (human = 1000). That tournament is not executed
during an A2E cell.

The in-run official-cell grader is an LLM-as-judge of the agent deliverable
against the task rubric (``expected_outputs[0]``). Registry name: ``gdp_grader``
(alias: ``llm_judge`` when this dataset is selected).
"""

from __future__ import annotations

from typing import Any, Callable


def make_gdp_grader(llm: Any) -> Callable[..., Any]:
    """Build the named GDPval grader (same judge family as ``llm_judge``)."""
    from ageneval.task.runners.registry import make_llm_judge

    return make_llm_judge(llm, label="gdp_grader")
