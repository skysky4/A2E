"""DeepSearchQA output cleanup around an unchanged agent harness.

The wrapper does not inject tools, fetch pages, or synthesize an answer. It
keeps the harness trajectory and only unwraps usable final-answer envelopes.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.result import TaskTrace, ToolCall


def _unwrap_final_text(text: str) -> str:
    value = (text or "").strip().strip("*").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    if '"final_answer"' not in value:
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        match = re.search(r'"final_answer"\s*:\s*"(.*?)"', value, re.DOTALL)
        return match.group(1).replace("\\n", "\n").strip() if match else value
    if isinstance(decoded, Mapping):
        inner = str(decoded.get("final_answer") or "").strip()
        if inner:
            return inner
    return value


def _is_unusable_final(text: str) -> bool:
    value = _unwrap_final_text(text)
    if not value:
        return True
    lowered = value.lower()
    if re.fullmatch(
        r"(?:user|assistant|system|human|tool)\s*:\s*(?:none|null)?",
        lowered,
    ):
        return True
    if re.search(r"\bto=(?:web_search|open_url)\b", lowered):
        return True
    if (
        "code:" in lowered
        and '"query"' in value
        and "web_search" in lowered
    ):
        return True
    return bool(
        re.search(r"\baction\s*input\s*:", lowered)
        and any(name in lowered for name in ("web_search", "open_url"))
    )


def _clean_final_answer(text: str) -> str:
    value = _unwrap_final_text(text)
    return "" if _is_unusable_final(value) else value


def wrap_dsqa_official_session(agent: AgentRunner) -> AgentRunner:
    if isinstance(agent, DeepSearchOfficialSession):
        return agent
    return DeepSearchOfficialSession(inner=agent)


class DeepSearchOfficialSession(AgentRunner):
    """Pass through harness tools and retain only a usable harness final."""

    def __init__(self, inner: AgentRunner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "dsqa-session")
        self.binding = getattr(inner, "binding", None)

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        trace = await self.inner.run(task)
        tools = [
            tool_call
            for tool_call in (trace.tool_calls or ())
            if isinstance(tool_call, ToolCall)
        ]
        raw: dict[str, Any] = dict(trace.raw or {})
        inner_final = str(raw.get("inner_final") or trace.final_answer or "")
        raw["inner_final"] = inner_final
        final = _clean_final_answer(trace.final_answer or "") or _clean_final_answer(
            inner_final
        )
        status = trace.status
        if final:
            status = "ok"
        elif status == "ok":
            status = "error"
        error = trace.error
        if (
            final
            and error
            and "upstream service temporarily unavailable" in error.lower()
        ):
            error = None
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,  # type: ignore[arg-type]
            turns=max(int(trace.turns or 0), len(tools)),
            tool_calls=tuple(tools),
            final_answer=final or None,
            elapsed_seconds=time.perf_counter() - start,
            error=error,
            raw=raw,
        )


__all__ = ["DeepSearchOfficialSession", "wrap_dsqa_official_session"]
