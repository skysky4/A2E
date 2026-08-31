"""DeepSearchQA session around an unchanged harness.

Official answers must come from the harness run: the SDK calls
``web_search`` / ``open_url`` and writes the final. This wrapper only
unwraps JSON / drops ReAct leaks and ``user: None``. It does **not**
inject tools, open extra pages, or compose an answer the agent never wrote.
"""

from __future__ import annotations

import time

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.native_tools import clean_final_answer, is_unusable_final
from ageneval.task.core.result import TaskTrace, ToolCall


def wrap_dsqa_official_session(agent: AgentRunner) -> AgentRunner:
    if isinstance(agent, DeepSearchOfficialSession):
        return agent
    return DeepSearchOfficialSession(inner=agent)


class DeepSearchOfficialSession(AgentRunner):
    """Pass through harness tools; keep only a usable harness final."""

    def __init__(self, inner: AgentRunner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "dsqa-session")
        self.binding = getattr(inner, "binding", None)

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        trace = await self.inner.run(task)
        tools = [tc for tc in (trace.tool_calls or ()) if isinstance(tc, ToolCall)]
        raw = dict(trace.raw or {})
        inner_final = str(raw.get("inner_final") or trace.final_answer or "")
        raw["inner_final"] = inner_final
        final = clean_final_answer(trace.final_answer or "") or clean_final_answer(
            inner_final
        )
        status = trace.status
        if final and not is_unusable_final(final):
            status = "ok"
        elif status == "ok" and not final:
            status = "error"
        error = trace.error
        if final and error and "upstream service temporarily unavailable" in error.lower():
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
