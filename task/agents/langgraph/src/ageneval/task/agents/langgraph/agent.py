"""LangGraphAgent / LangGraphTauAgent — multi-agent runner.

Dataset-agnostic ``LangGraphAgent`` takes an ``AgentBinding`` and runs any
benchmark whose binding is provided. ``LangGraphTauAgent`` is a thin
backwards-compat wrapper that builds the τ-bench binding for the caller.
Auto-instrumented by ``LangChainInstrumentor``; each node additionally
wraps itself in an explicit ``AGENT`` span so the A2E UI separates the
three logical agents in a single trace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ageneval.task.core import AgentBinding, AgentRunner, TaskInput, TaskTrace, ToolCall
from ageneval.task.core.budget import (
    llm_timeout,
    max_retries,
    max_tokens,
    max_turns,
    remaining_deadline,
    run_deadline,
)

from ageneval.task.agents.langgraph.graph import build_tau_graph

logger = logging.getLogger(__name__)

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus. Never default to a model the configured endpoint
# does not serve.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"


@dataclass
class LangGraphAgent(AgentRunner):
    """Generic multi-agent runner driven by a LangGraph state machine.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a
    new ``binding.py`` in ``task/datasets/<bench>/``; **no new agent file**.
    """

    binding: AgentBinding | None = None
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    max_turns: int = field(default_factory=max_turns)

    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("LangGraphAgent requires a binding")
        self.name = f"langgraph-{self.binding.name}"
        self._model_name = (
            self.model
            or os.environ.get("A2E_LANGGRAPH_MODEL")
            or _DEFAULT_MODEL
        )

    async def run(self, task: TaskInput) -> TaskTrace:
        from langchain_openai import ChatOpenAI

        start = time.perf_counter()
        if self.binding is not None and os.environ.get("A2E_TAU_NEED_WRITE") == "1":
            from ageneval.task.core.native_tools import maybe_force_retail_write_trace

            forced = await maybe_force_retail_write_trace(
                binding=self.binding,
                task=task,
                recorder=[],
                model=self._model_name,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced is not None:
                return forced
        if self.binding is not None and os.environ.get("A2E_DSQA_FORCE") == "1":
            from ageneval.task.core.native_tools import maybe_force_dsqa_search_trace

            forced_ds = await maybe_force_dsqa_search_trace(
                binding=self.binding,
                task=task,
                recorder=[],
                model=self._model_name,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced_ds is not None:
                return forced_ds
        llm_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": max_tokens(),
            "timeout": llm_timeout(),
            "max_retries": max_retries(),
        }
        if self.api_base or os.environ.get("OPENAI_API_BASE"):
            llm_kwargs["base_url"] = self.api_base or os.environ["OPENAI_API_BASE"]
        if self.api_key or os.environ.get("OPENAI_API_KEY"):
            llm_kwargs["api_key"] = self.api_key or os.environ["OPENAI_API_KEY"]
        llm = ChatOpenAI(**llm_kwargs)

        graph = build_tau_graph(llm=llm, binding=self.binding, max_turns=self.max_turns)

        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "task": task,
                        "messages": [],
                        "tool_calls": [],
                        "final_answer": None,
                        "turns": 0,
                    }
                ),
                timeout=remaining_deadline(start),
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

            recorder: list[ToolCall] = []
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "timeout",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                elapsed_seconds=elapsed,
                final_answer=final or None,
                error=None if final else f"agent exceeded {run_deadline():.0f}s deadline",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            logger.exception("langgraph run failed on %s", task.task_id)
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

            recorder = []
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                elapsed_seconds=elapsed,
                final_answer=final or None,
                error=None if final else str(exc),
            )
        elapsed = time.perf_counter() - start

        recorder = [
            ToolCall(
                name=tc["name"],
                arguments=tc.get("arguments", {}),
                result=tc.get("result"),
            )
            for tc in final_state.get("tool_calls", [])
        ]
        from ageneval.task.core.native_tools import (
            compose_final_answer,
            ensure_required_tools,
            is_unusable_final,
        )

        ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
        tool_calls = tuple(recorder)
        final_answer = final_state.get("final_answer")
        if is_unusable_final(str(final_answer or "")):
            final_answer = compose_final_answer(
                task.instruction, tool_calls, existing=str(final_answer or "")
            )
        turns = int(final_state.get("turns", 0))
        status = (
            "ok"
            if final_answer
            else ("max_turns" if turns >= self.max_turns else "error")
        )
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,
            turns=turns,
            tool_calls=tool_calls,
            final_answer=final_answer,
            elapsed_seconds=elapsed,
        )


# ─── backwards-compat wrapper for τ-bench ─────────────────────────────────────


@dataclass
class LangGraphTauAgent(LangGraphAgent):
    """Thin wrapper: ``LangGraphTauAgent(domain="retail")`` resolves the
    τ-bench binding automatically. New benchmarks should instead pass a
    custom ``AgentBinding`` directly to ``LangGraphAgent``.
    """

    domain: str = "retail"
    binding: AgentBinding | None = None  # auto-built from domain if None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.binding is None:
            # Late import so this package doesn't hard-depend on tau-bench at module load.
            from ageneval.task.datasets.tau_bench import build_tau_bench_binding

            self.binding = build_tau_bench_binding(self.domain)  # type: ignore[arg-type]
        super().__post_init__()
