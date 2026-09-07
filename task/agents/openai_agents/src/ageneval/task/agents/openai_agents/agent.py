"""OpenAIAgentsAgent — single-agent runner powered by the OpenAI Agents SDK.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-openai-agents`` instrumentor (installed by
``setup_instrumentation(framework="openai_agents")``) captures spans
automatically. **Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``agents`` / ``openai``
SDKs are imported lazily inside ``__post_init__`` and ``run`` so that
``import ageneval.task.agents.openai_agents`` never fails when the runtime SDK
is absent.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ageneval.task.core import (
    AgentBinding,
    AgentRunner,
    TaskInput,
    TaskTrace,
    ToolCall,
    clean_final_answer,
    followup_user_prompt,
    make_kwargs_tool,
    needs_followup_final,
)

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = 8


@dataclass(eq=False)
class OpenAIAgentsAgent(AgentRunner):
    """Single-agent runner powered by the OpenAI Agents SDK, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**. The
    OpenAI Agents SDK drives an LLM through an OpenAI-compatible endpoint; A2E's
    OpenInference instrumentor captures every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("OpenAIAgentsAgent requires a binding")
        self.name = f"openai-agents-{self.binding.name}"
        try:
            import agents  # noqa: F401  — the openai-agents package
        except ImportError as exc:
            raise RuntimeError(
                "openai-agents agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        try:
            from agents import Agent, Runner, function_tool
            from agents.exceptions import MaxTurnsExceeded
            from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
            from openai import AsyncOpenAI

            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            api_base = self.api_base or os.environ.get("OPENAI_API_BASE")
            if not api_key:
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="error",
                    turns=0,
                    tool_calls=(),
                    elapsed_seconds=time.perf_counter() - start,
                    error="openai-agents requires OPENAI_API_KEY",
                )
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)

            assert self.binding is not None  # for type-checkers
            tools = _build_function_tools(self.binding, task, recorder, function_tool)
            agent = Agent(
                name=self.name,
                instructions=self.binding.render_system_prompt(),
                model=OpenAIChatCompletionsModel(model=self.model, openai_client=client),
                tools=tools,
            )
            result = await Runner.run(agent, task.instruction, max_turns=self.max_turns)
            raw_final = str(getattr(result, "final_output", "") or "")
            final = clean_final_answer(raw_final)
            if needs_followup_final(final, recorder):
                raw_final, final = await _followup_no_tools(
                    Agent,
                    Runner,
                    OpenAIChatCompletionsModel,
                    client,
                    task,
                    recorder,
                )
            turns = len(getattr(result, "raw_responses", []) or []) or len(recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final or raw_final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or raw_final or None,
                elapsed_seconds=time.perf_counter() - start,
            )
        except MaxTurnsExceeded:
            raw_final, final = "", ""
            if recorder:
                try:
                    raw_final, final = await _followup_no_tools(
                        Agent,
                        Runner,
                        OpenAIChatCompletionsModel,
                        client,
                        task,
                        recorder,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final or raw_final else "max_turns",
                turns=self.max_turns,
                tool_calls=tuple(recorder),
                final_answer=final or raw_final or None,
                elapsed_seconds=time.perf_counter() - start,
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                tool_calls=tuple(recorder),
                elapsed_seconds=time.perf_counter() - start,
                error=(str(exc) or type(exc).__name__)[:1000],
            )


async def _followup_no_tools(
    agent_cls: Any,
    runner_cls: Any,
    model_cls: Any,
    client: Any,
    task: TaskInput,
    recorder: list[ToolCall],
) -> tuple[str, str]:
    """One extra Runner.run without tools after the official loop ends."""
    follow_agent = agent_cls(
        name="a2e-followup",
        instructions="Write the required final output from the tool results. Do not call tools.",
        model=model_cls(
            model=os.environ.get("A2E_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            openai_client=client,
        ),
        tools=[],
    )
    follow = await runner_cls.run(
        follow_agent,
        followup_user_prompt(task.instruction, recorder),
        max_turns=1,
    )
    raw = str(getattr(follow, "final_output", "") or "")
    return raw, clean_final_answer(raw)


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
    function_tool: Callable[..., Any],
) -> list[Any]:
    """Wrap each binding tool schema into an openai-agents function_tool.

    The SDK still drives the loop. We only publish the benchmark's real
    parameter schema so the model can call official tools natively.
    """
    tools = []
    for schema in binding.tool_schemas:
        fn = schema["function"]
        name = fn["name"]
        description = fn.get("description", "") or f"Invoke the {name} tool."
        wrapped = make_kwargs_tool(
            schema=schema, binding=binding, task=task, recorder=recorder
        )
        tools.append(
            function_tool(
                name_override=name,
                description_override=description,
            )(wrapped)
        )
    return tools
