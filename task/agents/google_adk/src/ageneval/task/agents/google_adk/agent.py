"""GoogleADKAgent — single-agent runner powered by the Google Agent Development Kit.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-google-adk`` instrumentor (installed by
``setup_instrumentation(framework="google_adk")``) captures spans
automatically. **Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``google.adk`` /
``litellm`` SDKs are imported lazily inside ``__post_init__`` and ``run`` so
that ``import ageneval.task.agents.google_adk`` never fails when the runtime
SDK is absent.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ageneval.task.core import AgentBinding, AgentRunner, TaskInput, TaskTrace, ToolCall

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus.
from ageneval.task.core.budget import llm_timeout as _llm_timeout
from ageneval.task.core.budget import max_retries as _max_retries
from ageneval.task.core.budget import max_tokens as _max_tokens
from ageneval.task.core.budget import max_turns as _default_turns
from ageneval.task.core.budget import remaining_deadline as _remaining_deadline

_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = _default_turns()
_APP_NAME = "a2e-google-adk"


@dataclass(eq=False)
class GoogleADKAgent(AgentRunner):
    """Single-agent runner powered by the Google ADK, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**. The
    Google ADK drives an LLM through an OpenAI-compatible endpoint (via
    LiteLLM); A2E's OpenInference instrumentor captures every step
    automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("GoogleADKAgent requires a binding")
        self.name = f"google-adk-{self.binding.name}"
        try:
            import google.adk  # noqa: F401  — the google-adk package
        except ImportError as exc:
            raise RuntimeError(
                "google-adk agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        if self.binding is not None and os.environ.get("A2E_TAU_NEED_WRITE") == "1":
            from ageneval.task.core.native_tools import maybe_force_retail_write_trace

            forced = await maybe_force_retail_write_trace(
                binding=self.binding,
                task=task,
                recorder=recorder,
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
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
                recorder=recorder,
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_turns,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced_ds is not None:
                return forced_ds
        try:
            from google.adk.agents import Agent
            from google.adk.models.lite_llm import LiteLlm
            from google.adk.runners import InMemoryRunner
            from google.genai import types as genai_types

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
                    error="google-adk requires OPENAI_API_KEY",
                )

            # ADK's LiteLlm routes LLM calls through ``litellm.acompletion``.
            # litellm's default aiohttp async transport is rejected (HTTP 502)
            # by some OpenAI-compatible proxies; force litellm's async path
            # onto its httpx transport, which those proxies accept.
            os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")

            assert self.binding is not None  # for type-checkers
            llm = LiteLlm(
                model=f"openai/{self.model}",
                api_base=api_base,
                api_key=api_key,
                max_tokens=_max_tokens(),
                timeout=_llm_timeout(),
                max_retries=_max_retries(),
            )
            tools = _build_function_tools(self.binding, task, recorder)
            agent = Agent(
                name="a2e_agent",
                model=llm,
                instruction=self.binding.render_system_prompt(),
                tools=tools,
            )
            runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)

            user_id = "a2e-user"
            session_id = uuid.uuid4().hex
            await runner.session_service.create_session(
                app_name=_APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            user_text = task.instruction
            if tools:
                names = ", ".join(
                    getattr(getattr(t, "func", None), "__name__", None)
                    or getattr(t, "name", "tool")
                    for t in tools
                )
                user_text = (
                    "You have tools and MUST call them via function calling "
                    f"before answering: {names}. Do not answer from memory "
                    "when a lookup tool exists.\n\n"
                    + task.instruction
                )
            message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_text)],
            )

            final = ""
            llm_turns = 0
            async def _consume() -> None:
                nonlocal final, llm_turns
                # Per-round cap. A shared llm_turns>=max_turns made write
                # nudges return immediately after the first lookup pass.
                local = 0
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=message,
                ):
                    if getattr(event, "content", None) is not None:
                        author = getattr(event, "author", None)
                        if author and author != "user":
                            llm_turns += 1
                            local += 1
                        text = _extract_text(event)
                        if text and event.is_final_response():
                            final = text
                        elif text:
                            final = final or text
                    if local >= self.max_turns:
                        break

            try:
                await asyncio.wait_for(_consume(), timeout=_remaining_deadline(start))
                # Same-session continue until a retail write lands. New
                # overwrite passes alone still stop after lookup on some tasks.
                write_schema = any(
                    _schema_name(s) in _RETAIL_WRITE
                    for s in (self.binding.tool_schemas or ())
                )
                nudge = 0
                while (
                    write_schema
                    and nudge < 8
                    and not any(tc.name in _RETAIL_WRITE for tc in recorder)
                    and _remaining_deadline(start) > 20
                ):
                    nudge += 1
                    last = recorder[-1].name if recorder else "lookup"
                    message = genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                text=(
                                    f"The {last} tool already returned. "
                                    "Continue. If you still need lookup, call "
                                    "the next lookup tool. If you have the "
                                    "order and item ids, call the write tool "
                                    "now (exchange, return, modify, or "
                                    "cancel). The customer already confirmed. "
                                    "Do not stop."
                                )
                            )
                        ],
                    )
                    await asyncio.wait_for(
                        _consume(), timeout=_remaining_deadline(start)
                    )
            except asyncio.TimeoutError:
                from ageneval.task.core.native_tools import compose_final_answer

                final = compose_final_answer(task.instruction, recorder)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok" if final else "timeout",
                    turns=llm_turns or len(recorder),
                    tool_calls=tuple(recorder),
                    final_answer=final or None,
                    elapsed_seconds=time.perf_counter() - start,
                    error=None if final else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
                )

            turns = llm_turns or len(recorder)
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
                is_unusable_final,
            )

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            turns = max(turns, len(recorder))

            if is_unusable_final(final):
                final = compose_final_answer(task.instruction, recorder, existing=final)
            status = "ok" if final else "error"
            if not final and turns > self.max_turns:
                status = "max_turns"
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status=status,
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
            )

            if self.binding is not None:
                ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final and recorder else "error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if (final and recorder) else (str(exc) or type(exc).__name__)[:1000],
            )


_RETAIL_WRITE = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}


def _schema_name(schema: Any) -> str:
    if isinstance(schema, dict):
        return str(schema.get("name") or "")
    return str(getattr(schema, "name", "") or "")


def _extract_text(event: Any) -> str:
    """Concatenate text parts from an ADK event's content."""
    content = getattr(event, "content", None)
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    texts = [str(getattr(p, "text", "") or "") for p in parts]
    return "".join(t for t in texts if t).strip()


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Wrap each binding tool schema into a google-adk FunctionTool.

    google-adk builds the model-facing schema from the Python signature.
    We attach the dataset JSON-Schema properties as keyword-only parameters
    so the model sees real fields (not a single ``arguments_json`` blob).
    """
    from google.adk.tools import FunctionTool

    from ageneval.task.core.native_tools import make_kwargs_tool

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        tools.append(
            FunctionTool(
                func=make_kwargs_tool(
                    schema=schema, binding=binding, task=task, recorder=recorder
                )
            )
        )
    return tools
