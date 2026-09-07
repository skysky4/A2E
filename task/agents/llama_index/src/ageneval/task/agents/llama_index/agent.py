"""LlamaIndexAgent — single-agent runner powered by LlamaIndex.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-llama-index`` instrumentor (installed by
``setup_instrumentation(framework="llama_index")``) captures spans
automatically. **Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``llama_index`` /
``openai`` SDKs are imported lazily inside ``__post_init__`` and ``run`` so that
``import ageneval.task.agents.llama_index`` never fails when the runtime SDK is
absent.
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
    llm_timeout,
    make_kwargs_tool,
    max_tokens as _budget_tokens,
    needs_followup_final,
    parameters_block,
    pydantic_args_model,
)
from ageneval.task.core.budget import max_retries
from ageneval.task.core.openai_compat import install_openai_compat

# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"
_MAX_TURNS = 8


@dataclass(eq=False)
class LlamaIndexAgent(AgentRunner):
    """Single-agent runner powered by LlamaIndex, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**.
    LlamaIndex's ``FunctionAgent`` drives an LLM through an OpenAI-compatible
    endpoint (``OpenAILike``); A2E's OpenInference instrumentor captures every
    step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("LlamaIndexAgent requires a binding")
        self.name = f"llama-index-{self.binding.name}"
        try:
            import llama_index.core  # noqa: F401  — the llama-index-core package
        except ImportError as exc:
            raise RuntimeError(
                "llama-index agent requires its runtime SDK. Install with:\n"
                "  uv sync at the A2E workspace root"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        llm = None
        try:
            from llama_index.core.agent.workflow import FunctionAgent
            from llama_index.core.tools import FunctionTool
            from llama_index.llms.openai_like import OpenAILike

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
                    error="llama-index requires OPENAI_API_KEY",
                )

            assert self.binding is not None  # for type-checkers
            install_openai_compat()
            llm = OpenAILike(
                model=self.model,
                api_base=api_base,
                api_key=api_key,
                is_chat_model=True,
                is_function_calling_model=True,
                temperature=1.0,
                timeout=llm_timeout(),
                max_retries=max_retries(),
                max_tokens=_budget_tokens(),
                context_window=128000,
            )
            tools = _build_function_tools(self.binding, task, recorder, FunctionTool)
            agent = FunctionAgent(
                tools=tools,
                llm=llm,
                system_prompt=self.binding.render_system_prompt(),
            )
            result = await agent.run(
                task.instruction,
                max_iterations=self.max_turns,
            )
            sdk_final, final = _extract_sdk_and_final(result)
            if needs_followup_final(final, recorder):
                follow = await agent.run(
                    followup_user_prompt(task.instruction, recorder),
                    max_iterations=2,
                )
                sdk2, fin2 = _extract_sdk_and_final(follow)
                sdk_final = fin2 or sdk2 or sdk_final
                final = fin2
            turns = len(recorder) or (1 if final else 0)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final or recorder or sdk_final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or sdk_final or None,
                elapsed_seconds=time.perf_counter() - start,
                raw={"inner_final": sdk_final},
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            error = (str(exc) or type(exc).__name__)[:1000]
            reached_limit = "Max iterations of" in error
            sdk_final, final = "", ""
            if needs_followup_final("", recorder) and llm is not None:
                try:
                    follow_agent = FunctionAgent(
                        tools=[],
                        llm=llm,
                        system_prompt=self.binding.render_system_prompt(),
                    )
                    follow = await follow_agent.run(
                        followup_user_prompt(task.instruction, recorder),
                        max_iterations=2,
                    )
                    sdk_final, final = _extract_sdk_and_final(follow)
                except Exception:  # noqa: BLE001
                    pass
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else ("max_turns" if reached_limit else "error"),
                turns=self.max_turns if reached_limit else len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or sdk_final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else error,
                raw={"inner_final": sdk_final},
            )


def _message_text(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    content = getattr(msg, "content", None)
    if content:
        return str(content)
    return ""


def _extract_sdk_and_final(result: Any) -> tuple[str, str]:
    """Pull a real answer out of FunctionAgent output; keep the SDK string.

    ``str(AgentOutput)`` is often ``user: None``. Prefer ``.response`` /
    chat-history assistant text, then clean through ``clean_final_answer``.
    """
    if result is None:
        return "", ""
    candidates: list[str] = []
    if isinstance(result, str):
        candidates.append(result)
    else:
        for attr in ("response", "output", "generated", "answer"):
            text = _message_text(getattr(result, attr, None))
            if text:
                candidates.append(text)
        for attr in ("chat_history", "messages"):
            hist = getattr(result, attr, None)
            if not hist:
                continue
            try:
                items = list(hist)
            except TypeError:
                continue
            for msg in reversed(items):
                role = str(getattr(msg, "role", "") or "").lower()
                text = _message_text(msg)
                if text and role in {"assistant", "ai", "model"}:
                    candidates.append(text)
                    break
        dumped = str(result).strip()
        if dumped:
            candidates.append(dumped)
    sdk_final = next((c for c in candidates if c.strip()), "")
    for raw in candidates:
        cleaned = clean_final_answer(raw)
        if cleaned:
            return sdk_final, cleaned
    return sdk_final, ""


def _extract_final(result: Any) -> str:
    """Best-effort final-answer extraction from a FunctionAgent run result."""
    return _extract_sdk_and_final(result)[1]


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
    function_tool_cls: Callable[..., Any],
) -> list[Any]:
    """Wrap each binding tool schema into a LlamaIndex ``FunctionTool``.

    LlamaIndex still runs ``FunctionAgent``. We only attach the benchmark's
    real parameter schema so official tools are callable natively.
    """
    tools: list[Any] = []
    for schema in binding.tool_schemas:
        fn = schema["function"]
        name = fn["name"]
        description = fn.get("description", "") or f"Invoke the {name} tool."
        tools.append(
            function_tool_cls.from_defaults(
                fn=make_kwargs_tool(
                    schema=schema, binding=binding, task=task, recorder=recorder
                ),
                name=name,
                description=description,
                fn_schema=pydantic_args_model(name, parameters_block(schema)),
            )
        )
    return tools
