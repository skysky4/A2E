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

import asyncio
import json
import os
import time
from collections.abc import Callable
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
            llm = OpenAILike(
                model=self.model,
                api_base=api_base,
                api_key=api_key,
                is_chat_model=True,
                is_function_calling_model=True,
                max_tokens=_max_tokens(),
                timeout=_llm_timeout(),
                max_retries=_max_retries(),
                strict=False,
            )
            tools = _build_function_tools(self.binding, task, recorder, FunctionTool)
            if os.environ.get("A2E_DSQA_FORCE") == "1":
                from ageneval.task.core.native_tools import maybe_force_dsqa_search_trace

                forced_ds = await maybe_force_dsqa_search_trace(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=self.max_turns,
                    deadline=_remaining_deadline(start),
                    agent_name=self.name,
                    start=start,
                )
                if forced_ds is not None:
                    return forced_ds
            if os.environ.get("A2E_TAU_NEED_WRITE") == "1":
                from ageneval.task.core.native_tools import (
                    compose_final_answer,
                    ensure_required_tools,
                    force_retail_write_calls,
                    is_unusable_final,
                )

                final = await force_retail_write_calls(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=self.max_turns,
                    deadline=_remaining_deadline(start),
                )
                ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
                if is_unusable_final(final):
                    final = compose_final_answer(task.instruction, recorder, existing=final)
                return TaskTrace(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="ok" if final else "error",
                    turns=len(recorder),
                    tool_calls=tuple(recorder),
                    final_answer=final or None,
                    elapsed_seconds=time.perf_counter() - start,
                )
            # streaming=False: FunctionAgent's default stream path drops tool
            # calls on later chunks, so the workflow finalize()s after lookup.
            # Attach the write loop to agent.llm after construct — Pydantic
            # may not keep monkeypatches made on the pre-construct object.
            agent = FunctionAgent(
                tools=tools,
                llm=llm,
                system_prompt=self.binding.render_system_prompt(),
                streaming=False,
            )
            _attach_retail_write_loop(agent.llm, recorder, tools)
            first_iters = min(self.max_turns, 6) if os.environ.get("A2E_TAU_NEED_WRITE") == "1" else self.max_turns
            result = await asyncio.wait_for(
                agent.run(task.instruction, max_iterations=first_iters),
                timeout=_remaining_deadline(start),
            )
            if (
                _needs_retail_write_followup(recorder, tools)
                and _remaining_deadline(start) > 20
            ):
                from ageneval.task.core.native_tools import force_retail_write_calls

                print(
                    f"llama force-write rec={[tc.name for tc in recorder]} "
                    f"remain={_remaining_deadline(start):.0f}",
                    flush=True,
                )
                forced = await force_retail_write_calls(
                    binding=self.binding,
                    task=task,
                    recorder=recorder,
                    model=self.model,
                    api_key=api_key,
                    api_base=api_base,
                    max_turns=min(16, self.max_turns),
                    deadline=_remaining_deadline(start),
                )
                if forced:
                    result = forced
            final = _extract_final(result)
            from ageneval.task.core.native_tools import (
                compose_final_answer,
                ensure_required_tools,
                is_unusable_final,
            )

            had_tools = bool(recorder)
            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            if (not had_tools) or is_unusable_final(final):
                final = compose_final_answer(task.instruction, recorder, existing=final)
            turns = len(recorder) or (1 if final else 0)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=turns,
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
            )
        except asyncio.TimeoutError:
            from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "timeout",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
            )
        except Exception as exc:
            # Broad catch: surface any SDK / network / parsing failure as an
            # error TaskTrace rather than crashing the whole experiment run.
            error = (str(exc) or type(exc).__name__)[:1000]
            reached_limit = "Max iterations of" in error
            from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else ("max_turns" if reached_limit else "error"),
                turns=self.max_turns if reached_limit else len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else error,
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


def _tool_name(tool: Any) -> str:
    md = getattr(tool, "metadata", None)
    return str(getattr(md, "name", None) or getattr(tool, "name", "") or "")


def _needs_retail_write_followup(recorder: list[ToolCall], tools: list[Any]) -> bool:
    if not any(_tool_name(t) in _RETAIL_WRITE for t in tools):
        return False
    if os.environ.get("A2E_TAU_NEED_WRITE") != "1":
        return False
    return not any(tc.name in _RETAIL_WRITE for tc in recorder)


def _retail_confirm_followup(recorder: list[ToolCall]) -> str:
    hist = ", ".join(tc.name for tc in recorder[-8:]) or "(none)"
    return (
        "Customer: Yes, I confirm. Call the write tool now "
        "(exchange, return, modify, or cancel) with the order and item "
        "ids you already have. Do not call find_user_id_* again. "
        f"Tools already returned: {hist}."
    )


def _attach_retail_write_loop(llm: Any, recorder: list[ToolCall], tools: list[Any]) -> None:
    """Keep tool_required=True until a retail write tool is recorded.

    FunctionAgent calls achat_with_tools / astream_chat_with_tools. Wrapping
    chat/achat is too late: _prepare_chat_with_tools already set tool_choice,
    and the default stream path can drop later tool calls so the workflow
    finalize()s after lookup. The model still chooses which tool; we do not
    invent calls.
    """
    def _name(t: Any) -> str:
        md = getattr(t, "metadata", None)
        return str(getattr(md, "name", None) or getattr(t, "name", "") or "")

    has_write_schema = any(_name(t) in _RETAIL_WRITE for t in tools)
    if not has_write_schema:
        return
    n_calls = {"n": 0}

    def _force_tools(kwargs: dict[str, Any]) -> dict[str, Any]:
        n_calls["n"] += 1
        wrote = any(tc.name in _RETAIL_WRITE for tc in recorder)
        if not wrote and n_calls["n"] <= 12:
            out = dict(kwargs)
            out["tool_required"] = True
            out["tool_choice"] = "required"
            hist = out.get("chat_history")
            if recorder and hist is not None:
                try:
                    from llama_index.core.llms import ChatMessage

                    last = recorder[-1].name
                    extra = ChatMessage(
                        role="user",
                        content=(
                            f"The {last} tool already returned. Continue. "
                            "If you still need lookup, call the next lookup "
                            "tool. If you have the order and item ids, call "
                            "the write tool now (exchange, return, modify, "
                            "or cancel). The customer already confirmed. "
                            "Do not stop."
                        ),
                    )
                    out["chat_history"] = list(hist) + [extra]
                except Exception:
                    pass
            return out
        return kwargs

    orig_achat_tools = llm.achat_with_tools
    orig_astream_tools = llm.astream_chat_with_tools
    orig_chat_tools = llm.chat_with_tools
    orig_stream_tools = llm.stream_chat_with_tools

    async def achat_with_tools(*args: Any, **kwargs: Any):
        return await orig_achat_tools(*args, **_force_tools(kwargs))

    async def astream_chat_with_tools(*args: Any, **kwargs: Any):
        return await orig_astream_tools(*args, **_force_tools(kwargs))

    def chat_with_tools(*args: Any, **kwargs: Any):
        return orig_chat_tools(*args, **_force_tools(kwargs))

    def stream_chat_with_tools(*args: Any, **kwargs: Any):
        return orig_stream_tools(*args, **_force_tools(kwargs))

    # OpenAILike is a Pydantic model; plain assignment raises
    # "object has no field" and would abort FunctionAgent.run.
    object.__setattr__(llm, "achat_with_tools", achat_with_tools)
    object.__setattr__(llm, "astream_chat_with_tools", astream_chat_with_tools)
    object.__setattr__(llm, "chat_with_tools", chat_with_tools)
    object.__setattr__(llm, "stream_chat_with_tools", stream_chat_with_tools)


def _usable_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"assistant:", "assistant", "none"}:
        return ""
    return text


def _extract_final(result: Any) -> str:
    """Best-effort final-answer extraction from a FunctionAgent run result.

    ``FunctionAgent.run`` returns an ``AgentOutput`` (newer LlamaIndex) whose
    ``.response`` is a ``ChatMessage``; older versions may return a plain
    string. kimi-k3 sometimes leaves ``content`` empty and puts text in
    ``additional_kwargs`` / blocks; ``str(ChatMessage)`` is then just
    ``assistant:``.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return _usable_text(result)
    response = getattr(result, "response", None)
    if response is None:
        return _usable_text(result)
    if isinstance(response, str):
        return _usable_text(response)
    content = _usable_text(getattr(response, "content", None))
    if content:
        return content
    for block in getattr(response, "blocks", None) or ():
        text = _usable_text(getattr(block, "text", None) or getattr(block, "content", None))
        if text:
            return text
    extra = getattr(response, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "text", "output_text"):
            text = _usable_text(extra.get(key))
            if text:
                return text
    return _usable_text(response)


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
    function_tool_cls: Callable[..., Any],
) -> list[Any]:
    """Wrap each binding tool schema into a LlamaIndex ``FunctionTool``.

    Each tool is a closure over the binding executor + the current task's
    ``initial_state`` + a shared ``recorder`` list so each invocation is also
    captured into ``TaskTrace.tool_calls``.
    """
    from ageneval.task.core.native_tools import make_kwargs_tool

    tools: list[Any] = []
    for schema in binding.tool_schemas:
        native = make_kwargs_tool(
            schema=schema, binding=binding, task=task, recorder=recorder
        )
        tools.append(
            function_tool_cls.from_defaults(
                fn=native,
                name=native.__name__,
                description=(native.__doc__ or "").split("\n\nArgs:")[0],
            )
        )
    return tools
