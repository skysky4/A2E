"""AutogenAgentChatAgent — single-agent runner powered by Microsoft AutoGen.

Dataset-agnostic: consumes an ``AgentBinding`` and drives any benchmark.
The ``openinference-instrumentation-autogen-agentchat`` instrumentor (installed
by ``setup_instrumentation(framework="autogen_agentchat")``) captures spans
automatically. **Do not add manual spans inside this module.**

Module-level imports are restricted to core + stdlib: the ``autogen_agentchat``
/ ``autogen_ext`` / ``autogen_core`` SDKs are imported lazily inside
``__post_init__`` and ``run`` so that ``import
ageneval.task.agents.autogen_agentchat`` never fails when the runtime SDK is
absent.

ISOLATION NOTE: AutoGen lives in an isolated uv project (see this package's
README.md) because ``autogen-core`` pins ``protobuf<5.30`` while A2E needs
``protobuf>=6.31``. The two cannot share one environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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
_RETAIL_WRITE = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}


def _schema_tool_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return str(getattr(schema, "name", "") or "")
    return str(
        schema.get("name")
        or (schema.get("function") or {}).get("name")
        or ""
    )


def _tool_transcript(recorder: list[ToolCall], *, limit: int = 10) -> str:
    bits: list[str] = []
    for tc in recorder[-limit:]:
        args = tc.arguments if isinstance(getattr(tc, "arguments", None), dict) else {}
        try:
            arg_s = json.dumps(args, ensure_ascii=False)[:240]
        except Exception:  # noqa: BLE001
            arg_s = str(args)[:240]
        res = "" if tc.result is None else str(tc.result)
        if len(res) > 400:
            res = res[:400] + "…"
        bits.append(f"- {tc.name}({arg_s}) -> {res}")
    return "\n".join(bits)


@dataclass(eq=False)
class AutogenAgentChatAgent(AgentRunner):
    """Single-agent runner powered by AutoGen AgentChat, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a new
    ``binding.py`` under ``task/datasets/<bench>/``; **no new agent file**.
    AutoGen drives an LLM through an OpenAI-compatible endpoint
    (``OpenAIChatCompletionClient``); A2E's OpenInference instrumentor captures
    every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_turns: int = _MAX_TURNS
    api_base: str | None = None
    api_key: str | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("AutogenAgentChatAgent requires a binding")
        self.name = f"autogen-agentchat-{self.binding.name}"
        try:
            import autogen_agentchat  # noqa: F401  — the autogen-agentchat package
        except ImportError as exc:
            raise RuntimeError(
                "autogen-agentchat agent requires its runtime SDK. Because "
                "autogen-core conflicts with A2E on protobuf, this agent "
                "lives in an isolated uv project. Install with:\n"
                "  cd task/agents/autogen_agentchat && "
                "uv sync --index-strategy unsafe-best-match"
            ) from exc

    async def run(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient

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
                    error="autogen-agentchat requires OPENAI_API_KEY",
                )

            assert self.binding is not None  # for type-checkers
            write_schema = any(
                _schema_tool_name(s) in _RETAIL_WRITE
                for s in (self.binding.tool_schemas or ())
            )
            need_write = write_schema and (
                os.environ.get("A2E_TAU_NEED_WRITE") == "1"
                or "already confirm" in (task.instruction or "").lower()
            )
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
            if need_write:
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
            client_kw: dict[str, Any] = dict(
                model=self.model,
                base_url=api_base,
                api_key=api_key,
                model_info=_build_model_info(self.model),
                timeout=_llm_timeout(),
                max_retries=_max_retries(),
                max_tokens=_max_tokens(),
            )
            if need_write:
                client_kw["extra_create_args"] = {"tool_choice": "required"}
            try:
                model_client = OpenAIChatCompletionClient(**client_kw)
            except TypeError:
                client_kw.pop("extra_create_args", None)
                model_client = OpenAIChatCompletionClient(**client_kw)
            tools = _build_function_tools(self.binding, task, recorder)
            system = self.binding.render_system_prompt()
            if tools:
                system += (
                    "\nYou are the support AGENT, not the customer. "
                    "On the first turn you MUST call a lookup tool "
                    "(find_user_id_by_email or find_user_id_by_name_zip) "
                    "with identifiers already in the task."
                )
            agent = AssistantAgent(
                name="a2e_agent",
                model_client=model_client,
                tools=tools,
                system_message=system,
                max_tool_iterations=self.max_turns,
            )
            user_text = task.instruction
            if tools:
                user_text = (
                    "Call a lookup tool before answering. "
                    "Do not speak as the customer.\n\n" + task.instruction
                )
            result = await asyncio.wait_for(
                agent.run(task=user_text),
                timeout=_remaining_deadline(start),
            )
            if (
                need_write
                and not any(tc.name in _RETAIL_WRITE for tc in recorder)
                and _remaining_deadline(start) > 20
            ):
                from ageneval.task.core.native_tools import force_retail_write_calls

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
                    result = type("R", (), {"messages": []})()
            try:
                await model_client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

            final = _extract_final(result)
            from ageneval.task.core.native_tools import (
                bootstrap_lookup_call,
                compose_final_answer,
                invoke_binding_tool,
                is_unusable_final,
                openai_function,
            )

            if tools and not recorder:
                available = [
                    str(openai_function(s).get("name") or "")
                    for s in (self.binding.tool_schemas or [])
                    if openai_function(s).get("name")
                ]
                boot = bootstrap_lookup_call(task.instruction, available)
                if boot:
                    invoke_binding_tool(
                        tool_name=str(boot["name"]),
                        kwargs=boot.get("arguments") or {},
                        binding=self.binding,
                        task=task,
                        recorder=recorder,
                    )

            if is_unusable_final(final or ""):
                final = compose_final_answer(task.instruction, recorder, existing=final or "")
            turns = _count_turns(result) or len(recorder)
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
            from ageneval.task.core.native_tools import compose_final_answer

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
            from ageneval.task.core.native_tools import compose_final_answer

            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=len(recorder),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else (str(exc) or type(exc).__name__)[:1000],
            )


def _build_model_info(model: str) -> dict[str, Any]:
    """Build a ``ModelInfo`` dict for a non-OpenAI-official model.

    ``OpenAIChatCompletionClient`` cannot infer capabilities for models it does
    not recognise (e.g. ``qwen-plus``), so an explicit ``model_info`` is
    required. Keys mirror the ``autogen_core.models.ModelInfo`` TypedDict;
    ``structured_output`` and ``multiple_system_messages`` are recent additions
    that older autogen-core versions tolerate as extra keys.
    """
    return {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
        "multiple_system_messages": True,
    }


def _extract_final(result: Any) -> str:
    """Best-effort final-answer extraction from an autogen ``TaskResult``."""
    messages = getattr(result, "messages", None) or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            parts = [str(p) for p in content if isinstance(p, str)]
            text = " ".join(parts).strip()
            if text:
                return text
    return ""


def _count_turns(result: Any) -> int:
    """Best-effort turn count: number of model-produced text messages."""
    messages = getattr(result, "messages", None) or []
    return sum(1 for m in messages if type(m).__name__ == "TextMessage")


def _build_function_tools(
    binding: AgentBinding,
    task: TaskInput,
    recorder: list[ToolCall],
) -> list[Any]:
    """Wrap each binding tool schema into a plain Python function for AutoGen.

    AutoGen infers the published schema from the function signature. We attach
    the dataset JSON-Schema properties as keyword-only parameters.
    """
    from ageneval.task.core.native_tools import make_kwargs_tool

    return [
        make_kwargs_tool(schema=schema, binding=binding, task=task, recorder=recorder)
        for schema in binding.tool_schemas
    ]
