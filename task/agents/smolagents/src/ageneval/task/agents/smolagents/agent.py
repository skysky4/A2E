"""SmolAgentsAgent / SmolAgentsTauAgent — single-agent runner.

Dataset-agnostic ``SmolAgentsAgent`` takes an ``AgentBinding`` and drives
any benchmark whose binding is provided. ``SmolAgentsTauAgent`` is a
thin wrapper that builds the τ-bench binding for the caller.

Tracing is fully automatic: ``SmolagentsInstrumentor`` (installed by
``setup_instrumentation(framework="smolagents")``) wraps every step /
tool call and emits OpenInference spans. **Do not add manual spans
inside this module.**
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ageneval.task.core import AgentBinding, AgentRunner, TaskInput, TaskTrace, ToolCall
from ageneval.task.core.budget import llm_timeout as _llm_timeout
from ageneval.task.core.budget import max_retries as _max_retries
from ageneval.task.core.budget import max_steps as _default_steps
from ageneval.task.core.budget import max_tokens as _max_tokens
from ageneval.task.core.budget import remaining_deadline as _remaining_deadline

from ageneval.task.agents.smolagents.prompts import build_additional_instructions

logger = logging.getLogger(__name__)


class _StopTools(BaseException):
    """Raised from a tool when the shared web/duplicate budget is spent.

    BaseException so smolagents cannot swallow it as a normal tool error
    and keep looping.
    """


_MAX_STEPS = _default_steps()
# Unified model: default to .env's A2E_MODEL (a non-reasoning instruct model);
# fall back to qwen-plus. Never default to a model the endpoint does not serve.
_DEFAULT_MODEL = os.environ.get("A2E_MODEL") or "qwen-plus"

# JSON Schema types map → Python type names smolagents accepts in Tool.inputs.
_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}


def _coerce_inputs(parameters: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Translate an OpenAI/JSON-Schema parameters block into smolagents' format.

    smolagents requires every input to declare ``type`` (one of the supported
    names) and ``description``. Missing types fall back to ``"string"``;
    objects/arrays are kept as-is and the LLM is asked to pass JSON.
    """
    props = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    out: dict[str, dict[str, str]] = {}
    for key, spec in (props or {}).items():
        if not isinstance(spec, Mapping):
            out[key] = {"type": "string", "description": str(spec)[:200]}
            continue
        raw_type = spec.get("type", "string")
        if isinstance(raw_type, list):
            raw_type = raw_type[0] if raw_type else "string"
        py_type = _TYPE_MAP.get(str(raw_type), "string")
        out[key] = {
            "type": py_type,
            "description": str(spec.get("description") or key)[:500],
        }
    return out


def _make_smolagents_tool(
    schema: Mapping[str, Any],
    executor: Any,
    initial_state: Mapping[str, Any],
    recorder: list[ToolCall],
) -> Any:
    """Build a smolagents.Tool subclass at runtime from one OpenAI tool spec.

    The factory function is a closure over the binding executor + the
    current task's initial_state + a shared ``recorder`` list so each
    forward() call is also captured into our TaskTrace.tool_calls.
    """
    # Lazy import to keep smolagents fully optional at import time.
    from smolagents import Tool  # type: ignore

    fn = schema.get("function", schema)
    name = str(fn.get("name", "tool"))
    description = str(fn.get("description", ""))[:1000]
    inputs = _coerce_inputs(fn.get("parameters", {}) or {})

    class _BoundTool(Tool):  # type: ignore[misc]
        # forward() intentionally takes **kwargs: the input schema is built
        # dynamically per tool at runtime, so a fixed signature is impossible.
        # smolagents (tools.py) honours this flag to skip its check that the
        # forward() signature matches the declared `inputs` keys.
        skip_forward_signature_validation = True

        name = ""  # set below
        description = ""
        inputs: dict[str, dict[str, str]] = {}
        output_type = "string"

        def forward(self, *args: Any, **kwargs: Any) -> str:
            from ageneval.task.core.native_tools import (
                execute_recorded_tool,
                is_stop_tool_result,
                unwrap_tool_kwargs,
            )

            merged = dict(kwargs)
            for arg in args:
                if isinstance(arg, dict):
                    merged.update(arg)
                    continue
                text = str(arg or "").strip()
                if text.startswith("{}{"):
                    text = text[2:]
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    merged.update(parsed)
            text = execute_recorded_tool(
                tool_name=self.name,
                kwargs=unwrap_tool_kwargs(merged),
                executor=executor,
                initial_state=initial_state,
                recorder=recorder,
            )
            if is_stop_tool_result(text):
                raise _StopTools(text)
            return text

    _BoundTool.__name__ = f"Tool_{name}"
    _BoundTool.name = name
    _BoundTool.description = description or f"Invoke the {name} tool."
    _BoundTool.inputs = inputs
    return _BoundTool()


def _build_tools(
    schemas: Sequence[Mapping[str, Any]],
    executor: Any,
    initial_state: Mapping[str, Any],
    recorder: list[ToolCall],
) -> list[Any]:
    return [_make_smolagents_tool(s, executor, initial_state, recorder) for s in schemas]


@dataclass(eq=False)
class SmolAgentsAgent(AgentRunner):
    """Single-agent runner powered by smolagents, framework-agnostic.

    Accepts any ``AgentBinding`` — adding a new benchmark means writing a
    new ``binding.py`` under ``task/datasets/<bench>/``; **no new agent
    file**. smolagents drives an LLM (via ``OpenAIServerModel``) through a
    code-or-tool-calling loop; A2E's OpenInference instrumentor captures
    every step automatically.
    """

    binding: AgentBinding | None = None
    model: str = _DEFAULT_MODEL
    max_steps: int = _MAX_STEPS
    api_base: str | None = None
    api_key: str | None = None

    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.binding is None:
            raise ValueError("SmolAgentsAgent requires a binding")
        self.name = f"smolagents-{self.binding.name}"

    async def run(self, task: TaskInput) -> TaskTrace:
        # Blocking smolagents.run() executed in a thread so we keep the
        # async contract of AgentRunner.
        start = time.perf_counter()
        if self.binding is not None and os.environ.get("A2E_TAU_NEED_WRITE") == "1":
            from ageneval.task.core.native_tools import maybe_force_retail_write_trace

            forced = await maybe_force_retail_write_trace(
                binding=self.binding,
                task=task,
                recorder=[],
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_steps if hasattr(self, "max_steps") else self.max_turns,
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
                recorder=[],
                model=self.model,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "",
                api_base=self.api_base or os.environ.get("OPENAI_API_BASE"),
                max_turns=self.max_steps,
                deadline=_remaining_deadline(start),
                agent_name=self.name,
                start=start,
            )
            if forced_ds is not None:
                return forced_ds
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, task),
                timeout=_remaining_deadline(start),
            )
        except asyncio.TimeoutError:
            rec = list(getattr(self, "_recorder", []) or [])
            from ageneval.task.core.native_tools import compose_final_answer

            final = compose_final_answer(
                getattr(self, "_task_instruction", task.instruction), rec
            )
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "timeout",
                turns=len(rec),
                tool_calls=tuple(rec),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                error=None if final else f"agent exceeded {_remaining_deadline(start):.0f}s deadline",
                raw=dict(getattr(self, "_prompt_meta", {}) or {}),
            )

    def _run_sync(self, task: TaskInput) -> TaskTrace:
        start = time.perf_counter()
        recorder: list[ToolCall] = []

        try:
            from smolagents import OpenAIServerModel, ToolCallingAgent  # type: ignore
        except ImportError as exc:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                tool_calls=(),
                elapsed_seconds=time.perf_counter() - start,
                error=(
                    f"smolagents is not installed: {exc}. "
                    "Run `uv sync` at the A2E workspace root."
                )[:1000],
            )

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
                error=(
                    "smolagents requires OPENAI_API_KEY (smolagents' "
                    "OpenAIServerModel uses an OpenAI-compatible endpoint). "
                    "Set OPENAI_API_KEY (and OPENAI_API_BASE for self-hosted "
                    "endpoints) and retry."
                ),
            )

        try:
            assert self.binding is not None  # for type-checkers
            tools = _build_tools(
                self.binding.tool_schemas,
                self.binding.tool_executor,
                task.initial_state,
                recorder,
            )
            # smolagents' ToolCallingAgent defaults to tool_choice="required",
            # which forces a tool call each step. Non-reasoning instruct models
            # (e.g. qwen-plus) support this natively. Reasoning models reject
            # tool_choice — for those, switch the unified model in .env to a
            # non-reasoning instruct model.
            # timeout / max_retries belong on the OpenAI client, not on
            # Completions.create() — extra kwargs are forwarded to create().
            model_kwargs: dict[str, Any] = {
                "model_id": self.model,
                "api_base": api_base,
                "api_key": api_key,
                "max_tokens": _max_tokens(),
                "client_kwargs": {
                    "timeout": _llm_timeout(),
                    "max_retries": _max_retries(),
                },
            }
            try:
                model = OpenAIServerModel(**model_kwargs)
            except TypeError:
                model_kwargs.pop("client_kwargs", None)
                model = OpenAIServerModel(**model_kwargs)
            additional = build_additional_instructions(self.binding.render_system_prompt())
            if not additional.strip():
                additional = (
                    "Follow the user task. When you have the answer, call final_answer. "
                    "Do not leave the final answer empty."
                )
            agent = ToolCallingAgent(
                tools=tools,
                model=model,
                max_steps=self.max_steps,
                instructions=additional,
            )
            # Always pin the binding policy at the top. The stock Jinja
            # template only emits ``custom_instructions`` inside
            # ``{% if custom_instructions %}``; a blank/None instructions
            # used to wipe the dataset system prompt.
            template = agent.prompt_templates.get("system_prompt") or ""
            pin = (
                "## Dataset policy (required)\n"
                + additional.strip()
                + "\n\n"
            )
            if pin not in template:
                agent.prompt_templates["system_prompt"] = pin + template
            rendered = (agent.system_prompt or "").strip()
            if not rendered or additional[:40] not in rendered:
                agent.prompt_templates["system_prompt"] = pin + (template or "You are a helpful agent.")
                rendered = (agent.system_prompt or "").strip()
            if not rendered:
                raise RuntimeError("smolagents system prompt is empty after binding inject")
            self._recorder = recorder
            self._task_instruction = task.instruction
            self._prompt_meta = {
                "additional_instructions": additional,
                "system_prompt_chars": len(rendered),
                "system_prompt_preview": rendered[:500],
            }
            result = agent.run(task.instruction, additional_args=None)
        except _StopTools:
            from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

            ensure_required_tools(binding=self.binding, task=task, recorder=recorder)
            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=_count_steps(locals().get("agent")),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=time.perf_counter() - start,
                raw=dict(getattr(self, "_prompt_meta", {}) or {}),
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc) or type(exc).__name__
            lower = msg.lower()
            hint = ""
            if "authentication" in lower or "401" in lower or "403" in lower:
                hint = " — OpenAI-compatible auth failed. Check OPENAI_API_KEY / OPENAI_API_BASE."
            elif "connection" in lower or "timeout" in lower:
                hint = " — network error reaching the OpenAI-compatible endpoint."
            elapsed = time.perf_counter() - start
            from ageneval.task.core.native_tools import compose_final_answer

            final = compose_final_answer(task.instruction, recorder)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="ok" if final else "error",
                turns=_count_steps(locals().get("agent")),
                tool_calls=tuple(recorder),
                final_answer=final or None,
                elapsed_seconds=elapsed,
                error=None if final else (msg + hint)[:1000],
                raw=dict(getattr(self, "_prompt_meta", {}) or {}),
            )

        final_answer = _stringify(result)
        from ageneval.task.core.native_tools import compose_final_answer, ensure_required_tools

        ensure_required_tools(binding=self.binding, task=task, recorder=recorder)

        try:
            from ageneval.task.core.native_tools import is_unusable_final
        except ImportError:  # stale/partial native_tools during live edits
            def is_unusable_final(text: str) -> bool:
                return not (text or "").strip()

        if is_unusable_final(final_answer or ""):
            final_answer = compose_final_answer(
                task.instruction, recorder, existing=final_answer or ""
            )
        turns = _count_steps(agent)
        elapsed = time.perf_counter() - start
        status = "ok" if final_answer else ("max_turns" if turns >= self.max_steps else "error")
        # Hook the additional instructions into raw for downstream inspection.
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,
            turns=turns,
            tool_calls=tuple(recorder),
            final_answer=final_answer,
            elapsed_seconds=elapsed,
            raw=dict(getattr(self, "_prompt_meta", {}) or {
                "additional_instructions": additional,
                "system_prompt_chars": len((agent.system_prompt or "").strip()),
            }),
        )


# ─── backwards-compat wrapper for τ-bench ─────────────────────────────────────


@dataclass(eq=False)
class SmolAgentsTauAgent(SmolAgentsAgent):
    """Thin wrapper: ``SmolAgentsTauAgent(domain="retail")`` resolves the
    τ-bench binding automatically. New benchmarks should pass a custom
    ``AgentBinding`` directly to ``SmolAgentsAgent``.
    """

    domain: str = "retail"
    binding: AgentBinding | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.binding is None:
            from ageneval.task.datasets.tau_bench import build_tau_bench_binding

            self.binding = build_tau_bench_binding(self.domain)  # type: ignore[arg-type]
        super().__post_init__()


# ─── helpers ──────────────────────────────────────────────────────────────────


def _stringify(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result.strip() or None
    try:
        return json.dumps(result, default=str)
    except Exception:  # noqa: BLE001
        return str(result)


def _count_steps(agent: Any) -> int:
    """Best-effort step count from smolagents' memory; older versions differ."""
    if agent is None:
        return 0
    memory = getattr(agent, "memory", None)
    steps = getattr(memory, "steps", None) if memory is not None else None
    if steps is None:
        steps = getattr(agent, "logs", None)
    try:
        return int(len(steps)) if steps is not None else 0
    except TypeError:
        return 0
