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

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ageneval.task.agents.smolagents.prompts import build_agent_instructions
from ageneval.task.core import (
    AgentBinding,
    AgentRunner,
    TaskInput,
    TaskTrace,
    ToolCall,
    run_sync_in_daemon_thread,
)
from ageneval.task.core.budget import max_steps as _default_steps
from ageneval.task.core.budget import max_tokens as _max_tokens

logger = logging.getLogger(__name__)

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


def _coerce_inputs(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate an OpenAI/JSON-Schema parameters block into smolagents' format.

    smolagents requires every input to declare ``type`` (one of the supported
    names) and ``description``. Missing types fall back to ``"string"``;
    objects/arrays are kept as-is and the LLM is asked to pass JSON.
    """
    props = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    raw_required = parameters.get("required", ()) if isinstance(parameters, Mapping) else ()
    required = (
        {str(key) for key in raw_required}
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes))
        else set()
    )
    out: dict[str, dict[str, Any]] = {}
    for key, spec in (props or {}).items():
        if not isinstance(spec, Mapping):
            input_spec: dict[str, Any] = {
                "type": "string",
                "description": str(spec)[:200],
            }
            if key not in required:
                input_spec["nullable"] = True
            out[key] = input_spec
            continue
        raw_type = spec.get("type", "string")
        if isinstance(raw_type, list):
            raw_type = raw_type[0] if raw_type else "string"
        py_type = _TYPE_MAP.get(str(raw_type), "string")
        input_spec = {
            "type": py_type,
            "description": str(spec.get("description") or key)[:500],
        }
        # smolagents derives its model-facing ``required`` list from the
        # inverse of ``nullable``. Preserve the source JSON Schema's required
        # list or optional command-specific fields (for example old_str on a
        # str_replace_editor ``view`` call) are incorrectly rejected.
        if key not in required:
            input_spec["nullable"] = True
        out[key] = input_spec
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
        inputs: ClassVar[dict[str, dict[str, Any]]] = {}
        output_type = "string"

        def forward(self, **kwargs: Any) -> str:
            from ageneval.task.core.native_tools import execute_recorded_tool

            return execute_recorded_tool(
                tool_name=self.name,
                kwargs=kwargs,
                executor=executor,
                initial_state=initial_state,
                recorder=recorder,
            )

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
        # smolagents has no native async runner. A daemon thread preserves the
        # async AgentRunner contract without making asyncio.run() wait forever
        # for an SDK call that outlives a task timeout.
        return await run_sync_in_daemon_thread(
            self._run_sync,
            task,
            thread_name=f"a2e-{self.name}-{task.task_id}",
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
                    f"smolagents is not installed: {exc}. Run `uv sync` at the A2E workspace root."
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
            # non-reasoning instruct model (see test/mingxuan/README.md).
            model = OpenAIServerModel(
                model_id=self.model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=_max_tokens(),
            )
            # The binding's policy text must reach the model, so it goes in as
            # smolagents' `instructions` (spliced into the rendered system
            # prompt). `run(additional_args=...)` is a variables dict, not an
            # instruction channel — passing it there silently dropped the
            # prompt and left every harness but this one task-aware.
            instructions = build_agent_instructions(self.binding.render_system_prompt())
            agent = ToolCallingAgent(
                tools=tools,
                model=model,
                max_steps=self.max_steps,
                instructions=instructions or None,
            )
            result = agent.run(task.instruction)
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            lower = msg.lower()
            hint = ""
            if "authentication" in lower or "401" in lower or "403" in lower:
                hint = " — OpenAI-compatible auth failed. Check OPENAI_API_KEY / OPENAI_API_BASE."
            elif "connection" in lower or "timeout" in lower:
                hint = " — network error reaching the OpenAI-compatible endpoint."
            elapsed = time.perf_counter() - start
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=_count_steps(locals().get("agent")),
                tool_calls=tuple(recorder),
                elapsed_seconds=elapsed,
                error=(msg + hint)[:1000],
            )

        final_answer = _stringify(result)
        turns = _count_steps(agent)
        elapsed = time.perf_counter() - start
        status = "ok" if final_answer else ("max_turns" if turns >= self.max_steps else "error")
        # Record the instructions actually handed to the agent, for auditing.
        return TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,
            turns=turns,
            tool_calls=tuple(recorder),
            final_answer=final_answer,
            elapsed_seconds=elapsed,
            raw={"instructions": instructions} if instructions else {},
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
    except Exception:
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
        return len(steps) if steps is not None else 0
    except TypeError:
        return 0
