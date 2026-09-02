"""Fair total_token_usage: dataset system prompt + traj input + traj output.

Does not use LLM spans or harness-reported token fields. Every run uses the
same tokenizer and the same A2E binding-derived system prompt per benchmark.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from collections.abc import Mapping, Sequence

from core.eval_common import _as_dict, _sum_total_tokens, _task_output, _token_usage_from_task_output
from process_values.correct_eval import normalize_benchmark_name

A2E_TASK_ROOT = Path(os.getenv("A2E_TASK_ROOT", "/home/yuchenyue/A2E/task"))

_TAU_WIKI = A2E_TASK_ROOT / "datasets/tau_bench/src/ageneval/task/datasets/tau_bench/upstream"
_TAU_SUFFIX = (
    "\n\nYou have the tools listed in the function-calling interface. "
    "Call a tool by invoking the function with its named arguments. "
    "Do not emit a JSON action object as plain text. "
    "Identify the user first (email or name+zip) before changing any records. "
    "When the request is complete, reply to the customer in plain language."
)

_GDPVAL_SYSTEM = (
    "You are a top-tier professional completing a real-world, economically "
    "valuable work task in your field of expertise. Read the user's request "
    "carefully and produce the COMPLETE requested deliverable directly as your "
    "reply.\n"
    "- Match the format the task asks for (report, memo, table, spreadsheet "
    "contents, plan, analysis, code, etc.).\n"
    "- If the task references attached files you cannot see, state your "
    "assumptions explicitly and still deliver a full, usable result.\n"
    "- Be thorough, accurate and well-structured. Your reply is the final "
    "deliverable — do not ask clarifying questions."
)

_DEEPSEARCHQA_SYSTEM_PREFIX = (
    "You are a DeepSearchQA research agent. The user asks a multi-step "
    "factual question that must be answered from the open web.\n"
    "You MUST call web_search at least once before answering. "
    "You MUST open official source URLs with open_url "
    "(NHS, federalreserve.gov, or whichever site the question names). "
    "Do not answer from memory. Do not substitute Wikipedia for those sites.\n"
    "Use the listed tools via the function-calling interface with their "
    "named arguments (query=... / url=...). Do not emit a JSON action "
    "object as plain text. "
    "When you have the answer, reply with one JSON object only:\n"
    '  {"final_answer": "<concise answer>"}\n'
    "For list questions, put every required item in final_answer, "
    "separated by commas. Do not mention hidden labels such as "
    "answer_type.\n"
    "AVAILABLE TOOLS:\n"
)


@lru_cache(maxsize=1)
def _tokenizer():
    name = "cl100k_base"
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return name, lambda text: len(enc.encode(text))
    except Exception:
        return "len/4 fallback", lambda text: max(0, len(text) // 4)


def count_tokens(text: str) -> int:
    _, encode = _tokenizer()
    return encode(text or "")


def token_benchmark_family(benchmark: Any) -> str:
    canonical = normalize_benchmark_name(benchmark)
    if canonical in {"tau-bench", "tau2", "tau3"}:
        return "tau"
    if "deepsearchqa" in canonical:
        return "deepsearchqa"
    if "gdpval" in canonical:
        return "gdpval"
    return canonical


def _tau_domain(metadata: Mapping[str, Any], task_id: str) -> str:
    domain = str(metadata.get("domain") or "")
    if domain in {"retail", "airline"}:
        return domain
    tid = task_id.lower()
    return "airline" if "airline" in tid else "retail"


@lru_cache(maxsize=4)
def _tau_system(domain: str) -> str:
    wiki = _TAU_WIKI / domain / "wiki.md"
    if not wiki.is_file():
        raise FileNotFoundError(f"tau wiki missing: {wiki}")
    return wiki.read_text(encoding="utf-8").rstrip() + _TAU_SUFFIX


@lru_cache(maxsize=1)
def _deepsearchqa_system() -> str:
    tools_path = (
        A2E_TASK_ROOT
        / "datasets/deepsearchqa/src/ageneval/task/datasets/deepsearchqa/tools.py"
    )
    if tools_path.is_file():
        import importlib.util
        import sys

        core = A2E_TASK_ROOT / "packages/ageneval-task-core/src"
        dsq = A2E_TASK_ROOT / "datasets/deepsearchqa/src"
        for p in (str(core), str(dsq)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from ageneval.task.datasets.deepsearchqa.binding import build_deepsearchqa_binding

        return build_deepsearchqa_binding().render_system_prompt()
    return _DEEPSEARCHQA_SYSTEM_PREFIX + "- web_search: web search\n- open_url: fetch url\n"


@lru_cache(maxsize=8)
def system_prompt_text(family: str, domain: str = "retail") -> str:
    if family == "tau":
        return _tau_system(domain)
    if family == "deepsearchqa":
        return _deepsearchqa_system()
    if family == "gdpval":
        return _GDPVAL_SYSTEM
    return ""


def instruction_text(input_payload: Any) -> str:
    data = _as_dict(input_payload)
    instruction = str(data.get("instruction") or "")
    initial_state = data.get("initial_state")
    if initial_state not in (None, {}, []):
        instruction += "\n\ninitial_state:\n" + json.dumps(initial_state, ensure_ascii=False)
    return instruction.strip()


def trajectory_output_text(output: Any) -> str:
    task_output = _task_output(output)
    parts: list[str] = []
    final_answer = task_output.get("final_answer")
    if final_answer not in (None, ""):
        parts.append(f"final_answer:\n{final_answer}")
    tool_calls = task_output.get("tool_calls_full") or task_output.get("tool_calls")
    if tool_calls:
        parts.append("tool_calls:\n" + json.dumps(tool_calls, ensure_ascii=False))
    tool_spans = task_output.get("tool_spans")
    if tool_spans:
        parts.append("tool_spans:\n" + json.dumps(tool_spans, ensure_ascii=False))
    return "\n\n".join(parts).strip()


def _tool_record_name(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        name = item.get("name") or item.get("tool") or item.get("tool_name") or item.get("action")
        if not name and isinstance(item.get("function"), dict):
            name = item["function"].get("name")
        return str(name or "").strip()
    return ""


def agent_tool_step_count(output: Any) -> int:
    """Count agent steps with a non-empty tool call (RedundancyBench action a_i,tool)."""
    task_output = _task_output(output)
    for key in ("tool_spans", "tool_calls_full", "tool_calls"):
        arr = task_output.get(key)
        if not isinstance(arr, list) or not arr:
            continue
        return sum(1 for item in arr if _tool_record_name(item))
    return 0


def reported_agent_turn_count(output: Any) -> int | None:
    """Agent-side step count from harness fields, without loop-estimation fallbacks."""
    task_output = _task_output(output)
    raw = task_output.get("turns")
    if raw is None:
        raw = task_output.get("turn_count")
    if raw is None:
        return None
    try:
        turns = int(raw)
    except (TypeError, ValueError):
        return None
    return turns if turns >= 0 else None


def _assistant_message_has_tool_call(message: Mapping[str, Any]) -> bool:
    """True when an assistant message carries a non-empty tool invocation."""
    for key in ("tool_calls", "tool_calls_full"):
        value = message.get(key)
        if isinstance(value, list) and value:
            return True
    for key in ("function_call", "tool_call"):
        if message.get(key):
            return True
    return False


def _idle_turn_count_from_messages(messages: Sequence[Any]) -> tuple[int, int, int]:
    """Return (idle_count, agent_step_count, agent_tool_step_count) for assistant messages."""
    idle = 0
    agent_steps = 0
    tool_steps = 0
    for raw in messages:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("role") or "").lower() != "assistant":
            continue
        agent_steps += 1
        if _assistant_message_has_tool_call(raw):
            tool_steps += 1
        else:
            idle += 1
    return idle, agent_steps, tool_steps


def idle_turn_count(output: Any) -> tuple[int | None, str]:
    """Count RedundancyBench empty-tool-call agent steps.

    RedundancyBench interactive protocol (Sec. 3): an agent action may be an empty tool
    call when the agent sends a message without invoking tools. Appendix C excludes these
    from redundant *tool* judging, but they are still billable agent steps. User-simulator
    steps are excluded: only ``role=assistant`` messages are considered.

    Primary path (full message log): assistant messages with empty/missing tool_calls.
    Fallback (A2E task_output): ``max(0, reported_agent_turns - agent_tool_step_count)``,
    where agent tool steps come from tool_spans / tool_calls_full / tool_calls.
    """
    task_output = _task_output(output)
    messages = task_output.get("messages")
    if isinstance(messages, list) and messages:
        idle, agent_steps, tool_steps = _idle_turn_count_from_messages(messages)
        if agent_steps == 0:
            return None, "messages present but no assistant steps; cannot score idle_turn_count"
        return (
            idle,
            (
                f"{idle} empty-tool-call agent step(s); RedundancyBench Sec.3 "
                f"(messages: agent_steps={agent_steps}, agent_tool_steps={tool_steps})"
            ),
        )

    tool_steps = agent_tool_step_count(output)
    reported_turns = reported_agent_turn_count(output)
    if reported_turns is not None:
        idle = max(0, reported_turns - tool_steps)
        turns_per_tool = reported_turns / max(1, tool_steps)
        return (
            idle,
            (
                f"{idle} empty-tool-call agent step(s); RedundancyBench Sec.3 proxy "
                f"(agent_turns={reported_turns}, agent_tool_steps={tool_steps}, "
                f"turns_per_tool={turns_per_tool:.2f})"
            ),
        )

    if tool_steps > 0:
        return (
            0,
            (
                f"0 empty-tool-call agent step(s); only {tool_steps} agent tool step(s) "
                "observed and agent turns field is missing"
            ),
        )

    return None, "agent turns/messages and tool steps missing; cannot score idle_turn_count"


def label_for_idle_turn_count(count: int) -> str:
    """Return numeric label for frontends that display label instead of score."""
    return str(int(count))


def agent_turn_count(output: Any) -> int:
    """Best-effort turn count for agent-loop token estimation."""
    reported = reported_agent_turn_count(output)
    if reported is not None and reported > 0:
        return reported
    task_output = _task_output(output)
    tool_calls = task_output.get("tool_calls") or task_output.get("tool_calls_full") or []
    if isinstance(tool_calls, list) and tool_calls:
        return max(1, len(tool_calls))
    tool_spans = task_output.get("tool_spans") or []
    if isinstance(tool_spans, list) and tool_spans:
        return max(1, len(tool_spans))
    return 1


def loop_estimated_total_tokens(
    *,
    benchmark: Any,
    input_payload: Any,
    output: Any,
    example_metadata: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Estimate billed tokens from agent-loop replay without per-turn spans.

    Formula (stateless API, system+instruction resent each turn, triangular history):
      total ≈ N*S + N*U + R*(N+1)/2
    where N=turns, S=system tokens, U=instruction tokens, R=trajectory output tokens.
    """
    meta = dict(example_metadata or {})
    family = token_benchmark_family(benchmark)
    task_id = str(meta.get("task_id") or "")
    domain = _tau_domain(meta, task_id) if family == "tau" else "retail"
    system = system_prompt_text(family, domain)
    if not system:
        return 0.0, f"unsupported benchmark family {family!r}"

    instruction = instruction_text(input_payload)
    trajectory = trajectory_output_text(output)
    turns = agent_turn_count(output)
    system_tokens = count_tokens(system)
    instruction_tokens = count_tokens(instruction)
    trajectory_tokens = count_tokens(trajectory)
    total = float(turns * system_tokens + turns * instruction_tokens + trajectory_tokens * (turns + 1) / 2.0)
    tok_name, _ = _tokenizer()
    explanation = (
        f"{total:.0f} tokens; loop_estimated({tok_name}: N={turns}, S={system_tokens}, "
        f"U={instruction_tokens}, R={trajectory_tokens}; N*S+N*U+R*(N+1)/2"
        + (f"; family={family}, domain={domain}" if family == "tau" else f"; family={family}")
        + ")"
    )
    return total, explanation


def total_tokens_preferred(
    *,
    spans: Sequence[Mapping[str, Any]] | None = None,
    benchmark: Any = "",
    input_payload: Any = None,
    output: Any = None,
    example_metadata: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Prefer LLM span sums (same magnitude as Phoenix cumulative), else trajectory-fair."""
    if spans:
        total, source = _sum_total_tokens(spans)
        if total > 0:
            return total, f"{total:.0f} tokens; span_llm_sum({source})"

    if output is not None:
        reported = _token_usage_from_task_output(output)
        if reported and reported[0] > 0:
            return reported[0], f"{reported[0]:.0f} tokens; {reported[1]}"

    loop_total, loop_source = loop_estimated_total_tokens(
        benchmark=benchmark,
        input_payload=input_payload,
        output=output,
        example_metadata=example_metadata,
    )
    if loop_total > 0:
        return loop_total, loop_source

    return trajectory_total_tokens(
        benchmark=benchmark,
        input_payload=input_payload,
        output=output,
        example_metadata=example_metadata,
    )


def trajectory_total_tokens(
    *,
    benchmark: Any,
    input_payload: Any,
    output: Any,
    example_metadata: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Return (token_count, explanation) for system + instruction + trajectory."""
    meta = dict(example_metadata or {})
    family = token_benchmark_family(benchmark)
    task_id = str(meta.get("task_id") or "")
    domain = _tau_domain(meta, task_id) if family == "tau" else "retail"
    system = system_prompt_text(family, domain)
    if not system:
        return 0.0, f"unsupported benchmark family {family!r}"

    instruction = instruction_text(input_payload)
    trajectory = trajectory_output_text(output)
    combined = "\n\n".join(x for x in (system, instruction, trajectory) if x)
    total = float(count_tokens(combined))
    tok_name, _ = _tokenizer()
    explanation = (
        f"{total:.0f} tokens; trajectory_fair({tok_name}: "
        f"system+instruction+output; family={family}"
        + (f", domain={domain}" if family == "tau" else "")
        + ")"
    )
    return total, explanation


def label_for_total_tokens(total: float) -> str:
    if total < 2000:
        return "low"
    if total < 10000:
        return "medium"
    return "high"
