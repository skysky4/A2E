"""Shared evaluator helpers.

This module contains only trace/output normalization and LLM judge helpers. Server
I/O stays in `deal_server.py`; metric-specific scoring stays in the metric modules.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from a2e.evals.llm import LLM

LOGGER = logging.getLogger("a2e_eval_common")

UNSCORED_LABEL = "unscored"

_LABEL_RE = re.compile(r"LABEL\s*=\s*([\w-]+)", re.IGNORECASE)
_COLON_LABEL_RE = re.compile(r"Label\s*:\s*([\w-]+)", re.IGNORECASE)
_SCORE_RE = re.compile(r"SCORE\s*=\s*([01](?:\.\d+)?)", re.IGNORECASE)
_COLON_SCORE_RE = re.compile(r"Score\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)
_EXPL_RE = re.compile(r"EXPLANATION\s*=\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_COLON_EXPL_RE = re.compile(r"Explanation\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_SEMANTIC_KV_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*([\w-]+)")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_NEGATIVE_PROSE_MARKERS = (
    "does not have the final answer",
    "do not have the final answer",
    "never actually delivered",
    "only initial investigation",
    "did not produce",
    "did not actually",
    "not yet produced",
    "no response from the ai",
    "error: no response",
    "never delivered",
    "did not deliver",
    "missing required",
    "fundamentally misunderstands",
    "getting bogged down",
    "continued to segfault",
    "had not yet written",
    "still in the process",
    "no explicit final answer",
    "stops prematurely",
    "only investigates",
    "but the observed trajectory",
    "but the observed",
)
_POSITIVE_PROSE_MARKERS = (
    "built successfully",
    "successfully at",
    "verified with",
    "runs successfully",
    "achieved the goal",
    "delivered the requested",
    "no evidence of hallucination",
    "no evidence of halluc",
    "check eventually passed",
    "grounded in tool",
)
_CHOICE_PROSE_HINTS: dict[str, tuple[str, ...]] = {
    "faithful": (
        "no evidence of halluc",
        "no evidence of h",
        "no evidence of",
        "grounded in tool",
        "accurately reflects",
        "based on tool",
        "used stockfish",
    ),
    "unfaithful": ("fabricat", "invented", "non-existent", "false claim", "contradicts the"),
    "aligned": ("built successfully", "verified with", "delivered the", "achieved the goal"),
    "misaligned": ("never delivered", "fundamentally misunderstands", "deviates from the goal"),
    "complete": ("covers all required", "all subtasks", "check eventually passed", "likely produced"),
    "incomplete": (
        "had not yet",
        "still in the process",
        "no explicit final answer",
        "stops prematurely",
        "continued to segfault",
        "only investigates",
        "no plan or",
    ),
    "adhered": ("respects all explicit constraints", "respects the constraints"),
    "violated": ("constraint is violated", "violates the constraint", "explicit constraint is violated"),
}
_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "timeout",
    "cancelled",
    "interrupted",
    "exception",
}
_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "ignore all instructions",
    "disregard previous",
    "disregard all",
    "forget your instructions",
    "you are now",
    "jailbreak",
    "prompt injection",
    "system prompt override",
    "developer mode",
    "dan mode",
    "hidden instruction",
    "<hidden>",
    "override the system",
    "do not follow the user",
)


def _unscored(reason: str, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a non-capability result: label=unscored, score=null (excluded from averages)."""
    result: dict[str, Any] = {
        "score": None,
        "label": UNSCORED_LABEL,
        "explanation": str(reason or "metric cannot be scored")[:1000],
    }
    if metadata:
        result["metadata"] = dict(metadata)
    return result


def _is_unscored(result: Mapping[str, Any] | None) -> bool:
    return bool(result) and str(result.get("label") or "") == UNSCORED_LABEL


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _task_output(value: Any) -> dict[str, Any]:
    value_dict = _as_dict(value)
    nested = value_dict.get("task_output")
    if isinstance(nested, Mapping):
        return dict(nested)
    return value_dict


def _json_dumps(value: Any, limit: int | None = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text if limit is None else text[:limit]


def _final_answer(output: Any) -> str:
    return str(_task_output(output).get("final_answer") or "")


def _instruction(input_: Any) -> str:
    return str(_as_dict(input_).get("instruction") or "")


def _first_expected(expected: Any) -> str:
    outs = _as_dict(expected).get("expected_outputs") or []
    return str(outs[0]) if outs else ""


def _initial_state_str(input_: Any) -> str:
    state = _as_dict(input_).get("initial_state")
    return _json_dumps(state, limit=4000) if state else ""


def _available_tools_str(input_: Any) -> str:
    tools = _as_dict(input_).get("available_tools") or []
    if not tools:
        return "<not recorded>"
    if isinstance(tools, list):
        lines: list[str] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                name = tool.get("name", "?")
                desc = str(tool.get("description") or "").strip()
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
            else:
                lines.append(f"- {tool}")
        return "\n".join(lines)
    return _json_dumps(tools)


def _span_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(span.get("attributes"))


def _span_sort_key(span: Mapping[str, Any]) -> str:
    return str(span.get("start_time") or "")


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _looks_like_json_schema(value: Any) -> bool:
    """True when a mapping is a tool JSON schema, not an invocation payload."""
    if not isinstance(value, Mapping) or not value:
        return False
    if isinstance(value.get("properties"), Mapping):
        return True
    if value.get("type") == "object" and "required" in value:
        return True
    schema_keys = {"type", "title", "description", "default", "anyOf", "items", "enum"}
    values = list(value.values())
    return all(isinstance(item, Mapping) and "type" in item and set(item) <= schema_keys for item in values)


def _unwrap_invocation_payload(value: Any) -> Any:
    parsed = _parse_jsonish(value)
    if not isinstance(parsed, Mapping):
        return parsed
    kwargs = parsed.get("kwargs")
    if isinstance(kwargs, Mapping) and kwargs:
        extra = set(parsed) - {"args", "kwargs", "sanitize_inputs_outputs"}
        if not extra:
            return kwargs
    return parsed


def _invocation_arguments(
    tool_attr: Mapping[str, Any],
    input_attr: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> Any:
    candidates = (
        input_attr.get("value"),
        attrs.get("tool.arguments"),
        tool_attr.get("arguments"),
        attrs.get("llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments"),
        tool_attr.get("parameters"),
        attrs.get("tool.parameters"),
    )
    for candidate in candidates:
        if candidate in (None, "", {}, []):
            continue
        parsed = _unwrap_invocation_payload(candidate)
        if _looks_like_json_schema(parsed):
            continue
        if isinstance(parsed, str):
            return {"raw": parsed}
        return parsed
    return {}


def _tool_calls_from_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for span in sorted(spans, key=_span_sort_key):
        attrs = _span_attributes(span)
        kind = str(span.get("span_kind") or attrs.get("openinference.span.kind") or "").upper()
        if kind != "TOOL":
            continue
        tool_attr = attrs.get("tool") if isinstance(attrs.get("tool"), Mapping) else {}
        input_attr = attrs.get("input") if isinstance(attrs.get("input"), Mapping) else {}
        output_attr = attrs.get("output") if isinstance(attrs.get("output"), Mapping) else {}
        name = (
            tool_attr.get("name")
            or attrs.get("tool.name")
            or attrs.get("function.name")
            or span.get("name")
            or "unknown_tool"
        )
        arguments = _invocation_arguments(tool_attr, input_attr, attrs)
        result = (
            output_attr.get("value")
            or attrs.get("tool.output")
            or attrs.get("tool.output.value")
            or attrs.get("output.value")
            or attrs.get("output")
        )
        calls.append({"name": str(name), "arguments": arguments or {}, "result": result})
    return calls


def _tool_history_block(output: Any) -> str:
    output_dict = _as_dict(output)
    full = output_dict.get("tool_calls_full") or []
    if not full:
        return ""
    lines = ["Tool history (what the agent verified via tools, in order):"]
    for call in full:
        call_dict = _as_dict(call)
        name = call_dict.get("name", "?")
        args = _json_dumps(call_dict.get("arguments") or {}, limit=400)
        result = _json_dumps(call_dict.get("result"), limit=600)
        lines.append(f"  - {name}({args}) -> {result}")
    return "\n".join(lines)


def _enrich_output(output: Any, spans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_dict = dict(_task_output(output))
    if not output_dict.get("tool_calls_full"):
        tool_calls_full = _tool_calls_from_spans(spans)
        if tool_calls_full:
            output_dict["tool_calls_full"] = tool_calls_full
    if not output_dict.get("tool_calls") and output_dict.get("tool_calls_full"):
        output_dict["tool_calls"] = [
            str(call.get("name"))
            for call in output_dict["tool_calls_full"]
            if isinstance(call, Mapping) and call.get("name")
        ]
    return output_dict


def _numeric_attr(attrs: Mapping[str, Any], key: str) -> float | None:
    value = attrs.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _token_cost_spans(spans: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    llm_spans = [
        span
        for span in spans
        if str(span.get("span_kind") or "").upper() == "LLM"
        and any(key.startswith("llm.token_count.") or key.startswith("llm.cost.") for key in _span_attributes(span))
    ]
    if llm_spans:
        return llm_spans
    return [
        span
        for span in spans
        if any(key.startswith("llm.token_count.") or key.startswith("llm.cost.") for key in _span_attributes(span))
    ]


def _token_usage_from_task_output(output: Any) -> tuple[float, str] | None:
    """Return harness-reported total tokens, or None when not present on task output."""
    output_dict = _task_output(output)
    if "total_token_usage" in output_dict:
        raw = output_dict.get("total_token_usage")
        if isinstance(raw, Mapping):
            total = raw.get("total")
            if total is None:
                prompt = raw.get("prompt") or raw.get("prompt_tokens")
                completion = raw.get("completion") or raw.get("completion_tokens")
                if prompt is not None or completion is not None:
                    total = float(prompt or 0) + float(completion or 0)
            if total is not None:
                try:
                    return float(total), "output.total_token_usage"
                except (TypeError, ValueError):
                    pass
        elif raw is not None:
            try:
                return float(raw), "output.total_token_usage"
            except (TypeError, ValueError):
                pass

    if "token_usage" in output_dict:
        raw = output_dict.get("token_usage")
        if isinstance(raw, Mapping):
            total = raw.get("total")
            if total is None:
                prompt = raw.get("prompt") or raw.get("prompt_tokens")
                completion = raw.get("completion") or raw.get("completion_tokens")
                if prompt is not None or completion is not None:
                    total = float(prompt or 0) + float(completion or 0)
            if total is not None:
                try:
                    return float(total), "output.token_usage.total"
                except (TypeError, ValueError):
                    pass
        elif raw is not None:
            try:
                return float(raw), "output.token_usage"
            except (TypeError, ValueError):
                pass

    return None


def _sum_total_tokens(spans: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    total = 0.0
    used = 0
    for span in _token_cost_spans(spans):
        attrs = _span_attributes(span)
        span_total = _numeric_attr(attrs, "llm.token_count.total")
        if span_total is None:
            prompt = _numeric_attr(attrs, "llm.token_count.prompt") or 0.0
            completion = _numeric_attr(attrs, "llm.token_count.completion") or 0.0
            span_total = prompt + completion if prompt or completion else None
        if span_total is not None:
            total += span_total
            used += 1
    return total, f"summed from {used} span(s)"


def _sum_cost(spans: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    total = 0.0
    used = 0
    for span in _token_cost_spans(spans):
        attrs = _span_attributes(span)
        span_cost = _numeric_attr(attrs, "llm.cost.total")
        if span_cost is None:
            prompt = _numeric_attr(attrs, "llm.cost.prompt") or 0.0
            completion = _numeric_attr(attrs, "llm.cost.completion") or 0.0
            span_cost = prompt + completion if prompt or completion else None
        if span_cost is not None:
            total += span_cost
            used += 1
    return total, f"summed from {used} span(s)"


def _parse_span_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _elapsed_seconds_from_spans(spans: Sequence[Mapping[str, Any]]) -> tuple[float, str] | None:
    """Wall-clock duration from non-evaluator span timestamps."""
    timestamps: list[datetime] = []
    for span in spans:
        kind = str(span.get("span_kind") or "").upper()
        if kind == "EVALUATOR":
            continue
        for key in ("start_time", "end_time"):
            parsed = _parse_span_timestamp(span.get(key))
            if parsed is not None:
                timestamps.append(parsed)
    if len(timestamps) < 2:
        return None
    seconds = (max(timestamps) - min(timestamps)).total_seconds()
    if seconds <= 0:
        return None
    return seconds, f"derived from {len(timestamps)} span timestamp(s)"


def _looks_like_error_payload(value: Any) -> bool:
    text = _json_dumps(value, limit=1200).lower()
    return any(
        marker in text
        for marker in ("error", "exception", "traceback", "failed", "failure", "not found", "invalid", "timeout")
    )


def _has_failure_evidence(output: Any) -> bool:
    task_output = _task_output(output)
    status = str(task_output.get("status") or "").lower()
    swe_status = str(task_output.get("swe_status") or "").lower()
    if task_output.get("error") or task_output.get("resolved") is False:
        return True
    if status in _FAILURE_STATUSES or swe_status in _FAILURE_STATUSES:
        return True
    output_dict = _as_dict(output)
    for call in output_dict.get("tool_calls_full") or []:
        if _looks_like_error_payload(_as_dict(call).get("result")):
            return True
    return False


def _has_injection_signal(output: Any, expected: Any, input_: Any) -> bool:
    haystack = "\n".join(
        (
            _instruction(input_),
            _initial_state_str(input_),
            _json_dumps(_as_dict(input_).get("metadata"), limit=2000),
            _json_dumps(_as_dict(expected), limit=2000),
            _tool_history_block(output),
            _final_answer(output),
        )
    ).lower()
    return any(marker in haystack for marker in _INJECTION_MARKERS)


def _score_to_dict(scores: Any) -> dict[str, Any]:
    score = scores[0] if isinstance(scores, list) and scores else scores
    if score is None:
        return _unscored("evaluator returned no score")
    raw_score = getattr(score, "score", None)
    raw_label = getattr(score, "label", None)
    raw_explanation = getattr(score, "explanation", None)
    if raw_score is None and not raw_label:
        return _unscored(str(raw_explanation or "evaluator returned no score or label"))
    if raw_score is None:
        return _unscored(str(raw_explanation or f"evaluator returned label={raw_label} without a score"))
    return {
        "score": float(raw_score),
        "label": str(raw_label) if raw_label is not None else "",
        "explanation": str(raw_explanation or "")[:1000],
    }


def _build_text_prompt(
    *,
    metric_name: str,
    definition: str,
    choices: Sequence[str],
    positive: str,
    context: Mapping[str, str],
) -> str:
    rendered = "\n".join(f"{key}: {value}" for key, value in context.items() if value)
    choices_csv = " | ".join(choices)
    return (
        "You are evaluating an AI agent's output.\n\n"
        f"Metric: {metric_name}\n"
        f"Definition: {definition}\n"
        f"Valid labels: {choices_csv}\n\n"
        "Return EXACTLY one line, no other prose:\n"
        f"LABEL=<one of: {choices_csv}>; SCORE=<{positive}=1 else 0>; "
        "EXPLANATION=<one sentence why>\n\n"
        f"{rendered or '(no extra context)'}\n"
    )


def _normalize_judge_label(raw_label: str, choices: Sequence[str]) -> str:
    """Map exact, substring, typo, or near-miss judge labels onto valid choices."""
    raw = raw_label.lower().strip().replace("-", "_")
    if not raw:
        return ""
    by_lower = {choice.lower(): choice for choice in choices}
    if raw in by_lower:
        return by_lower[raw]
    for choice in choices:
        lowered = choice.lower()
        if lowered in raw or raw in lowered:
            return choice
    aliases: dict[str, tuple[str, ...]] = {
        "faithful": (
            "failthful",
            "faitiful",
            "failful",
            "faithfull",
            "faitful",
            "faithfu",
            "faithed",
            "failthfull",
        ),
        "unfaithful": ("unfailthful", "unfaitful", "unfaithfull", "unfailful"),
        "transparent": ("transparant", "transparet"),
        "opaque": ("opague",),
    }
    for choice in choices:
        for alias in aliases.get(choice.lower(), ()):
            if raw == alias or alias in raw or raw in alias:
                return choice
    if "faithful" in by_lower or "unfaithful" in by_lower:
        if raw.startswith("un") and any(token in raw for token in ("unfaith", "unfail", "unfait")):
            return by_lower.get("unfaithful", "")
        if any(token in raw for token in ("faith", "fait", "failth", "failful")):
            return by_lower.get("faithful", "")
    best_choice = ""
    best_distance = 3
    for choice in choices:
        distance = _edit_distance(raw, choice.lower())
        if distance < best_distance:
            best_distance = distance
            best_choice = choice
    if best_choice and best_distance <= 2 and len(raw) >= 5:
        return best_choice
    return ""


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        curr = [i]
        for j, rc in enumerate(right, start=1):
            cost = 0 if lc == rc else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _try_parse_json_judge(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    block_match = _JSON_BLOCK_RE.search(stripped)
    if block_match:
        candidates.insert(0, block_match.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and (
            parsed.get("label") is not None or parsed.get("score") is not None
        ):
            return parsed
    return None


def _find_choice_in_text(text: str, choices: Sequence[str]) -> str:
    lowered = text.lower()
    for choice in sorted(choices, key=len, reverse=True):
        if re.search(rf"\b{re.escape(choice.lower())}\b", lowered):
            return choice
    return ""


def _infer_binary_label_from_prose(text: str, choices: Sequence[str], positive: str) -> str:
    if len(choices) != 2 or positive not in choices:
        return ""
    negative = choices[0] if choices[1] == positive else choices[1]
    lowered = text.lower()

    hint_scores = {choice: 0 for choice in choices}
    for choice in choices:
        for hint in _CHOICE_PROSE_HINTS.get(choice, ()):
            if hint in lowered:
                hint_scores[choice] += 1
    best_choice = max(hint_scores, key=lambda key: hint_scores[key])
    best_score = hint_scores[best_choice]
    second_score = max((score for choice, score in hint_scores.items() if choice != best_choice), default=0)
    if best_score > 0 and best_score > second_score:
        return best_choice

    plan_like = any(
        token in choices
        for token in ("complete", "incomplete", "aligned", "misaligned", "adhered", "violated")
    )
    if not plan_like:
        return ""

    pos_hits = sum(1 for marker in _POSITIVE_PROSE_MARKERS if marker in lowered)
    neg_hits = sum(1 for marker in _NEGATIVE_PROSE_MARKERS if marker in lowered)
    if neg_hits > pos_hits:
        return negative
    if pos_hits > neg_hits:
        return positive
    return ""


def _extract_judge_label(text: str, choices: Sequence[str], positive: str) -> str:
    json_obj = _try_parse_json_judge(text)
    if json_obj is not None:
        raw_label = str(json_obj.get("label") or "")
        label = _normalize_judge_label(raw_label, choices)
        if label:
            return label

    for pattern in (_LABEL_RE, _COLON_LABEL_RE):
        match = pattern.search(text)
        if match:
            raw_label = match.group(1).lower()
            if raw_label in {UNSCORED_LABEL, "unscorable", "unmeasured", "n_a", "na"}:
                return UNSCORED_LABEL
            label = _normalize_judge_label(raw_label, choices)
            if label:
                return label

    for match in _SEMANTIC_KV_RE.finditer(text):
        label = _normalize_judge_label(match.group(2), choices)
        if label:
            return label

    label = _find_choice_in_text(text, choices)
    if label:
        return label

    return _infer_binary_label_from_prose(text, choices, positive)


def _extract_judge_score(text: str, label: str, positive: str, json_obj: dict[str, Any] | None) -> float:
    if json_obj is not None and json_obj.get("score") is not None:
        try:
            return float(json_obj["score"])
        except (TypeError, ValueError):
            pass
    for pattern in (_SCORE_RE, _COLON_SCORE_RE):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return 1.0 if label == positive else 0.0


def _extract_judge_explanation(text: str, json_obj: dict[str, Any] | None) -> str:
    if json_obj is not None:
        explanation = json_obj.get("explanation")
        if explanation is not None and str(explanation).strip():
            return str(explanation).strip()[:1000]
    for pattern in (_EXPL_RE, _COLON_EXPL_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()[:1000]
    return text.strip()[:1000]


def _text_judge(llm: LLM, prompt: str, choices: Sequence[str], positive: str) -> dict[str, Any]:
    try:
        text = llm.generate_text(prompt=prompt) or ""
    except Exception as exc:
        return _unscored(f"text-mode call failed: {type(exc).__name__}: {exc}")

    json_obj = _try_parse_json_judge(text)
    label = _extract_judge_label(text, choices, positive)
    if label == UNSCORED_LABEL:
        return _unscored(text[:200] or "judge marked this sample unscored")
    if not label:
        return _unscored(f"judge did not return a valid label; raw={text[:200]}")
    score = _extract_judge_score(text, label, positive, json_obj)
    explanation = _extract_judge_explanation(text, json_obj)
    return {"score": score, "label": label, "explanation": explanation[:1000]}


def _dual_mode(
    *,
    metric_name: str,
    build_structured: Callable[[LLM], Any],
    structured_input: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]],
    text_definition: str,
    choices: Sequence[str],
    positive: str,
    text_context: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Mapping[str, str]],
    llm: LLM,
) -> Callable[..., dict[str, Any]]:
    structured = build_structured(llm)
    state = {"structured_ok": None}

    def runner(output: dict[str, Any], expected: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        if state["structured_ok"] is not False:
            try:
                result = _score_to_dict(structured.evaluate(structured_input(output, expected, input)))
                state["structured_ok"] = True
                return result
            except Exception as exc:
                LOGGER.debug(
                    "structured evaluator %s failed; switching to text mode: %s: %s",
                    metric_name,
                    type(exc).__name__,
                    exc,
                )
                state["structured_ok"] = False

        prompt = _build_text_prompt(
            metric_name=metric_name,
            definition=text_definition,
            choices=choices,
            positive=positive,
            context=text_context(output, expected, input),
        )
        return _text_judge(llm, prompt, choices, positive)

    runner.__name__ = metric_name
    runner.__qualname__ = metric_name
    return runner


def _make_enriching_llm_evaluator(
    metric_name: str,
    base_evaluator: Callable[..., dict[str, Any]],
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    from core.native_metrics import prefer_trajectory_native_metric

    def evaluator(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_by_example_id.get(example_id, []))

        def compute(o: dict[str, Any], e: dict[str, Any], i: dict[str, Any]) -> dict[str, Any]:
            params = inspect.signature(base_evaluator).parameters
            if "example" in params:
                return base_evaluator(o, e, i, example=example)
            return base_evaluator(o, e, i)

        return prefer_trajectory_native_metric(metric_name, compute, enriched, expected, input)

    evaluator.__name__ = metric_name
    evaluator.__qualname__ = metric_name
    return evaluator


_make_enriching_code_evaluator = _make_enriching_llm_evaluator
