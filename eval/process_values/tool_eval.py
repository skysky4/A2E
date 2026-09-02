"""Tool metric evaluator logic."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from a2e.evals.llm import LLM

from core.eval_common import (
    _as_dict,
    _available_tools_str,
    _build_text_prompt,
    _dual_mode,
    _enrich_output,
    _initial_state_str,
    _instruction,
    _json_dumps,
    _text_judge,
    _tool_history_block,
    _unscored,
)


def make_tool_recall(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
    llm: LLM | None = None,
) -> Callable[..., dict[str, Any]]:
    del llm  # CODE-only: missing expected_actions is unscored, not an LLM fallback

    def tool_recall(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_by_example_id.get(example_id, []))
        called = set(enriched.get("tool_calls") or [])
        expected_names = {
            action.get("name")
            for action in (_as_dict(expected).get("expected_actions") or [])
            if isinstance(action, Mapping) and action.get("name")
        }
        if not expected_names:
            return _unscored(
                "expected_actions is empty in dataset reference; tool_recall requires ground-truth "
                "required tool names and cannot be inferred from trajectory alone"
            )
        hits = called & expected_names
        score = len(hits) / len(expected_names)
        return {
            "score": float(score),
            "label": "complete" if score >= 1.0 else "missed",
            "explanation": (
                f"called={sorted(called)}; expected={sorted(expected_names)}; "
                f"hit={sorted(hits)}; recall={score:.3f}"
            ),
        }

    tool_recall.__name__ = "tool_recall"
    tool_recall.__qualname__ = "tool_recall"
    return tool_recall


def _tool_selection_str(output: dict[str, Any]) -> str:
    output_dict = _as_dict(output)
    full = output_dict.get("tool_calls_full") or []
    if full:
        return _json_dumps(
            [
                {
                    "name": _as_dict(call).get("name"),
                    "arguments": _as_dict(call).get("arguments") or {},
                }
                for call in full
            ],
            limit=6000,
        )
    return _json_dumps(output_dict.get("tool_calls") or [], limit=2000)


def _task_execution_evidence(output: dict[str, Any]) -> str:
    return _json_dumps(
        {
            "resolved": output.get("resolved"),
            "status": output.get("status"),
            "error": output.get("error"),
            "swe_status": output.get("swe_status"),
            "turns": output.get("turns"),
        },
        limit=4000,
    )


def make_tool_selection(llm: LLM) -> Callable[..., dict[str, Any]]:
    from a2e.evals.metrics import ToolSelectionEvaluator

    return _dual_mode(
        metric_name="tool_selection",
        build_structured=lambda llm_: ToolSelectionEvaluator(llm=llm_),
        structured_input=lambda output, expected, input_: {
            "input": _instruction(input_),
            "available_tools": _available_tools_str(input_),
            "tool_selection": _tool_selection_str(output),
        },
        text_definition=(
            "Were the selected tools appropriate for the task, given the available tool menu "
            "and the visible trajectory?"
        ),
        choices=("correct", "incorrect"),
        positive="correct",
        text_context=lambda output, expected, input_: {
            "Question": _instruction(input_),
            "Available tools": _available_tools_str(input_),
            "Tool selection": _tool_selection_str(output),
            "Task execution evidence": _task_execution_evidence(output),
        },
        llm=llm,
    )


def _tool_invocation_records(output: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return argument-bearing invocations, [] if none, None if names exist without args."""
    output_dict = _as_dict(output)
    records: list[dict[str, Any]] = []
    for call in output_dict.get("tool_calls_full") or []:
        item = _as_dict(call)
        name = item.get("name") or item.get("tool") or item.get("action")
        if not name:
            continue
        arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("args")
        records.append({"name": str(name), "arguments": _as_dict(arguments)})
    if records:
        return records
    for call in output_dict.get("tool_call_records") or []:
        item = _as_dict(call)
        name = item.get("name") or item.get("tool") or item.get("action")
        if not name:
            continue
        arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("args")
        if arguments is None:
            continue
        records.append({"name": str(name), "arguments": _as_dict(arguments)})
    if records:
        return records
    if output_dict.get("tool_calls"):
        return None
    return []


def _schema_has_parameters(schema: Mapping[str, Any]) -> bool:
    if schema.get("parameters") or schema.get("input_schema") or schema.get("properties"):
        return True
    function = _as_dict(schema.get("function"))
    return bool(function.get("parameters") or function.get("input_schema"))


def _tool_item_has_parameter_schema(item: Any) -> bool:
    item_dict = _as_dict(item)
    tool = item_dict.get("tool") if isinstance(item_dict.get("tool"), Mapping) else item_dict
    tool_dict = _as_dict(tool)
    if _schema_has_parameters(tool_dict):
        return True
    raw_schema = tool_dict.get("json_schema")
    if isinstance(raw_schema, str):
        try:
            raw_schema = json.loads(raw_schema)
        except json.JSONDecodeError:
            raw_schema = {}
    return isinstance(raw_schema, Mapping) and _schema_has_parameters(raw_schema)


def _tool_schema_payload(item: Any) -> dict[str, Any] | None:
    item_dict = _as_dict(item)
    tool = item_dict.get("tool") if isinstance(item_dict.get("tool"), Mapping) else item_dict
    tool_dict = _as_dict(tool)
    raw_schema = tool_dict.get("json_schema")
    if isinstance(raw_schema, str):
        try:
            raw_schema = json.loads(raw_schema)
        except json.JSONDecodeError:
            raw_schema = {}
    payload = dict(raw_schema) if isinstance(raw_schema, Mapping) and raw_schema else dict(tool_dict)
    name = _tool_schema_name(item)
    if name and not payload.get("name"):
        payload["name"] = name
    return payload or None


def _parameter_schemas_from_input(input_: dict[str, Any]) -> list[Any]:
    tools = _as_dict(input_).get("available_tools") or []
    if not isinstance(tools, list):
        return [tools] if _tool_item_has_parameter_schema(tools) else []
    return [item for item in tools if _tool_item_has_parameter_schema(item)]


_TERMINAL_BENCH_TOOL_NAMES = frozenset({"bash", "str_replace_editor"})


def _parse_tool_span_arguments(attrs: Mapping[str, Any]) -> dict[str, Any]:
    tool = _as_dict(attrs.get("tool"))
    parameters = tool.get("parameters")
    if isinstance(parameters, Mapping):
        return dict(parameters)
    input_attr = _as_dict(attrs.get("input"))
    raw_value = input_attr.get("value")
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return {}
        return _as_dict(parsed) if isinstance(parsed, Mapping) else {}
    return _as_dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _json_schema_type_for_value(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _parameters_schema_from_argument_keys(
    argument_keys: set[str],
    *,
    sample_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    properties = {
        key: {"type": _json_schema_type_for_value(sample_arguments.get(key))}
        for key in sorted(argument_keys)
    }
    return {"type": "object", "properties": properties}


def _parameter_schemas_from_tool_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Infer tool parameter schemas from TOOL span payloads when LLM spans omit tools."""
    by_name: dict[str, dict[str, Any]] = {}
    argument_keys: dict[str, set[str]] = {}
    argument_samples: dict[str, dict[str, Any]] = {}
    descriptions: dict[str, str] = {}

    for span in spans:
        if str(span.get("span_kind") or "").upper() != "TOOL":
            continue
        attrs = _as_dict(span.get("attributes"))
        tool = _as_dict(attrs.get("tool"))
        name = str(tool.get("name") or attrs.get("tool.name") or span.get("name") or "")
        if not name:
            continue
        description = str(tool.get("description") or "").strip()
        if description:
            descriptions[name] = description

        if _schema_has_parameters(tool):
            payload = _tool_schema_payload({"tool": tool}) or dict(tool)
            by_name[name] = payload
            continue

        raw_schema = tool.get("json_schema")
        if isinstance(raw_schema, str):
            try:
                raw_schema = json.loads(raw_schema)
            except json.JSONDecodeError:
                raw_schema = None
        if isinstance(raw_schema, Mapping) and _schema_has_parameters(raw_schema):
            by_name[name] = dict(raw_schema)
            continue

        arguments = _parse_tool_span_arguments(attrs)
        if not arguments:
            continue
        keys = argument_keys.setdefault(name, set())
        samples = argument_samples.setdefault(name, {})
        for key, value in arguments.items():
            keys.add(str(key))
            samples.setdefault(str(key), value)

    for name, keys in argument_keys.items():
        if name in by_name or not keys:
            continue
        parameters = _parameters_schema_from_argument_keys(keys, sample_arguments=argument_samples[name])
        function: dict[str, Any] = {"name": name, "parameters": parameters}
        if descriptions.get(name):
            function["description"] = descriptions[name]
        by_name[name] = {"type": "function", "function": function}

    return list(by_name.values())


def _parameter_schemas_from_llm_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for span in spans:
        attrs = _as_dict(span.get("attributes"))
        llm = _as_dict(attrs.get("llm"))
        for item in llm.get("tools") or []:
            if not _tool_item_has_parameter_schema(item):
                continue
            payload = _tool_schema_payload(item)
            name = _tool_schema_name(item)
            if payload and name and name not in by_name:
                by_name[name] = payload
    return list(by_name.values())


def _parameter_schemas_from_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for payload in (
        *_parameter_schemas_from_llm_spans(spans),
        *_parameter_schemas_from_tool_spans(spans),
    ):
        name = _tool_schema_name(payload) or _as_dict(payload.get("function")).get("name")
        if name and name not in by_name:
            by_name[str(name)] = payload
    return list(by_name.values())


def _tool_schema_menu_str(input_: dict[str, Any], spans: Sequence[Mapping[str, Any]]) -> str:
    from_input = _parameter_schemas_from_input(input_)
    if from_input:
        return _json_dumps(from_input, limit=8000)
    from_spans = _parameter_schemas_from_spans(spans)
    if from_spans:
        return _json_dumps(from_spans, limit=8000)
    return ""


def make_tool_invocation(
    llm: LLM,
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Score argument/schema validity of actual tool calls.

    Unscored when there are no invocations, when only tool names were kept
    (no argument payloads), or when no parameter schemas are available.
    Does not score tool selection, call count, or final-answer correctness.
    """
    from a2e.evals.metrics import ToolInvocationEvaluator

    spans_map = spans_by_example_id or {}
    inner = _dual_mode(
        metric_name="tool_invocation",
        build_structured=lambda llm_: ToolInvocationEvaluator(llm=llm_),
        structured_input=lambda output, expected, input_: {
            "input": _instruction(input_) + "\n\nVisible state:\n" + _initial_state_str(input_),
            "available_tools": str(input_.get("_tool_schema_menu") or _available_tools_str(input_)),
            "tool_selection": _tool_selection_str(output),
        },
        text_definition=(
            "Score ONLY actual tool invocations against the provided schemas. "
            "correct = every call has valid format, required fields, schema-legal arguments, "
            "values grounded in the user request / visible state, and no unsafe argument content. "
            "incorrect = any call is malformed, missing required fields, uses invented fields, "
            "has ungrounded values, or puts unsafe content in arguments. "
            "Do not judge whether the chosen tool was the right tool, how many calls were made, "
            "or whether the final answer is correct."
        ),
        choices=("correct", "incorrect"),
        positive="correct",
        text_context=lambda output, expected, input_: {
            "Question": _instruction(input_),
            "Visible state": _initial_state_str(input_),
            "Available tool schemas": str(
                input_.get("_tool_schema_menu") or _available_tools_str(input_)
            ),
            "Tool invocations": _tool_selection_str(output),
            "Tool history": _tool_history_block(output),
        },
        llm=llm,
    )

    def tool_invocation(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        spans = spans_map.get(example_id, [])
        enriched = _enrich_output(output, spans)
        records = _tool_invocation_records(enriched)
        if records is None:
            return _unscored(
                "Tool names were recorded without argument payloads; "
                "tool_invocation cannot score parameter validity."
            )
        if not records:
            return _unscored("No tool invocations observed; tool_invocation is not applicable.")
        schema_menu = _tool_schema_menu_str(input, spans)
        if not schema_menu:
            return _unscored(
                "No parameter schemas were recorded on the task or LLM spans; "
                "cannot score argument/schema validity."
            )
        patched_input = dict(_as_dict(input))
        patched_input["_tool_schema_menu"] = schema_menu
        return inner(enriched, expected, patched_input)

    tool_invocation.__name__ = "tool_invocation"
    tool_invocation.__qualname__ = "tool_invocation"
    return tool_invocation

def make_tool_call_count(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    def tool_call_count(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_by_example_id.get(example_id, []))
        count = len(enriched.get("tool_calls") or enriched.get("tool_calls_full") or [])
        if count == 0:
            label = "none"
        elif count < 3:
            label = "low"
        elif count < 6:
            label = "medium"
        else:
            label = "high"
        return {"score": float(count), "label": label, "explanation": f"{count} tool call(s)"}

    tool_call_count.__name__ = "tool_call_count"
    tool_call_count.__qualname__ = "tool_call_count"
    return tool_call_count


def _tool_schema_name(item: Any) -> str | None:
    item_dict = _as_dict(item)
    tool = item_dict.get("tool") if isinstance(item_dict.get("tool"), Mapping) else item_dict
    tool_dict = _as_dict(tool)
    raw_schema = tool_dict.get("json_schema")
    schema: dict[str, Any] = {}
    if isinstance(raw_schema, str):
        try:
            schema = json.loads(raw_schema)
        except json.JSONDecodeError:
            schema = {}
    elif isinstance(raw_schema, Mapping):
        schema = dict(raw_schema)
    function = _as_dict(schema.get("function"))
    return str(schema.get("name") or function.get("name") or tool_dict.get("name") or item_dict.get("name") or "") or None


def _available_tool_names_from_spans(spans: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for payload in _parameter_schemas_from_spans(spans):
        name = _tool_schema_name(payload) or _as_dict(payload.get("function")).get("name")
        if name:
            names.add(str(name))
    if names:
        return names
    for span in spans:
        attrs = _as_dict(span.get("attributes"))
        llm = _as_dict(attrs.get("llm"))
        for item in llm.get("tools") or []:
            name = _tool_schema_name(item)
            if name:
                names.add(name)
    called = set(_actual_tool_names_from_spans(spans))
    if called and called <= _TERMINAL_BENCH_TOOL_NAMES:
        return set(_TERMINAL_BENCH_TOOL_NAMES)
    return names


def _actual_tool_names_from_spans(spans: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for span in spans:
        attrs = _as_dict(span.get("attributes"))
        kind = str(span.get("span_kind") or _as_dict(_as_dict(attrs.get("openinference")).get("span")).get("kind") or "")
        if kind.upper() != "TOOL":
            continue
        tool = _as_dict(attrs.get("tool"))
        name = str(tool.get("name") or attrs.get("tool.name") or span.get("name") or "")
        if name:
            names.append(name)
    return names


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))


def _tool_output_for_identity(result: Any) -> Any:
    """Compare the tool payload, not OpenInference {mime_type, value} envelopes."""
    payload: Any = result
    if isinstance(payload, Mapping) and "value" in payload and (
        "mime_type" in payload or "mimeType" in payload
    ):
        payload = payload.get("value")
    if isinstance(payload, str):
        text = payload.strip()
        if text[:1] in "{[" and text[-1:] in "}]":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return payload
    return payload


def _loop_call_record(call: Any) -> dict[str, Any] | None:
    item = _as_dict(call)
    name = str(item.get("name") or item.get("tool") or item.get("action") or "").strip()
    if not name:
        return None
    if "arguments" not in item and "args" not in item:
        return None
    arguments = item.get("arguments")
    if arguments is None:
        arguments = item.get("args")
    result = item.get("result")
    if result is None and "output" in item:
        result = item.get("output")
    return {
        "name": name,
        "arguments": _as_dict(arguments) if isinstance(arguments, Mapping) else arguments,
        "output": _tool_output_for_identity(result),
    }


def _action_fingerprint(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("name") or ""), _canonical_json(record.get("arguments"))


_DIGIT_RE = re.compile(r"\d+")


def _near_fingerprint(record: Mapping[str, Any], *, min_skeleton: int = 40) -> tuple[str, str]:
    """Same tool + trivially varied args (digits in a long command) share a locator key."""
    name = str(record.get("name") or "")
    raw = _canonical_json(record.get("arguments"))
    skeleton = _DIGIT_RE.sub("#", raw)
    compact = re.sub(r"\s+", "", skeleton)
    if len(compact) >= min_skeleton:
        return name, skeleton
    return name, raw


def _collapse_monitor_clones(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop consecutive OpenInference clones that share name, args, and output."""
    collapsed: list[dict[str, Any]] = []
    prev: tuple[str, str, str] | None = None
    for record in records:
        key = (
            str(record.get("name") or ""),
            _canonical_json(record.get("arguments")),
            _canonical_json(record.get("output")),
        )
        if key == prev:
            continue
        collapsed.append(dict(record))
        prev = key
    return collapsed


def _merge_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: (int(item["start"]), int(item["end"])))
    merged: list[dict[str, Any]] = [dict(ordered[0])]
    for window in ordered[1:]:
        last = merged[-1]
        if int(window["start"]) <= int(last["end"]):
            last["end"] = max(int(last["end"]), int(window["end"]))
            last["kind"] = f"{last.get('kind')}|{window.get('kind')}"
        else:
            merged.append(dict(window))
    return merged


def _locate_loop_windows(
    records: Sequence[Mapping[str, Any]],
    *,
    min_streak: int = 3,
    min_cycle: int = 4,
    max_windows: int = 3,
) -> list[dict[str, Any]]:
    """CODE locator on tool calls only. Does not decide whether a window is a dead loop."""
    fingerprints = [_near_fingerprint(record) for record in records]
    windows: list[dict[str, Any]] = []
    idx = 0
    while idx < len(fingerprints):
        current = fingerprints[idx]
        nxt = idx + 1
        while nxt < len(fingerprints) and fingerprints[nxt] == current:
            nxt += 1
        if nxt - idx >= min_streak:
            windows.append(
                {
                    "kind": "consecutive_equivalent_action",
                    "start": idx,
                    "end": nxt,
                }
            )
        idx = nxt

    cycle_idx = 0
    while cycle_idx + min_cycle - 1 < len(fingerprints):
        first = fingerprints[cycle_idx]
        second = fingerprints[cycle_idx + 1]
        if (
            first
            and second
            and first != second
            and fingerprints[cycle_idx + 2] == first
            and fingerprints[cycle_idx + 3] == second
        ):
            end = cycle_idx + 4
            while (
                end + 1 < len(fingerprints)
                and fingerprints[end] == first
                and fingerprints[end + 1] == second
            ):
                end += 2
            windows.append({"kind": "alternating_cycle", "start": cycle_idx, "end": end})
            cycle_idx = end
            continue
        cycle_idx += 1

    return _merge_windows(windows)[:max_windows]


def _window_indices(window: Mapping[str, Any], n_records: int, *, cap: int = 6) -> list[int]:
    start = max(0, int(window["start"]))
    end = min(n_records, int(window["end"]))
    span = list(range(start, end))
    if len(span) <= cap:
        return span
    head = cap // 2
    tail = cap - head
    return span[:head] + span[-tail:]


def _render_loop_windows(
    records: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> str:
    blocks: list[str] = []
    for order, window in enumerate(windows, start=1):
        indices = _window_indices(window, len(records))
        lines = [
            f"Window {order} [{window.get('kind')}] tool steps {window['start']}..{int(window['end']) - 1} "
            f"(showing {len(indices)}/{int(window['end']) - int(window['start'])} TOOL calls):"
        ]
        for idx in indices:
            record = records[idx]
            args = _json_dumps(record.get("arguments"), limit=240)
            output = _json_dumps(record.get("output"), limit=240)
            lines.append(f"  [{idx}] {record.get('name')}({args}) -> {output}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _looping_tool_count(
    records: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> int:
    names: set[str] = set()
    for window in windows:
        start = max(0, int(window["start"]))
        end = min(len(records), int(window["end"]))
        for idx in range(start, end):
            name = str(records[idx].get("name") or "").strip()
            if name:
                names.add(name)
    return len(names)


_LOOP_JUDGE_DEFINITION = (
    "A dead loop is a stretch of TOOL calls that repeats equivalent actions without information gain "
    "or task progress (RedundancyBench Repeated Tool Call / WebArena repeating-equivalent-action). "
    "Equivalent means the same tool with the same or trivially-varied arguments, not merely the same "
    "tool name. Judge ONLY the candidate TOOL windows; ignore the rest of the trajectory. "
    "LABEL=looping if at least one window is a dead loop, even if the run later succeeded or timed out. "
    "LABEL=clean if every window is justified (retry after error, polling with changing state, or "
    "finishing an incomplete previous result)."
)


def make_repeated_tool_call_rate(
    llm: LLM,
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Locate candidate TOOL-call windows with CODE, then one LLM judge on those windows."""

    spans_map = spans_by_example_id or {}

    def repeated_tool_call_rate(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        del expected
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_map.get(example_id, []))
        raw_calls = [_as_dict(call) for call in (enriched.get("tool_calls_full") or [])]
        records = [record for record in (_loop_call_record(call) for call in raw_calls) if record]
        records = _collapse_monitor_clones(records)
        if not records:
            if raw_calls or (enriched.get("tool_calls") or []):
                return _unscored(
                    "Tool calls were observed, but loop localization needs name and arguments; "
                    "cannot build candidate TOOL windows."
                )
            return _unscored(
                "No tool invocations were observed in the trajectory; repeated_tool_call_rate is not applicable."
            )

        windows = _locate_loop_windows(records)
        n_calls = len(records)
        if not windows:
            return {
                "score": 0.0,
                "label": "clean",
                "explanation": (
                    f"CODE locator found no repeated/near-repeated TOOL windows in {n_calls} "
                    "call(s); looping-tool count=0; LLM was not invoked."
                ),
            }

        prompt = _build_text_prompt(
            metric_name="repeated_tool_call_rate",
            definition=_LOOP_JUDGE_DEFINITION,
            choices=("clean", "looping"),
            positive="clean",
            context={
                "Question": _instruction(input)[:800],
                "Locator": (
                    f"{len(windows)} TOOL-only candidate window(s) over {n_calls} tool calls. "
                    "Localization hints only, not the verdict."
                ),
                "Candidate TOOL windows": _render_loop_windows(records, windows),
            },
        )
        judged = _text_judge(llm, prompt, ("clean", "looping"), "clean")
        if judged.get("label") == "unscored":
            return judged
        if judged.get("label") == "clean":
            return {
                "score": 0.0,
                "label": "clean",
                "explanation": (
                    f"LLM judged TOOL windows as justified (not a dead loop); looping-tool count=0. "
                    f"{judged.get('explanation') or ''}"
                )[:1000],
            }
        count = _looping_tool_count(records, windows)
        looping_names = sorted(
            {
                str(records[idx].get("name") or "")
                for window in windows
                for idx in range(int(window["start"]), int(window["end"]))
                if str(records[idx].get("name") or "")
            }
        )
        return {
            "score": float(count),
            "label": "looping",
            "explanation": (
                f"LLM confirmed a dead loop in {count} tool(s): {looping_names}. "
                f"{judged.get('explanation') or ''}"
            )[:1000],
        }

    repeated_tool_call_rate.__name__ = "repeated_tool_call_rate"
    repeated_tool_call_rate.__qualname__ = "repeated_tool_call_rate"
    return repeated_tool_call_rate


_EXECUTION_LABEL_PRIORITY = (
    "environment_dead",
    "timeout",
    "harness_reject",
    "exception",
    "runtime",
)


def _walk_result_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            found.append(dict(item))
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if isinstance(item, str):
            text = item.strip()
            if text[:1] in "{[":
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return
                visit(parsed)

    visit(value)
    return found


def _classify_execution_result(value: Any) -> str | None:
    """BFCL executable eval: None means effective execution; otherwise an error type.

    Follows BFCL (ICML 2025) Executable Function Evaluation: the environment must
    complete the invocation. ToolSandbox-style exceptions are typed from the observation.
    """
    blobs = _walk_result_dicts(value)
    for blob in blobs:
        err = blob.get("error")
        if isinstance(err, str) and err.strip():
            low = err.lower()
            if "container not started" in low:
                return "environment_dead"
            if "duplicate tool call" in low:
                return "harness_reject"
            if "timed out" in low or "timeout" in low:
                return "timeout"
            return "runtime"
        code = blob.get("exit_code")
        stderr = str(blob.get("stderr") or "")
        stderr_low = stderr.lower()
        if "traceback (most recent call last)" in stderr_low:
            return "exception"
        if "timed out" in stderr_low or re.search(r"\btimeout\b", stderr_low):
            return "timeout"
        if isinstance(code, bool) or code is None:
            continue
        try:
            exit_code = int(code)
        except (TypeError, ValueError):
            continue
        if exit_code != 0:
            if "timed out" in stderr_low or "timeout" in stderr_low:
                return "timeout"
            if "traceback" in stderr_low:
                return "exception"
            return "runtime"

    text = _json_dumps(value, limit=2000).lower()
    if "container not started" in text:
        return "environment_dead"
    if "duplicate tool call" in text:
        return "harness_reject"
    return None


def make_tool_execution_error_rate(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    """Count of tool calls the environment did not complete (BFCL executable failure)."""

    def tool_execution_error_rate(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        del expected, input
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_by_example_id.get(example_id, []))
        calls = [_as_dict(call) for call in (enriched.get("tool_calls_full") or [])]
        results: list[Any] = []
        for call in calls:
            result = call.get("result")
            if result is None:
                result = call.get("output")
            if result is not None:
                results.append(result)
        if not results:
            if calls or (enriched.get("tool_calls") or []):
                return _unscored(
                    "Tool calls were observed, but execution scoring needs tool outputs; "
                    "cannot apply BFCL executable evaluation."
                )
            return _unscored(
                "No tool invocations were observed in the trajectory; "
                "tool_execution_error_rate is not applicable."
            )

        types = [_classify_execution_result(result) for result in results]
        errors = [kind for kind in types if kind]
        n_calls = len(results)
        n_error = len(errors)
        if n_error == 0:
            label = "clean"
        else:
            counts = {kind: errors.count(kind) for kind in _EXECUTION_LABEL_PRIORITY}
            label = max(
                _EXECUTION_LABEL_PRIORITY,
                key=lambda kind: (counts[kind], -_EXECUTION_LABEL_PRIORITY.index(kind)),
            )
        type_bits = ", ".join(
            f"{kind}={errors.count(kind)}" for kind in _EXECUTION_LABEL_PRIORITY if errors.count(kind)
        )
        return {
            "score": float(n_error),
            "label": label,
            "explanation": (
                f"{n_error} tool call(s) did not complete out of {n_calls} observed "
                f"(0=none; each failed invocation counts as 1"
                f"{'; ' + type_bits if type_bits else ''})."
            )[:1000],
        }

    tool_execution_error_rate.__name__ = "tool_execution_error_rate"
    tool_execution_error_rate.__qualname__ = "tool_execution_error_rate"
    return tool_execution_error_rate


def _is_error_result(value: Any) -> bool:
    text = _json_dumps(value, limit=1200).lower()
    markers = ("error", "exception", "traceback", "failed", "failure", "not found", "invalid", "timeout")
    return any(marker in text for marker in markers)


def make_self_correction_rate(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    def self_correction_rate(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_by_example_id.get(example_id, []))
        calls = [_as_dict(call) for call in (enriched.get("tool_calls_full") or [])]
        error_indices = [idx for idx, call in enumerate(calls) if _is_error_result(call.get("result"))]
        if not error_indices:
            return {
                "score": 1.0,
                "label": "no_tool_errors",
                "explanation": (
                    "No tool error outputs were observed; self-correction not required (score=1.0)."
                ),
            }

        corrected = 0
        for idx in error_indices:
            name = calls[idx].get("name")
            if any(call.get("name") == name and not _is_error_result(call.get("result")) for call in calls[idx + 1 :]):
                corrected += 1
        score = corrected / len(error_indices)
        return {
            "score": float(score),
            "label": "corrected" if score >= 1.0 else "uncorrected",
            "explanation": (
                f"{corrected} corrected error(s) / {len(error_indices)} observed tool error(s); "
                f"score={score:.3f}"
            ),
        }

    self_correction_rate.__name__ = "self_correction_rate"
    self_correction_rate.__qualname__ = "self_correction_rate"
    return self_correction_rate
