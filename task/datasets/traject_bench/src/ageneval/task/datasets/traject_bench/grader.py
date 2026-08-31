"""Unofficial best-effort local grading for TRAJECT-Bench.

The upstream benchmark's full trajectory judge is not reproduced here.  This
module combines expected tool-action correctness with an expected final-answer
match when those references are available and always reports ``official=False``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ageneval.task.core.grading import GraderSpec


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _answer(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("final_answer", value.get("answer", ""))
    text = str(value or "").strip()
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, Mapping):
            text = str(decoded.get("final_answer", decoded.get("answer", text))).strip()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.;").casefold()


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _normalize_arguments(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, Mapping) else {}


def _tool_call(value: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, str):
        return value, {}
    function = _field(value, "function")
    if function:
        name = _field(function, "name", "")
        arguments = _field(function, "arguments", {})
    else:
        name = _field(value, "name", _field(value, "tool", ""))
        arguments = _field(value, "arguments", _field(value, "args", {}))
    return str(name or ""), _normalize_arguments(arguments)


def _arguments_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Treat expected arguments as required fields while allowing trace extras."""
    return all(key in actual and actual[key] == value for key, value in expected.items())


def _action_metrics(output: Any, expected: Any) -> dict[str, float] | None:
    expected_values = _as_sequence(_field(expected, "expected_actions", ()))
    if not expected_values:
        return None
    actual_values = _as_sequence(
        _field(output, "tool_call_records", _field(output, "tool_calls", ()))
    )
    expected_calls = [_tool_call(value) for value in expected_values]
    actual_calls = [_tool_call(value) for value in actual_values]

    unmatched = set(range(len(actual_calls)))
    matches = 0
    for expected_name, expected_arguments in expected_calls:
        for index in tuple(unmatched):
            actual_name, actual_arguments = actual_calls[index]
            if actual_name == expected_name and _arguments_match(actual_arguments, expected_arguments):
                matches += 1
                unmatched.remove(index)
                break

    recall = matches / len(expected_calls)
    precision = matches / len(actual_calls) if actual_calls else 0.0
    correctness = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tool_action_correctness": correctness,
        "tool_action_precision": precision,
        "tool_action_recall": recall,
    }


def _answer_metric(output: Any, expected: Any) -> float | None:
    references = _as_sequence(_field(expected, "expected_outputs", ()))
    references = [reference for reference in references if str(reference or "").strip()]
    if not references:
        return None
    predicted = _answer(output)
    return float(
        any(
            (reference_text := _answer(reference))
            and predicted
            and (predicted == reference_text or reference_text in predicted)
            for reference in references
        )
    )


def grade_traject_bench(
    output: Any,
    expected: Any,
    input: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine available action and answer references into an unofficial score."""
    del input
    action_metrics = _action_metrics(output, expected)
    answer_score = _answer_metric(output, expected)
    components: list[float] = []
    metrics: dict[str, float] = {}
    if action_metrics is not None:
        metrics.update(action_metrics)
        components.append(action_metrics["tool_action_correctness"])
    if answer_score is not None:
        metrics["answer_match"] = answer_score
        components.append(answer_score)

    source = str((metadata or {}).get("source") or "bigboss24/TRAJECT-Bench")
    if not components:
        return {
            "score": None,
            "passed": None,
            "label": "unsupported",
            "metrics": {},
            "metadata": {"status": "unsupported"},
            "explanation": "No expected actions or answer are available for local grading.",
            "official": False,
            "source": source,
            "version": "unofficial-combined-v1",
        }

    score = sum(components) / len(components)
    return {
        "score": score,
        "passed": score == 1.0,
        "metrics": metrics,
        "metadata": {"components": len(components)},
        "explanation": (
            "Unofficial mean of available expected-tool-action correctness and "
            "reference-answer match."
        ),
        "official": False,
        "source": source,
        "version": "unofficial-combined-v1",
    }


grade = grade_traject_bench
GRADER = GraderSpec(
    id="traject_grader",
    grade=grade_traject_bench,
    official=False,
    source="bigboss24/TRAJECT-Bench",
    version="unofficial-combined-v1",
    aliases=("tool_recall",),
)

__all__ = ["GRADER", "grade", "grade_traject_bench"]
