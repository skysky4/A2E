"""Efficiency metric evaluator logic."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from a2e.evals.llm import LLM

from core.eval_common import (
    _as_dict,
    _dual_mode,
    _elapsed_seconds_from_spans,
    _final_answer,
    _instruction,
    _sum_cost,
    _task_output,
    _unscored,
)
from core.trajectory_token_usage import (
    label_for_total_tokens,
    label_for_idle_turn_count,
    idle_turn_count,
    total_tokens_preferred,
)


def _count_answer_tokens(text: str) -> tuple[int, str]:
    if not text:
        return 0, "empty final_answer"
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text)), "tiktoken cl100k_base"
    except Exception:
        return max(1, len(text) // 4), "len(text)//4 fallback"


def make_total_token_usage(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    benchmark: str = "",
) -> Callable[..., dict[str, Any]]:
    def total_token_usage(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_dict = _as_dict(example)
        example_id = str(example_dict.get("id") or "")
        metadata = example_dict.get("metadata") or {}
        spans = spans_by_example_id.get(example_id, [])
        total, source = total_tokens_preferred(
            spans=spans,
            benchmark=benchmark,
            input_payload=input,
            output=output,
            example_metadata=metadata if isinstance(metadata, Mapping) else {},
        )
        if total <= 0:
            return _unscored(f"token usage is missing; {source}")
        return {
            "score": total,
            "label": label_for_total_tokens(total),
            "explanation": source,
        }

    total_token_usage.__name__ = "total_token_usage"
    total_token_usage.__qualname__ = "total_token_usage"
    return total_token_usage


def make_cost(spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> Callable[..., dict[str, Any]]:
    def cost(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        spans = spans_by_example_id.get(example_id, [])
        total, source = _sum_cost(spans)
        if total == 0.0:
            output_dict = _task_output(output)
            fallback = output_dict.get("cost") or output_dict.get("total_cost") or output_dict.get("cost_usd")
            if fallback is not None:
                try:
                    total = float(fallback)
                    source = "output cost field"
                except (TypeError, ValueError):
                    pass
        if total == 0:
            return _unscored(f"cost is missing; {source}")
        if total < 0.01:
            label = "low"
        elif total < 0.1:
            label = "medium"
        else:
            label = "high"
        return {"score": float(total), "label": label, "explanation": f"${total:.6f}; {source}"}

    cost.__name__ = "cost"
    cost.__qualname__ = "cost"
    return cost


def make_answer_cost() -> Callable[..., dict[str, Any]]:
    def answer_cost(output: dict[str, Any], expected: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        count, source = _count_answer_tokens(_final_answer(output))
        if count == 0:
            label = "empty"
        elif count < 50:
            label = "low"
        elif count < 200:
            label = "medium"
        else:
            label = "high"
        return {"score": float(count), "label": label, "explanation": f"{count} answer token(s); {source}"}

    answer_cost.__name__ = "answer_cost"
    answer_cost.__qualname__ = "answer_cost"
    return answer_cost


def make_turn_count() -> Callable[..., dict[str, Any]]:
    def turn_count(output: dict[str, Any], expected: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        output_dict = _task_output(output)
        raw = output_dict.get("turns")
        if raw is None:
            raw = output_dict.get("turn_count")
        if raw is None:
            return _unscored("turn_count is missing from task output")
        try:
            count = int(raw)
        except (TypeError, ValueError):
            return _unscored(f"turn_count is not numeric: {raw!r}")
        if count == 0:
            label = "zero"
        elif count < 3:
            label = "low"
        elif count < 8:
            label = "medium"
        else:
            label = "high"
        return {"score": float(count), "label": label, "explanation": f"{count} turn(s)"}

    turn_count.__name__ = "turn_count"
    turn_count.__qualname__ = "turn_count"
    return turn_count


def make_idle_turn_count() -> Callable[..., dict[str, Any]]:
    def idle_turn_count_metric(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
    ) -> dict[str, Any]:
        del expected, input
        count, source = idle_turn_count(output)
        if count is None:
            return _unscored(source)
        return {
            "score": float(count),
            "label": label_for_idle_turn_count(count),
            "explanation": source,
        }

    idle_turn_count_metric.__name__ = "idle_turn_count"
    idle_turn_count_metric.__qualname__ = "idle_turn_count"
    return idle_turn_count_metric


def make_wall_time(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Callable[..., dict[str, Any]]:
    spans_map = spans_by_example_id or {}

    def wall_time(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        output_dict = _task_output(output)
        raw = output_dict.get("elapsed_seconds") or output_dict.get("elapsed_time") or output_dict.get("duration")
        source = "task output"
        if raw is None:
            example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
            derived = _elapsed_seconds_from_spans(spans_map.get(example_id, []))
            if derived is not None:
                raw, source = derived
        if raw is None:
            return _unscored("wall_time is missing from task output and spans")
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            return _unscored(f"wall_time is not numeric: {raw!r}")
        if seconds <= 0:
            return _unscored(f"wall_time is not a positive duration: {seconds}")
        if seconds < 5:
            label = "fast"
        elif seconds < 30:
            label = "medium"
        else:
            label = "slow"
        return {
            "score": float(seconds),
            "label": label,
            "explanation": f"{seconds:.3f} second(s); {source}",
        }

    wall_time.__name__ = "wall_time"
    wall_time.__qualname__ = "wall_time"
    return wall_time


# Backward-compatible alias for older configs and notebooks.
make_elapsed_time = make_wall_time


def make_conciseness(llm: LLM) -> Callable[..., dict[str, Any]]:
    from a2e.evals.metrics import ConcisenessEvaluator

    return _dual_mode(
        metric_name="conciseness",
        build_structured=lambda llm_: ConcisenessEvaluator(llm=llm_),
        structured_input=lambda output, expected, input_: {
            "input": _instruction(input_),
            "output": _final_answer(output),
        },
        text_definition="Is the agent's output concise, with only necessary information?",
        choices=("concise", "verbose"),
        positive="concise",
        text_context=lambda output, expected, input_: {
            "Question": _instruction(input_),
            "Agent answer": _final_answer(output),
        },
        llm=llm,
    )
