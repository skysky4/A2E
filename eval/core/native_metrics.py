"""Prefer official / trajectory-native metric values over post-hoc eval when they conflict."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.eval_common import UNSCORED_LABEL, _as_dict, _is_unscored, _task_output

# Nested bags on task output that may carry harness / benchmark scores.
_METRIC_BAG_KEYS = (
    "metrics",
    "eval_metrics",
    "benchmark_metrics",
    "official_metrics",
    "upstream_metrics",
    "scores",
    "annotations",
)

# Token / cost keys inside metric bags — never map to correctness-like metrics.
_EFFICIENCY_ONLY_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
        "cost_usd",
        "total_steps",
    }
)

# Eval metric -> equivalent field names that may appear on imported trajectories.
NATIVE_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "correctness": (
        "correctness",
        "accuracy",
        "exact_match",
        "numeric_match",
        "mc_letter",
        "resolved",
        "tb_resolved",
        "swe_resolved",
    ),
    "task_completion": ("task_succeeded", "task_completion"),
    "submitted": ("submitted", "delivery_outcome"),
    "execution_completion": ("execution_completion", "completed"),
    "error_absence": ("error_absence",),
    "tool_recall": ("tool_recall",),
    "tool_call_count": ("tool_call_count",),
    "tool_execution_error_rate": ("tool_execution_error_rate",),
    "repeated_tool_call_rate": ("repeated_tool_call_rate",),
    "redcode_risky_operation_count": ("redcode_risky_operation_count",),
    "self_correction_rate": ("self_correction_rate",),
    "total_token_usage": ("total_token_usage", "total_tokens"),
    "cost": ("cost", "total_cost_usd", "cost_usd"),
    "turn_count": ("turn_count", "turns", "total_steps"),
    "idle_turn_count": ("idle_turn_count",),
    "wall_time": ("wall_time", "elapsed_time"),
}

# Task-runner / harness evaluator names treated as official upstream scores.
OFFICIAL_UPSTREAM_EVALUATORS = frozenset(
    {
        "exact_match",
        "substring",
        "numeric_match",
        "mc_letter",
        "tool_recall",
        "swe_resolved",
        "tb_resolved",
        "swe_fail_to_pass",
        "swe_pass_to_pass",
        "llm_judge",
    }
)

# Map upstream evaluator annotation names onto V2 eval metrics.
UPSTREAM_EVALUATOR_TO_METRIC: dict[str, str] = {
    "exact_match": "correctness",
    "numeric_match": "correctness",
    "mc_letter": "correctness",
    "tb_resolved": "correctness",
    "swe_resolved": "correctness",
    "llm_judge": "correctness",
    "tool_recall": "tool_recall",
}

_SCORE_TOLERANCE = 1e-9


def _metric_bags(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    bags: list[dict[str, Any]] = []
    root = _as_dict(output)
    task = _task_output(output)
    for container in (task, root):
        for key in _METRIC_BAG_KEYS:
            nested = _as_dict(container.get(key))
            if nested:
                bags.append(nested)
        final_metrics = _as_dict(container.get("final_metrics"))
        if final_metrics:
            bags.append(final_metrics)
    return bags


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "pass", "passed", "resolved"}:
        return True
    if text in {"false", "0", "no", "fail", "failed", "unresolved"}:
        return False
    return None


def _normalize_raw_native_value(metric_name: str, raw: Any, *, source_key: str) -> dict[str, Any] | None:
    if raw is None:
        return None

    if isinstance(raw, Mapping):
        raw_dict = dict(raw)
        if raw_dict.get("label") == UNSCORED_LABEL or raw_dict.get("score") is None and raw_dict.get("label") == UNSCORED_LABEL:
            return None
        score = raw_dict.get("score")
        label = raw_dict.get("label")
        explanation = str(raw_dict.get("explanation") or "")[:1000]
        if score is not None:
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                return None
            metadata = {
                "evaluation_method": f"native:{source_key}",
                "native_source_key": source_key,
            }
            if isinstance(raw_dict.get("metadata"), Mapping):
                metadata.update(dict(raw_dict["metadata"]))
            return {
                "score": score_f,
                "label": str(label or ""),
                "explanation": explanation or f"Used trajectory-native {source_key} score.",
                "metadata": metadata,
            }
        if label is not None:
            as_bool = _coerce_bool(label)
            if as_bool is not None:
                return _normalize_raw_native_value(metric_name, as_bool, source_key=source_key)

    if isinstance(raw, bool):
        if metric_name == "correctness":
            return {
                "score": 1.0 if raw else 0.0,
                "label": "correct" if raw else "incorrect",
                "explanation": f"Used trajectory-native {source_key}={raw!r}.",
                "metadata": {
                    "evaluation_method": f"native:{source_key}",
                    "native_source_key": source_key,
                },
            }
        return {
            "score": 1.0 if raw else 0.0,
            "label": "pass" if raw else "fail",
            "explanation": f"Used trajectory-native {source_key}={raw!r}.",
            "metadata": {
                "evaluation_method": f"native:{source_key}",
                "native_source_key": source_key,
            },
        }

    if isinstance(raw, (int, float)):
        score_f = float(raw)
        if metric_name == "correctness":
            if score_f in (0.0, 1.0):
                return {
                    "score": score_f,
                    "label": "correct" if score_f >= 0.5 else "incorrect",
                    "explanation": f"Used trajectory-native {source_key}={score_f}.",
                    "metadata": {
                        "evaluation_method": f"native:{source_key}",
                        "native_source_key": source_key,
                    },
                }
            return None
        if score_f < 0:
            return None
        return {
            "score": score_f,
            "label": str(source_key),
            "explanation": f"Used trajectory-native {source_key}={score_f}.",
            "metadata": {
                "evaluation_method": f"native:{source_key}",
                "native_source_key": source_key,
            },
        }

    return None


def _alias_keys(metric_name: str) -> tuple[str, ...]:
    keys = [metric_name]
    keys.extend(NATIVE_METRIC_ALIASES.get(metric_name, ()))
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return tuple(deduped)


def _lookup_native_raw(metric_name: str, output: Mapping[str, Any]) -> tuple[Any, str] | None:
    aliases = _alias_keys(metric_name)
    task = _task_output(output)
    root = _as_dict(output)

    for bag in _metric_bags(output):
        for key in aliases:
            if key in _EFFICIENCY_ONLY_KEYS:
                continue
            if key in bag and bag.get(key) is not None:
                return bag[key], key

    for key in aliases:
        if key in _EFFICIENCY_ONLY_KEYS:
            continue
        if key in task and task.get(key) is not None:
            return task[key], key
        if key in root and root.get(key) is not None:
            return root[key], key

    if metric_name == "total_token_usage":
        for bag in _metric_bags(output):
            prompt = bag.get("total_prompt_tokens")
            completion = bag.get("total_completion_tokens")
            if prompt is not None and completion is not None:
                try:
                    return float(prompt) + float(completion), "total_prompt_tokens+total_completion_tokens"
                except (TypeError, ValueError):
                    pass

    return None


def extract_native_metric(metric_name: str, output: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized eval result dict if the trajectory carries an official score."""
    found = _lookup_native_raw(metric_name, output)
    if found is None:
        return None
    raw, source_key = found
    return _normalize_raw_native_value(metric_name, raw, source_key=source_key)


def _scores_conflict(native: Mapping[str, Any], computed: Mapping[str, Any]) -> bool:
    if _is_unscored(native) or _is_unscored(computed):
        return False
    native_score = native.get("score")
    computed_score = computed.get("score")
    if native_score is not None and computed_score is not None:
        try:
            return abs(float(native_score) - float(computed_score)) > _SCORE_TOLERANCE
        except (TypeError, ValueError):
            pass
    return str(native.get("label") or "") != str(computed.get("label") or "")


def prefer_trajectory_native_metric(
    metric_name: str,
    compute_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]],
    output: dict[str, Any],
    expected: dict[str, Any],
    input_: dict[str, Any],
) -> dict[str, Any]:
    """Compute eval score, but prefer trajectory-native official values on conflict."""
    native = extract_native_metric(metric_name, output)
    computed = compute_fn(output, expected, input_)
    if native is None:
        return computed
    if _is_unscored(computed) or _scores_conflict(native, computed):
        explanation = str(native.get("explanation") or "")
        if _scores_conflict(native, computed) and not _is_unscored(computed):
            computed_score = computed.get("score")
            native_score = native.get("score")
            explanation = (
                f"{explanation} Overrode post-hoc eval ({computed_score!r}) "
                f"with trajectory-native official score ({native_score!r})."
            )[:1000]
            native = dict(native)
            native["explanation"] = explanation
        return native
    return computed


def _annotation_result_to_native(raw: Mapping[str, Any]) -> Any:
    if "score" in raw or "label" in raw:
        return dict(raw)
    result = raw.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    return raw


def merge_upstream_eval_annotations(experiment: Mapping[str, Any]) -> None:
    """Copy official task-runner annotations into run output.metrics (does not overwrite)."""
    upstream_by_run: dict[str, dict[str, Any]] = {}

    for eval_run in experiment.get("evaluation_runs", []):
        if isinstance(eval_run, Mapping):
            run_id = str(eval_run.get("experiment_run_id") or "")
            name = str(eval_run.get("name") or "")
            result = eval_run.get("result")
        else:
            run_id = str(getattr(eval_run, "experiment_run_id", "") or "")
            name = str(getattr(eval_run, "name", "") or "")
            result = getattr(eval_run, "result", None)

        if not run_id or name not in OFFICIAL_UPSTREAM_EVALUATORS or result is None:
            continue

        target_metric = UPSTREAM_EVALUATOR_TO_METRIC.get(name, name)
        payload = _annotation_result_to_native(_as_dict(result))
        bucket = upstream_by_run.setdefault(run_id, {})
        bucket.setdefault(target_metric, payload)
        bucket.setdefault(name, payload)

    for run in experiment.get("task_runs", []):
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("id") or "")
        if not run_id or run_id not in upstream_by_run:
            continue

        output = dict(_as_dict(run.get("output")))
        task_output = _task_output(output)
        metrics = dict(_as_dict(task_output.get("metrics")))
        if not metrics and isinstance(output.get("metrics"), Mapping):
            metrics = dict(_as_dict(output.get("metrics")))

        for key, value in upstream_by_run[run_id].items():
            metrics.setdefault(key, value)

        if metrics:
            if isinstance(output.get("task_output"), Mapping):
                task_output = dict(_task_output(output))
                task_output["metrics"] = metrics
                output["task_output"] = task_output
            else:
                output["metrics"] = metrics
            run["output"] = output


def with_native_metric_preference(
    metric_name: str,
    base_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Wrap an evaluator so trajectory-native scores win on conflict."""

    def wrapped(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        def compute(o: dict[str, Any], e: dict[str, Any], i: dict[str, Any]) -> dict[str, Any]:
            if example is not None:
                try:
                    return base_fn(o, e, i, example=example)
                except TypeError:
                    pass
            return base_fn(o, e, i)

        return prefer_trajectory_native_metric(metric_name, compute, output, expected, input)

    wrapped.__name__ = metric_name
    wrapped.__qualname__ = metric_name
    return wrapped
