"""Benchmark-owned grading contracts.

The task layer distinguishes a benchmark's primary grader from optional
platform evaluators.  A grader owns the benchmark score and provenance; the
adapter at the bottom of this module only bridges that score into the existing
``a2e-client`` experiment API.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

GraderMode = Literal["posthoc", "inline"]
GradeCallable = Callable[..., Any]
GraderFactory = Callable[[Any], GradeCallable]


@dataclass(frozen=True)
class GradeReport:
    """Normalized result produced by every benchmark grader."""

    score: float | None = None
    passed: bool | None = None
    label: str | None = None
    explanation: str | None = None
    metrics: Mapping[str, float | int | bool | None] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    official: bool = True
    source: str | None = None
    version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly shape used by runners and upload sinks."""
        label = self.label
        if label is None and self.passed is not None:
            label = "pass" if self.passed else "fail"
        return {
            "score": self.score,
            "passed": self.passed,
            "label": label,
            "explanation": self.explanation,
            "metrics": dict(self.metrics),
            "metadata": {
                **dict(self.metadata),
                "official": self.official,
                "source": self.source,
                "version": self.version,
            },
            "error": self.error,
        }


@dataclass(frozen=True)
class GraderSpec:
    """Description and resolver for one benchmark's primary grader."""

    id: str
    grade: GradeCallable | None = None
    summarize: GradeCallable | None = None
    factory: GraderFactory | None = None
    mode: GraderMode = "posthoc"
    official: bool = True
    source: str | None = None
    version: str | None = None
    required: bool = True
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, runtime: Any | None = None) -> GradeCallable:
        if self.factory is not None:
            if runtime is None:
                raise ValueError(f"grader {self.id!r} requires a grading runtime")
            return self.factory(runtime)
        if self.grade is None:
            raise ValueError(f"grader {self.id!r} has no callable")
        return self.grade

    def summarize_inline(self, value: Any) -> GradeReport:
        """Convert a live-environment report into the canonical grade shape."""
        if self.summarize is None:
            return normalize_grade(value, self)
        summarized = self.summarize(value)
        if inspect.isawaitable(summarized):
            raise TypeError("inline grader summarizers must be synchronous")
        return normalize_grade(summarized, self)


def normalize_grade(value: Any, spec: GraderSpec) -> GradeReport:
    """Normalize legacy scalar/dict grader outputs without changing semantics."""
    if isinstance(value, GradeReport):
        return value
    if isinstance(value, bool):
        return GradeReport(
            score=float(value),
            passed=value,
            metadata=dict(spec.metadata),
            official=spec.official,
            source=spec.source,
            version=spec.version,
        )
    if isinstance(value, (int, float)):
        score = float(value)
        return GradeReport(
            score=score,
            passed=score >= 1.0,
            metadata=dict(spec.metadata),
            official=spec.official,
            source=spec.source,
            version=spec.version,
        )
    if isinstance(value, Mapping):
        nested = value.get("result")
        result = nested if isinstance(nested, Mapping) else value
        score = result.get("score")
        passed = result.get("passed")
        if passed is None and isinstance(score, (int, float)):
            passed = float(score) >= 1.0
        metadata = value.get("metadata")
        metrics = value.get("metrics")
        metadata_dict = {
            **dict(spec.metadata),
            **(dict(metadata) if isinstance(metadata, Mapping) else {}),
        }
        if not isinstance(metrics, Mapping):
            metrics = metadata_dict.pop("metrics", {})
        if passed is None:
            passed = metadata_dict.pop("passed", None)
        return GradeReport(
            score=float(score) if isinstance(score, (int, float)) else None,
            passed=passed if isinstance(passed, bool) else None,
            label=result.get("label"),
            explanation=result.get("explanation"),
            metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
            metadata=metadata_dict,
            error=value.get("error"),
            official=bool(value.get("official", metadata_dict.pop("official", spec.official))),
            source=value.get("source") or metadata_dict.pop("source", None) or spec.source,
            version=value.get("version") or metadata_dict.pop("version", None) or spec.version,
        )
    return GradeReport(
        label=None if value is None else str(value),
        metadata=dict(spec.metadata),
        official=spec.official,
        source=spec.source,
        version=spec.version,
    )


def _call_kwargs(
    fn: GradeCallable,
    *,
    output: Mapping[str, Any],
    expected: Mapping[str, Any],
    input: Mapping[str, Any],
    metadata: Mapping[str, Any],
    example: Any = None,
) -> dict[str, Any]:
    available = {
        "output": output,
        "expected": expected,
        "input": input,
        "metadata": metadata,
        "example": example,
        "task": example,
    }
    signature = inspect.signature(fn)
    return {name: available[name] for name in signature.parameters if name in available}


async def run_grader(
    spec: GraderSpec,
    *,
    output: Mapping[str, Any],
    expected: Mapping[str, Any],
    input: Mapping[str, Any],
    metadata: Mapping[str, Any],
    example: Any = None,
    runtime: Any | None = None,
) -> GradeReport:
    """Invoke a benchmark grader with the context parameters it declares."""
    grader = spec.resolve(runtime)
    value = grader(
        **_call_kwargs(
            grader,
            output=output,
            expected=expected,
            input=input,
            metadata=metadata,
            example=example,
        )
    )
    if inspect.isawaitable(value):
        value = await value
    return normalize_grade(value, spec)


def platform_evaluator(spec: GraderSpec, runtime: Any | None = None) -> GradeCallable:
    """Adapt a benchmark grader to the existing a2e-client evaluator boundary."""

    async def _adapter(
        output: dict,
        expected: dict,
        input: dict,
        metadata: dict,
    ) -> dict[str, Any]:
        embedded = (output or {}).get("grade_report")
        if isinstance(embedded, Mapping):
            report = normalize_grade(embedded, spec)
        elif spec.mode == "inline" and spec.summarize is not None:
            report = spec.summarize_inline(output or {})
        else:
            report = await run_grader(
                spec,
                output=output or {},
                expected=expected or {},
                input=input or {},
                metadata=metadata or {},
                runtime=runtime,
            )
        metadata = {
            **dict(report.metadata),
            "metrics": dict(report.metrics),
            "passed": report.passed,
            "official": report.official,
            "source": report.source,
            "version": report.version,
        }
        if report.error:
            metadata["grader_error"] = report.error
        return {
            "score": report.score,
            "label": report.label
            or (
                "pass"
                if report.passed is True
                else "fail"
                if report.passed is False
                else None
            ),
            "explanation": report.explanation,
            "metadata": metadata,
        }

    _adapter.__name__ = spec.id
    _adapter.__qualname__ = spec.id
    _adapter.__doc__ = f"Platform adapter for benchmark grader {spec.id!r}."
    return _adapter
