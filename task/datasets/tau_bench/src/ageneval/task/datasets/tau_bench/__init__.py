"""τ-bench dataset adapter for A2E."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GRADER",
    "GRADER_METADATA",
    "TauBenchDataset",
    "build_tau_bench_binding",
    "get_tool_schemas",
    "load_tau_bench_tasks",
    "official_tau_reward",
    "tau_grader",
    "wrap_tau_official_session",
]


def __getattr__(name: str) -> Any:
    if name == "build_tau_bench_binding":
        from ageneval.task.datasets.tau_bench.binding import build_tau_bench_binding

        return build_tau_bench_binding
    if name in {"TauBenchDataset", "load_tau_bench_tasks"}:
        from ageneval.task.datasets.tau_bench.loader import TauBenchDataset, load_tau_bench_tasks

        return TauBenchDataset if name == "TauBenchDataset" else load_tau_bench_tasks
    if name == "get_tool_schemas":
        from ageneval.task.datasets.tau_bench.tools import get_tool_schemas

        return get_tool_schemas
    if name in {"GRADER", "GRADER_METADATA", "tau_grader"}:
        from ageneval.task.datasets.tau_bench.grader import (
            GRADER,
            GRADER_METADATA,
            tau_grader,
        )

        return {
            "GRADER": GRADER,
            "GRADER_METADATA": GRADER_METADATA,
            "tau_grader": tau_grader,
        }[name]
    if name == "official_tau_reward":
        from ageneval.task.datasets.tau_bench.reward import official_tau_reward

        return official_tau_reward
    if name == "wrap_tau_official_session":
        from ageneval.task.datasets.tau_bench.session import wrap_tau_official_session

        return wrap_tau_official_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
