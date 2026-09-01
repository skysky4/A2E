"""One-call runners and benchmark/agent registries.

Public surface:
    - run_tau_claude / run_tau_langgraph   — convenience helpers for τ-bench
    - DATASETS / AGENTS                     — name → factory registries
    - grader_for_dataset                    — benchmark-owned primary grader
"""

from ageneval.task.runners.benchmark_profile import (
    BenchmarkProfile,
    discover_benchmark_profiles,
    load_benchmark_profile,
)
from ageneval.task.runners.registry import (
    AGENTS,
    DATASETS,
    build_experiment_metadata,
    framework_for_agent,
    grader_for_dataset,
    list_registries,
    wrap_agent_for_dataset,
)
from ageneval.task.runners.run_context import (
    DEFAULT_SAMPLE_SIZE,
    RunIdentity,
    SampleSelection,
    build_run_identity,
    new_run_id,
    sample_dataset,
)
from ageneval.task.runners.settings import (
    apply_run_settings,
    benchmark_run_settings,
    format_run_settings,
    resolve_run_settings,
)
from ageneval.task.runners.tau_claude_runner import run_tau_claude
from ageneval.task.runners.tau_langgraph_runner import run_tau_langgraph

__all__ = [
    "AGENTS",
    "DATASETS",
    "DEFAULT_SAMPLE_SIZE",
    "BenchmarkProfile",
    "RunIdentity",
    "SampleSelection",
    "apply_run_settings",
    "benchmark_run_settings",
    "build_experiment_metadata",
    "build_run_identity",
    "discover_benchmark_profiles",
    "format_run_settings",
    "framework_for_agent",
    "grader_for_dataset",
    "list_registries",
    "load_benchmark_profile",
    "new_run_id",
    "resolve_run_settings",
    "run_tau_claude",
    "run_tau_langgraph",
    "sample_dataset",
    "wrap_agent_for_dataset",
]
