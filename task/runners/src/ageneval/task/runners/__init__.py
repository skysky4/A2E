"""One-call runners and benchmark/agent registries.

Public surface:
    - run_tau_claude / run_tau_langgraph   — convenience helpers for τ-bench
    - DATASETS / AGENTS                     — name → factory registries
    - grader_for_dataset                    — benchmark-owned primary grader
"""

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
from ageneval.task.runners.tau_claude_runner import run_tau_claude
from ageneval.task.runners.tau_langgraph_runner import run_tau_langgraph
from ageneval.task.runners.settings import (
    apply_run_settings,
    benchmark_run_settings,
    format_run_settings,
    resolve_run_settings,
)

__all__ = [
    "AGENTS",
    "DATASETS",
    "DEFAULT_SAMPLE_SIZE",
    "RunIdentity",
    "SampleSelection",
    "apply_run_settings",
    "benchmark_run_settings",
    "build_experiment_metadata",
    "build_run_identity",
    "framework_for_agent",
    "grader_for_dataset",
    "list_registries",
    "format_run_settings",
    "new_run_id",
    "run_tau_claude",
    "run_tau_langgraph",
    "resolve_run_settings",
    "sample_dataset",
    "wrap_agent_for_dataset",
]
