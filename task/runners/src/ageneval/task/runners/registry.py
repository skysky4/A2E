"""Registries of available datasets, agents, and benchmark-owned graders.

This is the **single source of truth** that the CLI (`task/examples/run_experiment.py`)
and any future UI (A2E REST + React form) consult to populate dropdowns.

Scoring implementations live in each dataset package's ``grader.py``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict

from ageneval.task.core.grading import GraderSpec
from ageneval.task.core.instrumentation import Framework

# ─── DATASETS ────────────────────────────────────────────────────────────────
# Each entry: name → (loader fn, binding fn). Loader returns Dataset; binding
# returns AgentBinding (consumed by agents).


def _load_tau_bench(**kw: Any):
    from ageneval.task.datasets.tau_bench import load_tau_bench_tasks
    return load_tau_bench_tasks(**kw)


def _bind_tau_bench(**kw: Any):
    from ageneval.task.datasets.tau_bench import build_tau_bench_binding
    return build_tau_bench_binding(**kw)


def _load_tau2(**kw: Any):
    from ageneval.task.datasets.tau2 import load_tau2_tasks
    return load_tau2_tasks(**kw)


def _bind_tau2(**kw: Any):
    from ageneval.task.datasets.tau2 import build_tau2_binding
    return build_tau2_binding(domain=kw.get("domain") or "retail")


def _load_tau3(**kw: Any):
    from ageneval.task.datasets.tau3 import load_tau3_tasks
    allowed = {k: v for k, v in kw.items() if k in ("n", "split", "domain")}
    return load_tau3_tasks(**allowed)


def _bind_tau3(**kw: Any):
    from ageneval.task.datasets.tau3 import build_tau3_binding
    return build_tau3_binding(domain=kw.get("domain") or "retail")


def _load_mmlu(**kw: Any):
    from ageneval.task.datasets.mmlu import load_mmlu_tasks
    return load_mmlu_tasks(**kw)


def _bind_mmlu(**_kw: Any):
    from ageneval.task.datasets.mmlu import build_mmlu_binding
    return build_mmlu_binding()


def _load_gsm8k(**kw: Any):
    from ageneval.task.datasets.gsm8k import load_gsm8k_tasks
    return load_gsm8k_tasks(**kw)


def _bind_gsm8k(**_kw: Any):
    from ageneval.task.datasets.gsm8k import build_gsm8k_binding
    return build_gsm8k_binding()


def _load_humaneval(**kw: Any):
    from ageneval.task.datasets.humaneval import load_humaneval_tasks
    return load_humaneval_tasks(**kw)


def _bind_humaneval(**_kw: Any):
    from ageneval.task.datasets.humaneval import build_humaneval_binding
    return build_humaneval_binding()


def _load_persistbench(**kw: Any):
    from ageneval.task.datasets.persistbench import load_persistbench_tasks
    return load_persistbench_tasks(**kw)


def _bind_persistbench(**_kw: Any):
    from ageneval.task.datasets.persistbench import build_persistbench_binding
    return build_persistbench_binding()


def _load_traject_bench(**kw: Any):
    from ageneval.task.datasets.traject_bench import load_traject_bench_tasks
    return load_traject_bench_tasks(**{k: v for k, v in kw.items() if k in ("n", "split")})


def _bind_traject_bench(**_kw: Any):
    from ageneval.task.datasets.traject_bench import build_traject_bench_binding
    return build_traject_bench_binding()


def _load_gdpval(**kw: Any):
    from ageneval.task.datasets.gdpval import load_gdpval_tasks
    return load_gdpval_tasks(**{k: v for k, v in kw.items() if k in ("n", "split")})


def _bind_gdpval(**_kw: Any):
    from ageneval.task.datasets.gdpval import build_gdpval_binding
    return build_gdpval_binding()


def _load_deepsearchqa(**kw: Any):
    from ageneval.task.datasets.deepsearchqa import load_deepsearchqa_tasks
    return load_deepsearchqa_tasks(**{k: v for k, v in kw.items() if k in ("n", "split")})


def _bind_deepsearchqa(**_kw: Any):
    from ageneval.task.datasets.deepsearchqa import build_deepsearchqa_binding
    return build_deepsearchqa_binding()


# ─── sandbox datasets (SWE-bench) ─────────────────────────────────────────────
# These have kind="sandbox": the runner wraps the agent in a SandboxScoringRunner
# (see task_fn in run_experiment.py / a2e.py) which spins up the per-task
# container, lets the agent edit code, then grades with score_swe_bench.


def _load_swe(variant: str) -> Callable:
    def _f(**kw: Any):
        from ageneval.task.datasets.swe_bench import load_swe_bench_tasks
        kw = dict(kw)
        # Optionally pin a specific instance via A2E_SWE_INSTANCE so a demo / test
        # run targets an already-pulled image instead of fetching a random
        # multi-GB one. Explicit instance_ids (e.g. from the test harness) win.
        if not kw.get("instance_ids"):
            _pin = os.environ.get("A2E_SWE_INSTANCE")
            if _pin:
                kw["instance_ids"] = [_pin]
        return load_swe_bench_tasks(
            variant, **{k: v for k, v in kw.items() if k in ("n", "split", "instance_ids")}
        )
    return _f


def _bind_swe(**_kw: Any):
    from ageneval.task.datasets.swe_bench import build_swe_bench_binding
    return build_swe_bench_binding()


def _setup_swe(task: Any, sandbox: Any) -> None:
    from ageneval.task.datasets.swe_bench import setup_swe_bench
    return setup_swe_bench(task, sandbox)


# ─── SWE-bench Pro (ScaleAI) ──────────────────────────────────────────────────
# Same sandbox machinery, but a different HF dataset + official Scale harness
# (per-instance run_script.sh + parser.py, vendored). Repo lives at /app.
def _load_swe_pro(**kw: Any):
    from ageneval.task.datasets.swe_bench_pro import load_swe_bench_pro_tasks
    kw = dict(kw)
    # A2E_SWE_PRO_INSTANCE pins a specific instance (e.g. one whose multi-GB
    # image is already pulled) for demos/tests; explicit instance_ids win.
    if not kw.get("instance_ids"):
        _pin = os.environ.get("A2E_SWE_PRO_INSTANCE")
        if _pin:
            kw["instance_ids"] = [_pin]
    return load_swe_bench_pro_tasks(
        "swe-bench-pro", **{k: v for k, v in kw.items() if k in ("n", "split", "instance_ids")}
    )


def _bind_swe_pro(**_kw: Any):
    from ageneval.task.datasets.swe_bench_pro import build_swe_bench_pro_binding
    return build_swe_bench_pro_binding()


def _setup_swe_pro(task: Any, sandbox: Any) -> None:
    from ageneval.task.datasets.swe_bench_pro import setup_swe_bench_pro
    return setup_swe_bench_pro(task, sandbox)


# ─── Terminal-Bench 2.0 (laude-institute) ─────────────────────────────────────
# Same sandbox machinery as SWE-bench: each task's published docker image is the
# environment; the agent works via bash/editor tools, then the held-out verifier
# (tests/test.sh) is copied in and run to produce a 1/0 reward → ``resolved``.
def _load_tb2(**kw: Any):
    from ageneval.task.datasets.terminal_bench_2 import load_terminal_bench_2_tasks
    kw = dict(kw)
    # A2E_TB2_TASK pins one task (e.g. one whose image is already pulled) for
    # demos/tests; explicit task_ids win.
    if not kw.get("task_ids"):
        _pin = os.environ.get("A2E_TB2_TASK")
        if _pin:
            kw["task_ids"] = [_pin]
    return load_terminal_bench_2_tasks(**{k: v for k, v in kw.items() if k in ("n", "task_ids")})


def _bind_tb2(**_kw: Any):
    from ageneval.task.datasets.terminal_bench_2 import build_terminal_bench_2_binding
    return build_terminal_bench_2_binding()


def _setup_tb2(task: Any, sandbox: Any) -> None:
    from ageneval.task.datasets.terminal_bench_2 import setup_terminal_bench_2
    return setup_terminal_bench_2(task, sandbox)


# ─── Terminal-Bench 2.1 (harbor-framework) ───────────────────────────────────
def _load_tb21(**kw: Any):
    from ageneval.task.datasets.terminal_bench_2_1 import load_terminal_bench_2_1_tasks
    kw = dict(kw)
    if not kw.get("task_ids"):
        _pin = os.environ.get("AEP_TB21_TASK") or os.environ.get("A2E_TB21_TASK")
        if _pin:
            kw["task_ids"] = [_pin]
    return load_terminal_bench_2_1_tasks(
        **{
            k: v
            for k, v in kw.items()
            if k in ("n", "task_ids", "exclude_categories")
        }
    )


def _bind_tb21(**_kw: Any):
    from ageneval.task.datasets.terminal_bench_2_1 import build_terminal_bench_2_1_binding
    return build_terminal_bench_2_1_binding()


def _setup_tb21(task: Any, sandbox: Any) -> None:
    from ageneval.task.datasets.terminal_bench_2_1 import setup_terminal_bench_2_1
    return setup_terminal_bench_2_1(task, sandbox)


def _qa_load(bench: str) -> Callable:
    """Closure factory: load fn bound to one qa_suite benchmark key."""
    def _f(**kw: Any):
        from ageneval.task.datasets.qa_suite import load_qa_tasks
        return load_qa_tasks(bench, **{k: v for k, v in kw.items() if k in ("n", "split")})
    return _f


def _qa_bind(bench: str) -> Callable:
    """Closure factory: binding fn bound to one qa_suite benchmark key."""
    def _f(**_kw: Any):
        from ageneval.task.datasets.qa_suite import build_qa_binding
        return build_qa_binding(bench)
    return _f


_TAU_SETTINGS = {
    "max_turns": 30,
    "max_tokens": 4096,
    "llm_timeout": 180.0,
    "run_deadline": 1100.0,
    "wall": 1200,
}
_QA_SETTINGS = {
    "max_turns": 8,
    "max_tokens": 4096,
    "llm_timeout": 180.0,
    "run_deadline": 1800.0,
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "tau-bench": {"load": _load_tau_bench, "bind": _bind_tau_bench, "kind": "tool",
                  "official_settings": _TAU_SETTINGS,
                  "agent_overrides": {"max_turns": 30, "max_steps": 30}},
    "tau2":      {"load": _load_tau2, "bind": _bind_tau2, "kind": "tool",
                  "official_settings": _TAU_SETTINGS,
                  "agent_overrides": {"max_turns": 30, "max_steps": 30}},
    "tau3":      {"load": _load_tau3, "bind": _bind_tau3, "kind": "tool",
                  "official_settings": _TAU_SETTINGS,
                  "agent_overrides": {"max_turns": 30, "max_steps": 30}},
    "tau3bench": {"load": _load_tau3, "bind": _bind_tau3, "kind": "tool",
                  "official_settings": _TAU_SETTINGS,
                  "agent_overrides": {"max_turns": 30, "max_steps": 30}},
    "tau3-bench": {"load": _load_tau3, "bind": _bind_tau3, "kind": "tool",
                   "official_settings": _TAU_SETTINGS,
                   "agent_overrides": {"max_turns": 30, "max_steps": 30}},
    "mmlu":      {"load": _load_mmlu, "bind": _bind_mmlu, "kind": "qa",
                  "official_settings": _QA_SETTINGS,
                  "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "gsm8k":     {"load": _load_gsm8k, "bind": _bind_gsm8k, "kind": "qa",
                  "official_settings": _QA_SETTINGS,
                  "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "humaneval": {"load": _load_humaneval, "bind": _bind_humaneval, "kind": "qa",
                  "official_settings": _QA_SETTINGS,
                  "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "persistbench": {"load": _load_persistbench, "bind": _bind_persistbench, "kind": "qa",
                     "official_settings": _QA_SETTINGS,
                     "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "traject-bench": {"load": _load_traject_bench, "bind": _bind_traject_bench, "kind": "tool",
                      "official_settings": _QA_SETTINGS,
                      "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "gdpval-aa": {"load": _load_gdpval, "bind": _bind_gdpval, "kind": "qa",
                  "official_settings": _QA_SETTINGS,
                  "agent_overrides": {"max_turns": 8, "max_steps": 8}},
    "deepsearchqa": {"load": _load_deepsearchqa, "bind": _bind_deepsearchqa, "kind": "tool",
                     "official_settings": {**_QA_SETTINGS, "max_turns": 20},
                     "agent_overrides": {"max_turns": 20, "max_steps": 20}},
}

# qa_suite — 10 config-driven pure-QA benchmarks (no sandbox/tools).
for _b in ("gpqa", "mmlu-pro", "arc-challenge", "truthfulqa", "bbh",
           "agieval", "commonsenseqa", "hellaswag", "openbookqa", "math"):
    DATASETS[_b] = {"load": _qa_load(_b), "bind": _qa_bind(_b), "kind": "qa",
                    "official_settings": _QA_SETTINGS,
                    "agent_overrides": {"max_turns": 8, "max_steps": 8}}

# Sandbox datasets — run inside a docker container; graded by score_swe_bench
# while the container is alive (see SandboxScoringRunner). ``agent_overrides``
# raise the agent's turn budget (a SWE fix needs many explore/edit steps).
for _v in ("swe-bench-lite", "swe-bench-verified"):
    DATASETS[_v] = {
        "load": _load_swe(_v), "bind": _bind_swe, "kind": "sandbox",
        "setup": _setup_swe,
        "official_settings": {"max_turns": 40, "run_deadline": 3600.0},
        "agent_overrides": {"max_turns": 40, "max_steps": 40},
    }

# SWE-bench Pro (ScaleAI) — official Scale harness grading (see swe_bench_pro pkg).
DATASETS["swe-bench-pro"] = {
    "load": _load_swe_pro, "bind": _bind_swe_pro, "kind": "sandbox",
    "setup": _setup_swe_pro,
    "official_settings": {"max_turns": 40, "run_deadline": 3600.0},
    "agent_overrides": {"max_turns": 40, "max_steps": 40},
}

# Terminal-Bench 2.0 — sandbox dataset graded by the official held-out tests.
DATASETS["terminal-bench-2"] = {
    "load": _load_tb2, "bind": _bind_tb2, "kind": "sandbox",
    "setup": _setup_tb2,
    "official_settings": {"max_turns": 40, "run_deadline": 3600.0},
    "agent_overrides": {"max_turns": 40, "max_steps": 40},
}

# Terminal-Bench 2.1 remains separate so published results preserve the version.
DATASETS["terminal-bench-2.1"] = {
    "load": _load_tb21, "bind": _bind_tb21, "kind": "sandbox",
    "setup": _setup_tb21,
    "official_settings": {"max_turns": 10_000, "run_deadline": 3600.0},
    # Terminal-Bench is governed by each task.toml's [agent].timeout_sec.
    # Keep the framework-required turn/step cap effectively non-binding so it
    # cannot terminate a task before that wall-clock budget expires.
    "agent_overrides": {"max_turns": 10_000, "max_steps": 10_000},
}

_GRADER_MODULES: dict[str, str] = {
    "tau-bench": "ageneval.task.datasets.tau_bench.grader",
    "tau2": "ageneval.task.datasets.tau2.grader",
    "tau3": "ageneval.task.datasets.tau3.grader",
    "tau3bench": "ageneval.task.datasets.tau3.grader",
    "tau3-bench": "ageneval.task.datasets.tau3.grader",
    "mmlu": "ageneval.task.datasets.mmlu.grader",
    "gsm8k": "ageneval.task.datasets.gsm8k.grader",
    "humaneval": "ageneval.task.datasets.humaneval.grader",
    "persistbench": "ageneval.task.datasets.persistbench.grader",
    "traject-bench": "ageneval.task.datasets.traject_bench.grader",
    "gdpval-aa": "ageneval.task.datasets.gdpval.grader",
    "deepsearchqa": "ageneval.task.datasets.deepsearchqa.grader",
    "gpqa": "ageneval.task.datasets.qa_suite.grader",
    "mmlu-pro": "ageneval.task.datasets.qa_suite.grader",
    "arc-challenge": "ageneval.task.datasets.qa_suite.grader",
    "truthfulqa": "ageneval.task.datasets.qa_suite.grader",
    "bbh": "ageneval.task.datasets.qa_suite.grader",
    "agieval": "ageneval.task.datasets.qa_suite.grader",
    "commonsenseqa": "ageneval.task.datasets.qa_suite.grader",
    "hellaswag": "ageneval.task.datasets.qa_suite.grader",
    "openbookqa": "ageneval.task.datasets.qa_suite.grader",
    "math": "ageneval.task.datasets.qa_suite.grader",
    "swe-bench-lite": "ageneval.task.datasets.swe_bench.grader",
    "swe-bench-verified": "ageneval.task.datasets.swe_bench.grader",
    "swe-bench-pro": "ageneval.task.datasets.swe_bench_pro.grader",
    "terminal-bench-2": "ageneval.task.datasets.terminal_bench_2.grader",
    "terminal-bench-2.1": "ageneval.task.datasets.terminal_bench_2_1.grader",
}
for _dataset_key, _grader_module in _GRADER_MODULES.items():
    DATASETS[_dataset_key]["grader_module"] = _grader_module


def grader_for_dataset(dataset: str) -> GraderSpec:
    """Load the primary grader owned by one benchmark package."""
    try:
        entry = DATASETS[dataset]
    except KeyError as exc:
        raise KeyError(f"unknown dataset: {dataset}") from exc
    module_name = entry.get("grader_module")
    if not module_name:
        raise LookupError(f"dataset {dataset!r} does not declare a grader")
    module = importlib.import_module(str(module_name))
    resolver = getattr(module, "grader_for_benchmark", None)
    spec = resolver(dataset) if callable(resolver) else getattr(module, "GRADER", None)
    if not isinstance(spec, GraderSpec):
        raise TypeError(f"{module_name} must export a GraderSpec named GRADER")
    return spec


_SESSION_WRAPPERS: dict[str, tuple[str, str]] = {
    "tau-bench": (
        "ageneval.task.datasets.tau_bench.session",
        "wrap_tau_official_session",
    ),
    "tau2": (
        "ageneval.task.datasets.tau_bench.session",
        "wrap_tau_official_session",
    ),
    "tau3": (
        "ageneval.task.datasets.tau_bench.session",
        "wrap_tau_official_session",
    ),
    "tau3bench": (
        "ageneval.task.datasets.tau_bench.session",
        "wrap_tau_official_session",
    ),
    "tau3-bench": (
        "ageneval.task.datasets.tau_bench.session",
        "wrap_tau_official_session",
    ),
    "deepsearchqa": (
        "ageneval.task.datasets.deepsearchqa.session",
        "wrap_dsqa_official_session",
    ),
}


def wrap_agent_for_dataset(dataset: str, agent: Any) -> Any:
    """Apply a benchmark-owned stateful session wrapper when required."""
    target = _SESSION_WRAPPERS.get(dataset)
    if target is None:
        return agent
    module_name, attribute = target
    wrapper = getattr(importlib.import_module(module_name), attribute)
    return wrapper(agent)


# ─── AGENTS ──────────────────────────────────────────────────────────────────


def _build_langgraph(*, binding: Any, **kw: Any):
    from ageneval.task.agents.langgraph import LangGraphAgent
    # Filter kwargs to what LangGraphAgent accepts. Sandbox datasets inject
    # agent_overrides with BOTH `max_turns` and `max_steps` (different agents
    # name their budget differently); LangGraphAgent uses `max_turns`, so drop
    # `max_steps` here — every sibling builder applies the same whitelist.
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return LangGraphAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_claude_sdk(*, binding: Any, **kw: Any):
    """Build the generic ClaudeSDKAgent for any binding."""
    from ageneval.task.agents.claude_sdk import ClaudeSDKAgent
    # ClaudeSDKAgent talks Anthropic Messages API but can reuse the same
    # gateway credentials as OpenAI-compatible harnesses (api_base / api_key).
    accepted = {"model", "max_turns", "api_base", "api_key"}
    return ClaudeSDKAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_smolagents(*, binding: Any, **kw: Any):
    """Build the generic SmolAgentsAgent for any binding."""
    from ageneval.task.agents.smolagents import SmolAgentsAgent
    # Dataset overrides use max_turns; smolagents names the budget max_steps.
    if "max_steps" not in kw and kw.get("max_turns") is not None:
        kw = {**kw, "max_steps": kw["max_turns"]}
    accepted = {"model", "max_steps", "api_base", "api_key"}
    return SmolAgentsAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_openai_agents(*, binding: Any, **kw: Any):
    """Build the generic OpenAIAgentsAgent for any binding."""
    from ageneval.task.agents.openai_agents import OpenAIAgentsAgent
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return OpenAIAgentsAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_google_adk(*, binding: Any, **kw: Any):
    """Build the generic GoogleADKAgent for any binding."""
    from ageneval.task.agents.google_adk import GoogleADKAgent
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return GoogleADKAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_agno(*, binding: Any, **kw: Any):
    """Build the generic AgnoAgent for any binding."""
    from ageneval.task.agents.agno import AgnoAgent
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return AgnoAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_llama_index(*, binding: Any, **kw: Any):
    """Build the generic LlamaIndexAgent for any binding."""
    from ageneval.task.agents.llama_index import LlamaIndexAgent
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return LlamaIndexAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_crewai(*, binding: Any, **kw: Any):
    """Build the generic CrewAIAgent for any binding."""
    from ageneval.task.agents.crewai import CrewAIAgent
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return CrewAIAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


def _build_autogen(*, binding: Any, **kw: Any):
    """Build the generic AutogenAgentChatAgent for any binding.

    autogen-agentchat lives in an ISOLATED uv project (``autogen-core`` pins
    protobuf<5.30, which conflicts with A2E). It is therefore not installed
    into the main workspace ``.venv``; importing it from there will fail with a
    clear instruction to use the isolated environment.
    """
    try:
        from ageneval.task.agents.autogen_agentchat import AutogenAgentChatAgent
    except ImportError as exc:
        raise RuntimeError(
            "autogen-agentchat is an isolated agent (protobuf conflict with "
            "A2E). Install and run it from its own environment:\n"
            "  cd task/agents/autogen_agentchat && "
            "uv sync --index-strategy unsafe-best-match"
        ) from exc
    accepted = {"model", "api_base", "api_key", "max_turns"}
    return AutogenAgentChatAgent(binding=binding, **{k: v for k, v in kw.items() if k in accepted})


_OPENAI_REQUIREMENTS = {
    "protocols": ["openai_chat_completions"],
    "capabilities": {"tools": True},
}
_ANTHROPIC_REQUIREMENTS = {
    "protocols": ["anthropic_messages"],
    "capabilities": {"tools": True},
}


AGENTS: Dict[str, Dict[str, Any]] = {
    "langgraph":     {"build": _build_langgraph,     "framework": "langchain",     "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "claude-sdk":    {"build": _build_claude_sdk,    "framework": "anthropic",     "supports_any_binding": True, "requirements": _ANTHROPIC_REQUIREMENTS},
    "smolagents":    {"build": _build_smolagents,    "framework": "smolagents",    "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "openai-agents": {"build": _build_openai_agents, "framework": "openai_agents", "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "google-adk":    {"build": _build_google_adk,    "framework": "google_adk",    "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "agno":          {"build": _build_agno,          "framework": "agno",          "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "llama-index":   {"build": _build_llama_index,   "framework": "llama_index",   "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "crewai":        {"build": _build_crewai,        "framework": "crewai",        "supports_any_binding": True, "requirements": _OPENAI_REQUIREMENTS},
    "autogen-agentchat": {"build": _build_autogen,   "framework": "autogen_agentchat", "supports_any_binding": True, "isolated": True, "requirements": _OPENAI_REQUIREMENTS},
}


# Display grouping for the Run page <optgroup> taxonomy (UI-only metadata).
_AGENT_GROUP: Dict[str, str] = {
    "smolagents": "Agent-first frameworks",
    "agno": "Agent-first frameworks",
    "llama-index": "Agent-first frameworks",
    "langgraph": "Orchestration frameworks",
    "autogen-agentchat": "Orchestration frameworks",
    "crewai": "Orchestration frameworks",
    "google-adk": "Orchestration frameworks",
    "claude-sdk": "Vendor agent SDKs",
    "openai-agents": "Vendor agent SDKs",
}


def framework_for_agent(agent: str) -> Framework:
    """Single source of truth: which OpenInference instrumentor an agent needs."""
    return AGENTS.get(agent, {}).get("framework", "none")


def build_experiment_metadata(*, agent_name: str, agent: Any, sdk: str) -> Dict[str, Any]:
    """Metadata stored on A2E experiments for downstream DB analysis."""
    return {
        "agent_framework": agent_name,
        "model": getattr(agent, "model", None) or getattr(agent, "_model_name", None),
        "sdk": sdk,
    }


def list_registries() -> Dict[str, Any]:
    """Return datasets, agents, and each benchmark-owned primary grader."""
    grader_meta = {}
    for name in sorted(DATASETS):
        spec = grader_for_dataset(name)
        grader_meta[name] = {
            "id": spec.id,
            "mode": spec.mode,
            "official": spec.official,
            "source": spec.source,
            "version": spec.version,
            "required": spec.required,
        }
    return {
        "datasets": sorted(DATASETS),
        "agents": sorted(AGENTS),
        "graders": grader_meta,
        "agent_meta": {
            name: {
                "framework": meta.get("framework", "none"),
                "group": _AGENT_GROUP.get(name, "Agent-first frameworks"),
                "isolated": bool(meta.get("isolated", False)),
                "requirements": meta.get("requirements", {}),
            }
            for name, meta in AGENTS.items()
        },
        "dataset_meta": {
            name: {
                "kind": meta.get("kind", "qa"),
                "grader": grader_meta[name],
            }
            for name, meta in DATASETS.items()
        },
    }
