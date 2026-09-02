"""Registries of available datasets, agents, and benchmark-owned graders.

This is the **single source of truth** that the CLI (`task/examples/run_experiment.py`)
and any future UI (A2E REST + React form) consult to populate dropdowns.

Scoring implementations live in each dataset package's ``grader.py``.
"""

from __future__ import annotations

from typing import Any

from ageneval.task.core.grading import GraderSpec
from ageneval.task.core.instrumentation import Framework
from ageneval.task.runners.benchmark_profile import (
    build_dataset_registry,
    component_callable,
    resolve_component,
)

# ─── DATASETS ────────────────────────────────────────────────────────────────
# Every dataset package owns its benchmark YAML. This module only adapts those
# declarative profiles to the existing runner interface.
DATASETS = build_dataset_registry()


def grader_for_dataset(dataset: str) -> GraderSpec:
    """Load the primary grader owned by one benchmark package."""
    try:
        entry = DATASETS[dataset]
    except KeyError as exc:
        raise KeyError(f"unknown dataset: {dataset}") from exc
    profile = entry.get("profile")
    if profile is None:
        raise LookupError(f"dataset {dataset!r} does not declare a grader")
    spec = resolve_component(profile.grader)
    if not isinstance(spec, GraderSpec):
        raise TypeError(
            f"benchmark {dataset!r} grader {profile.grader.entrypoint!r} "
            "did not resolve to GraderSpec"
        )
    return spec


def wrap_agent_for_dataset(dataset: str, agent: Any) -> Any:
    """Apply a benchmark-owned stateful session wrapper when required."""
    try:
        entry = DATASETS[dataset]
    except KeyError as exc:
        raise KeyError(f"unknown dataset: {dataset}") from exc
    # Programmatically registered datasets (notably tests and third-party
    # extensions) may still use the minimal legacy entry shape.
    profile = entry.get("profile")
    if profile is None:
        return agent
    if profile.session is None:
        return agent
    return component_callable(profile.session)(agent)


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


AGENTS: dict[str, dict[str, Any]] = {
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
_AGENT_GROUP: dict[str, str] = {
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


def build_experiment_metadata(*, agent_name: str, agent: Any, sdk: str) -> dict[str, Any]:
    """Metadata stored on A2E experiments for downstream DB analysis."""
    return {
        "agent_framework": agent_name,
        "model": getattr(agent, "model", None) or getattr(agent, "_model_name", None),
        "sdk": sdk,
    }


def list_registries() -> dict[str, Any]:
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
