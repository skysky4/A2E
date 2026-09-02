"""Per-benchmark run settings and CLI/environment resolution.

Numeric budget precedence is:

    CLI flag > process environment > benchmark official settings
"""

from __future__ import annotations

import os
from typing import Any, Mapping


def benchmark_run_settings(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the grader and execution defaults stored on a dataset entry."""
    dataset = dict(entry or {})
    configured = dict(dataset.get("official_settings") or {})
    overrides = dict(dataset.get("agent_overrides") or {})
    grader_loader = dataset.get("grader")
    grader_id = ""
    if callable(grader_loader):
        try:
            grader_id = str(grader_loader().id)
        except Exception:
            grader_id = ""
    return {
        "max_turns": int(configured.get("max_turns", overrides.get("max_turns", 8))),
        "max_tokens": int(configured.get("max_tokens", 4096)),
        "llm_timeout": float(configured.get("llm_timeout", 180)),
        "run_deadline": float(configured.get("run_deadline", 1800)),
        "wall": configured.get("wall"),
        "grader": str(configured.get("grader") or grader_id),
    }


def resolve_run_settings(
    entry: Mapping[str, Any] | None,
    *,
    max_turns: int | None = None,
    max_tokens: int | None = None,
    llm_timeout: float | None = None,
    run_deadline: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one run and record the source chosen for each budget."""
    environ = env if env is not None else os.environ
    defaults = benchmark_run_settings(entry)
    sources: dict[str, str] = {}

    def pick(cli: Any, env_keys: tuple[str, ...], key: str, cast: Any) -> Any:
        if cli is not None:
            sources[key] = "cli"
            return cast(cli)
        for name in env_keys:
            raw = environ.get(name)
            if raw is not None and str(raw).strip():
                sources[key] = f"env:{name}"
                return cast(raw)
        sources[key] = "benchmark"
        return defaults[key]

    return {
        "max_turns": pick(
            max_turns,
            ("A2E_MAX_TURNS", "A2E_MAX_STEPS"),
            "max_turns",
            int,
        ),
        "max_tokens": pick(max_tokens, ("A2E_MAX_TOKENS",), "max_tokens", int),
        "llm_timeout": pick(
            llm_timeout,
            ("A2E_LLM_TIMEOUT",),
            "llm_timeout",
            float,
        ),
        "run_deadline": pick(
            run_deadline,
            ("A2E_RUN_DEADLINE", "A2E_AGNO_DEADLINE"),
            "run_deadline",
            float,
        ),
        "wall": defaults.get("wall"),
        "grader": defaults["grader"],
        "sources": sources,
    }


def apply_run_settings(settings: Mapping[str, Any]) -> None:
    """Apply one resolved budget consistently across agent harnesses."""
    os.environ["A2E_MAX_TURNS"] = str(int(settings["max_turns"]))
    os.environ["A2E_MAX_STEPS"] = str(int(settings["max_turns"]))
    os.environ["A2E_MAX_TOKENS"] = str(int(settings["max_tokens"]))
    os.environ["A2E_LLM_TIMEOUT"] = str(settings["llm_timeout"])
    os.environ["A2E_RUN_DEADLINE"] = str(settings["run_deadline"])
    os.environ["A2E_AGNO_DEADLINE"] = str(settings["run_deadline"])


def format_run_settings(settings: Mapping[str, Any], *, dataset: str) -> str:
    """Format the resolved benchmark settings for CLI output."""
    sources = dict(settings.get("sources") or {})
    rows = [
        f"run settings ({dataset})",
        f"  grader:       {settings.get('grader') or '(none)'}",
        f"  max_turns:    {settings['max_turns']}  [{sources.get('max_turns', 'benchmark')}]",
        f"  max_tokens:   {settings['max_tokens']}  [{sources.get('max_tokens', 'benchmark')}]",
        f"  llm_timeout:  {settings['llm_timeout']}  [{sources.get('llm_timeout', 'benchmark')}]",
        f"  run_deadline: {settings['run_deadline']}  [{sources.get('run_deadline', 'benchmark')}]",
    ]
    if settings.get("wall") is not None:
        rows.append(f"  wall:         {settings['wall']}")
    return "\n".join(rows)
