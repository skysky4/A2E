"""Official per-benchmark run settings and CLI/env resolution.

Precedence for every numeric budget:

    CLI flag  >  process env  >  dataset ``official_settings`` / ``agent_overrides``
"""

from __future__ import annotations

import os
from typing import Any, Mapping


def official_run_settings(ds_entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the official cell budgets stored on one ``DATASETS`` entry."""
    entry = dict(ds_entry or {})
    extra = dict(entry.get("official_settings") or {})
    overrides = dict(entry.get("agent_overrides") or {})
    graders = list(entry.get("default_evaluators") or [])
    return {
        "max_turns": int(extra.get("max_turns", overrides.get("max_turns", 8))),
        "max_tokens": int(extra.get("max_tokens", 4096)),
        "llm_timeout": float(extra.get("llm_timeout", 180)),
        "run_deadline": float(extra.get("run_deadline", 1800)),
        "wall": extra.get("wall"),
        "grader": str(extra.get("grader") or (graders[0] if graders else "")),
        "graders": graders,
    }


def resolve_run_settings(
    ds_entry: Mapping[str, Any] | None,
    *,
    max_turns: int | None = None,
    max_tokens: int | None = None,
    llm_timeout: float | None = None,
    run_deadline: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one run's budgets. ``sources`` records where each value came from."""
    environ = env if env is not None else os.environ
    official = official_run_settings(ds_entry)
    sources: dict[str, str] = {}

    def _pick(
        cli: Any,
        env_keys: tuple[str, ...],
        key: str,
        cast,
    ):
        if cli is not None:
            sources[key] = "cli"
            return cast(cli)
        for name in env_keys:
            raw = environ.get(name)
            if raw is not None and str(raw).strip() != "":
                sources[key] = f"env:{name}"
                return cast(raw)
        sources[key] = "official"
        return official[key]

    return {
        "max_turns": _pick(max_turns, ("A2E_MAX_TURNS", "A2E_MAX_STEPS"), "max_turns", int),
        "max_tokens": _pick(max_tokens, ("A2E_MAX_TOKENS",), "max_tokens", int),
        "llm_timeout": _pick(llm_timeout, ("A2E_LLM_TIMEOUT",), "llm_timeout", float),
        "run_deadline": _pick(
            run_deadline,
            ("A2E_RUN_DEADLINE", "A2E_AGNO_DEADLINE"),
            "run_deadline",
            float,
        ),
        "wall": official.get("wall"),
        "grader": official["grader"],
        "graders": list(official["graders"]),
        "sources": sources,
    }


def apply_run_settings(settings: Mapping[str, Any]) -> None:
    """Write resolved budgets into the process env so ``budget.py`` and harnesses agree."""
    os.environ["A2E_MAX_TURNS"] = str(int(settings["max_turns"]))
    os.environ["A2E_MAX_STEPS"] = str(int(settings["max_turns"]))
    os.environ["A2E_MAX_TOKENS"] = str(int(settings["max_tokens"]))
    os.environ["A2E_LLM_TIMEOUT"] = str(settings["llm_timeout"])
    os.environ["A2E_RUN_DEADLINE"] = str(settings["run_deadline"])
    os.environ["A2E_AGNO_DEADLINE"] = str(settings["run_deadline"])


def format_run_settings(settings: Mapping[str, Any], *, dataset: str) -> str:
    """Human-readable block printed at the start of a CLI run."""
    sources = dict(settings.get("sources") or {})
    rows = [
        f"⚙ run settings ({dataset})",
        f"  grader:       {settings.get('grader') or '(none)'}",
        f"  max_turns:    {settings['max_turns']}  [{sources.get('max_turns', 'official')}]",
        f"  max_tokens:   {settings['max_tokens']}  [{sources.get('max_tokens', 'official')}]",
        f"  llm_timeout:  {settings['llm_timeout']}  [{sources.get('llm_timeout', 'official')}]",
        f"  run_deadline: {settings['run_deadline']}  [{sources.get('run_deadline', 'official')}]",
    ]
    if settings.get("wall") is not None:
        rows.append(f"  wall:         {settings['wall']}")
    return "\n".join(rows)
