from __future__ import annotations

from ageneval.task.runners.settings import (
    benchmark_run_settings,
    resolve_run_settings,
)


def test_benchmark_settings_fall_back_to_agent_budget() -> None:
    settings = benchmark_run_settings(
        {"agent_overrides": {"max_turns": 30, "max_steps": 30}}
    )
    assert settings["max_turns"] == 30


def test_cli_budget_wins_over_env_and_benchmark() -> None:
    settings = resolve_run_settings(
        {"official_settings": {"max_turns": 8}},
        max_turns=12,
        env={"A2E_MAX_TURNS": "10"},
    )
    assert settings["max_turns"] == 12
    assert settings["sources"]["max_turns"] == "cli"


def test_environment_wins_over_benchmark() -> None:
    settings = resolve_run_settings(
        {"official_settings": {"max_tokens": 4096}},
        env={"A2E_MAX_TOKENS": "2048"},
    )
    assert settings["max_tokens"] == 2048
    assert settings["sources"]["max_tokens"] == "env:A2E_MAX_TOKENS"
