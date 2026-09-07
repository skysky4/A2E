"""Official settings resolution and grader names."""

from __future__ import annotations

from ageneval.task.runners import DATASETS, EVALUATORS, LLM_GRADERS, resolve_run_settings
from ageneval.task.datasets.tau_bench.grader import tau_grader
from ageneval.task.datasets.tau_bench.reward import official_tau_reward


def test_official_tau_settings():
    s = resolve_run_settings(DATASETS["tau-bench"], env={})
    assert s["max_turns"] == 30
    assert s["max_tokens"] == 4096
    assert s["llm_timeout"] == 180.0
    assert s["run_deadline"] == 1100.0
    assert s["grader"] == "tau_grader"
    assert s["sources"]["max_turns"] == "official"


def test_official_dsqa_and_gdp_settings():
    dsqa = resolve_run_settings(DATASETS["deepsearchqa"], env={})
    assert dsqa["max_turns"] == 8
    assert dsqa["grader"] == "deepsearch_grader"
    gdp = resolve_run_settings(DATASETS["gdpval-aa"], env={})
    assert gdp["max_turns"] == 250
    assert gdp["max_tokens"] == 16384
    assert gdp["llm_timeout"] == 600.0
    assert gdp["run_deadline"] == 7000.0
    assert gdp["grader"] == "gdp_grader"


def test_explicit_empty_env_ignores_process_budget_leaks(monkeypatch):
    """apply_run_settings from a previous cell must not pin the next dataset."""
    monkeypatch.setenv("A2E_MAX_TURNS", "30")
    monkeypatch.setenv("A2E_AGNO_DEADLINE", "900")
    monkeypatch.setenv("A2E_RUN_DEADLINE", "1100")
    dsqa = resolve_run_settings(DATASETS["deepsearchqa"], env={})
    assert dsqa["max_turns"] == 8
    assert dsqa["run_deadline"] == 620.0
    gdp = resolve_run_settings(DATASETS["gdpval-aa"], env={})
    assert gdp["max_turns"] == 250
    assert gdp["run_deadline"] == 7000.0


def test_cli_overrides_env_and_official():
    s = resolve_run_settings(
        DATASETS["tau-bench"],
        max_turns=12,
        env={"A2E_MAX_TURNS": "99", "A2E_MAX_TOKENS": "111"},
    )
    assert s["max_turns"] == 12
    assert s["sources"]["max_turns"] == "cli"
    assert s["max_tokens"] == 111
    assert s["sources"]["max_tokens"] == "env:A2E_MAX_TOKENS"


def test_grader_registry_names():
    assert DATASETS["tau-bench"]["default_evaluators"] == ["tau_grader"]
    assert DATASETS["deepsearchqa"]["default_evaluators"] == ["deepsearch_grader"]
    assert DATASETS["gdpval-aa"]["default_evaluators"] == ["gdp_grader"]
    assert "tau_grader" in EVALUATORS
    assert "tau_reward" in EVALUATORS
    assert "deepsearch_grader" in EVALUATORS
    assert "deepsearch_match" in EVALUATORS
    assert "gdp_grader" in LLM_GRADERS


def test_tau_grader_matches_official_reward():
    output = {"tau_data_hash": "x", "final_answer": ""}
    expected = {"expected_actions": [], "expected_outputs": []}
    assert tau_grader(output, expected, {}) == official_tau_reward(
        output=output, expected=expected, input={}
    )
