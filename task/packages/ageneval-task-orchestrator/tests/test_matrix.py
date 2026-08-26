from __future__ import annotations

from pathlib import Path

import pytest
from ageneval.model.gateway import ModelProfile, load_model_profile
from ageneval.task.orchestrator.matrix import UnsupportedCombination, expand_campaign
from ageneval.task.orchestrator.schema import CampaignConfig


def _profile(identifier: str = "model-a", *, limit: int = 2) -> ModelProfile:
    return ModelProfile.model_validate(
        {
            "id": identifier,
            "provider": "test",
            "model": identifier,
            "upstream_protocol": "openai_chat_completions",
            "connection": {"api_key_env": "TEST_KEY"},
            "concurrency": {"group": "shared", "max_sessions": limit},
        }
    )


def _config() -> CampaignConfig:
    return CampaignConfig.model_validate(
        {
            "name": "stable",
            "models": ["model-a"],
            "benchmarks": [{"id": "bench"}],
            "harnesses": ["harness-a", "harness-b"],
            "repetitions": 2,
            "execution": {"queue_capacity": 3, "n_concurrent_trials": 3},
        }
    )


def test_expansion_is_deterministic_and_round_robin() -> None:
    kwargs = {
        "profiles": {"model-a": _profile()},
        "selected_task_ids": {"bench": ["one", "two"]},
        "harness_requirements": {
            "harness-a": {"protocols": ["openai_chat_completions"]},
            "harness-b": {"protocols": ["openai_chat_completions"]},
        },
    }
    first = expand_campaign(_config(), **kwargs)
    second = expand_campaign(_config(), **kwargs)
    assert first == second
    assert len(first.cells) == 2
    assert len(first.trials) == 8
    assert first.trials[0].cell_id != first.trials[1].cell_id


def test_incompatible_combination_fails_instead_of_skipping() -> None:
    with pytest.raises(UnsupportedCombination, match="exposes"):
        expand_campaign(
            _config(),
            profiles={"model-a": _profile()},
            selected_task_ids={"bench": ["one"]},
            harness_requirements={
                "harness-a": {"protocols": ["anthropic_messages"]},
                "harness-b": {"protocols": ["openai_chat_completions"]},
            },
        )


def test_shared_group_must_have_one_limit() -> None:
    config = _config().model_copy(update={"models": ["model-a", "model-b"]})
    with pytest.raises(ValueError, match="conflicting limits"):
        expand_campaign(
            config,
            profiles={"model-a": _profile(), "model-b": _profile("model-b", limit=5)},
            selected_task_ids={"bench": ["one"]},
            harness_requirements={
                "harness-a": {"protocols": ["openai_chat_completions"]},
                "harness-b": {"protocols": ["openai_chat_completions"]},
            },
        )


@pytest.mark.parametrize("profile_name", ["glm-5.3", "gpt-5.6-sol"])
def test_one_bundled_profile_supports_openai_and_claude_harnesses(
    profile_name: str,
) -> None:
    task_root = Path(__file__).resolve().parents[3]
    profile = load_model_profile(task_root / "models" / f"{profile_name}.yaml")
    config = CampaignConfig.model_validate(
        {
            "name": "claude-gateway",
            "models": [profile_name],
            "benchmarks": [{"id": "bench"}],
            "harnesses": ["openai-agents", "claude-sdk"],
            "execution": {"queue_capacity": 2, "n_concurrent_trials": 2},
        }
    )
    plan = expand_campaign(
        config,
        profiles={profile_name: profile},
        selected_task_ids={"bench": ["one"]},
        harness_requirements={
            "openai-agents": {
                "protocols": ["openai_chat_completions"],
                "capabilities": {"tools": True},
            },
            "claude-sdk": {
                "protocols": ["anthropic_messages"],
                "capabilities": {"tools": True},
            },
        },
    )
    assert len(plan.cells) == 2
    assert len(plan.trials) == 2
