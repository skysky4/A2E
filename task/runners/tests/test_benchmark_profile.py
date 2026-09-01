from __future__ import annotations

from pathlib import Path

from ageneval.task.runners.benchmark_profile import (
    ComponentConfig,
    component_callable,
    discover_benchmark_profiles,
    load_benchmark_profile,
)


def test_repository_profiles_cover_registry() -> None:
    profiles = discover_benchmark_profiles()

    assert "tau-bench" in profiles
    assert "deepsearchqa" in profiles
    assert "terminal-bench-2.1" in profiles
    assert "tau3bench" in profiles
    assert profiles["tau3bench"][0].id == "tau3"


def test_tau_profile_owns_user_simulator_configuration() -> None:
    profile, path = discover_benchmark_profiles()["tau-bench"]

    assert path.name == "benchmark.yaml"
    assert profile.session is not None
    assert profile.session.defaults == {
        "user_strategy": "llm",
        "user_model": "deepseek-v4-flash",
        "user_error_policy": "fail",
        "max_responds": 12,
    }


def test_component_maps_campaign_task_ids_to_loader_argument(tmp_path: Path) -> None:
    module = tmp_path / "profile_target.py"
    module.write_text("def load(*, instance_ids=None):\n    return instance_ids\n")
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        component = ComponentConfig(
            entrypoint="profile_target:load",
            accepts=["instance_ids"],
            argument_map={"task_ids": "instance_ids"},
        )
        assert component_callable(component)(task_ids=["one"]) == ["one"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("profile_target", None)


def test_profile_digest_changes_with_behavior(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "datasets/tau_bench/benchmark.yaml"
    profile = load_benchmark_profile(source)
    changed = profile.model_copy(
        update={
            "agent_overrides": {**profile.agent_overrides, "max_turns": 31},
        }
    )

    assert profile.digest() != changed.digest()
