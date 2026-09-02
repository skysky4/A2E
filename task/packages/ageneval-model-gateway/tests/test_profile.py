from __future__ import annotations

import json
from pathlib import Path

import pytest
from ageneval.model.gateway import ModelProfile, ModelRuntime, load_model_profile, resolve_model


def _profile() -> ModelProfile:
    return ModelProfile.model_validate(
        {
            "schema_version": 1,
            "id": "test-model",
            "provider": "test",
            "model": "model-1",
            "upstream_protocol": "openai_chat_completions",
            "connection": {
                "base_url_env": "TEST_MODEL_URL",
                "api_key_env": "TEST_MODEL_KEY",
            },
            "capabilities": {"tools": True},
            "concurrency": {"group": "test", "max_sessions": 3},
        }
    )


def test_profile_public_form_never_contains_secret() -> None:
    profile = _profile()
    resolved = resolve_model(
        profile,
        {"TEST_MODEL_URL": "https://models.test/v1", "TEST_MODEL_KEY": "secret-value"},
    )
    serialized = json.dumps(profile.public_dict())
    assert "secret-value" not in serialized
    assert resolved.api_key.get_secret_value() == "secret-value"
    assert resolved.agent_kwargs()["api_base"] == "https://models.test/v1"


def test_profile_rejects_credentials_embedded_in_url() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        resolve_model(
            _profile(),
            {
                "TEST_MODEL_URL": "https://user:pass@models.test/v1",
                "TEST_MODEL_KEY": "secret",
            },
        )


def test_load_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema_version: 1
id: test
provider: test
model: test
protocol: openai_chat_completions
connection: {api_key_env: TEST_KEY}
concurrency: {group: test, max_sessions: 1}
unexpected: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_model_profile(path)


def test_runtime_starts_declared_middleware_on_loopback() -> None:
    profile = _profile().model_copy(update={"middleware": ["glm_tool_call_compat"]})
    resolved = resolve_model(
        profile,
        {"TEST_MODEL_URL": "http://127.0.0.1:9/v1", "TEST_MODEL_KEY": "secret"},
    )
    runtime = ModelRuntime(resolved)
    try:
        active = runtime.start()
        assert active.base_url is not None
        assert active.base_url.startswith("http://127.0.0.1:")
        assert active.base_url.endswith("/v1")
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "name",
    [
        "openai",
        "openai-compatible",
        "anthropic-compatible",
        "glm-5.3",
        "gpt-5.6-sol",
    ],
)
def test_bundled_profiles_are_valid(name: str) -> None:
    task_root = Path(__file__).resolve().parents[3]
    profile = load_model_profile(task_root / "models" / f"{name}.yaml")
    assert profile.id == name


def test_gateway_interfaces_must_include_upstream_protocol() -> None:
    with pytest.raises(ValueError, match="must include the model upstream_protocol"):
        ModelProfile.model_validate(
            {
                **_profile().model_dump(mode="json"),
                "gateway": {"interfaces": ["anthropic_messages"]},
            }
        )


def test_runtime_starts_one_multi_protocol_gateway() -> None:
    payload = _profile().model_dump(mode="json")
    payload["gateway"] = {"interfaces": ["openai_chat_completions", "anthropic_messages"]}
    profile = ModelProfile.model_validate(payload)
    resolved = resolve_model(
        profile,
        {"TEST_MODEL_URL": "http://127.0.0.1:9/v1", "TEST_MODEL_KEY": "secret"},
    )
    runtime = ModelRuntime(resolved)
    try:
        active = runtime.start()
        assert active.base_url is not None
        assert active.base_url.startswith("http://127.0.0.1:")
        assert active.base_url.endswith("/v1")
    finally:
        runtime.close()


def test_legacy_protocol_field_is_normalized_to_upstream_protocol() -> None:
    payload = _profile().model_dump(mode="json")
    payload["protocol"] = payload.pop("upstream_protocol")
    profile = ModelProfile.model_validate(payload)
    assert profile.upstream_protocol.value == "openai_chat_completions"
    assert "protocol" not in profile.public_dict()
    assert profile.public_dict()["upstream_protocol"] == "openai_chat_completions"
