"""Versioned, secret-safe model profiles."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class ModelProtocol(str, Enum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class ConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url_env: str | None = None
    api_key_env: str


class CapabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: bool = True
    streaming: bool = True
    vision: bool = False
    structured_output: bool = False


class ConcurrencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str
    max_sessions: int = Field(default=1, gt=0)


class GatewayConfig(BaseModel):
    """Protocols exposed by one managed loopback endpoint for this model."""

    model_config = ConfigDict(extra="forbid")

    interfaces: list[ModelProtocol]

    @field_validator("interfaces")
    @classmethod
    def _interfaces_are_nonempty_and_unique(
        cls, values: list[ModelProtocol]
    ) -> list[ModelProtocol]:
        if not values:
            raise ValueError("gateway.interfaces must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("gateway.interfaces entries must be unique")
        return values


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1)
    id: str
    provider: str
    model: str
    upstream_protocol: ModelProtocol
    connection: ConnectionConfig
    capabilities: CapabilityConfig = Field(default_factory=CapabilityConfig)
    concurrency: ConcurrencyConfig
    gateway: GatewayConfig | None = None
    middleware: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_protocol(cls, value: Any) -> Any:
        """Accept schema-v1 profiles written before upstream was made explicit."""
        if not isinstance(value, dict) or "protocol" not in value:
            return value
        if "upstream_protocol" in value:
            raise ValueError("use upstream_protocol, not both protocol fields")
        migrated = dict(value)
        migrated["upstream_protocol"] = migrated.pop("protocol")
        return migrated

    @field_validator("schema_version")
    @classmethod
    def _version_is_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported model profile schema_version: {value}")
        return value

    @field_validator("id", "provider", "model")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("middleware")
    @classmethod
    def _middleware_is_known(cls, values: list[str]) -> list[str]:
        unknown = set(values) - {"glm_tool_call_compat"}
        if unknown:
            raise ValueError(f"unknown middleware: {sorted(unknown)}")
        if len(values) != len(set(values)):
            raise ValueError("middleware entries must be unique")
        return values

    @model_validator(mode="after")
    def _middleware_matches_protocol(self) -> ModelProfile:
        if (
            "glm_tool_call_compat" in self.middleware
            and self.upstream_protocol is not ModelProtocol.OPENAI_CHAT_COMPLETIONS
        ):
            raise ValueError("glm_tool_call_compat requires openai_chat_completions")
        if len(self.middleware) > 1:
            raise ValueError("model profiles currently support one middleware runtime")
        if self.gateway is not None:
            interfaces = set(self.gateway.interfaces)
            if self.upstream_protocol not in interfaces:
                raise ValueError("gateway.interfaces must include the model upstream_protocol")
            supported = {self.upstream_protocol}
            if self.upstream_protocol is ModelProtocol.OPENAI_CHAT_COMPLETIONS:
                supported.add(ModelProtocol.ANTHROPIC_MESSAGES)
            unsupported = interfaces - supported
            if unsupported:
                raise ValueError(
                    "gateway cannot expose interfaces "
                    f"{sorted(item.value for item in unsupported)} from upstream "
                    f"{self.upstream_protocol.value}"
                )
        return self

    @property
    def protocol(self) -> ModelProtocol:
        """Backward-compatible name for callers that need the upstream protocol."""
        return self.upstream_protocol

    def exposed_protocols(self) -> frozenset[ModelProtocol]:
        if self.gateway is None:
            return frozenset({self.upstream_protocol})
        return frozenset(self.gateway.interfaces)

    def public_dict(self) -> dict[str, Any]:
        """Return the lock-file representation; it contains env names, never values."""
        return self.model_dump(mode="json")

    def digest(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ResolvedModel(BaseModel):
    """Runtime-only model data. Do not serialize this object into run files."""

    model_config = ConfigDict(extra="forbid")

    profile: ModelProfile
    base_url: str | None = None
    api_key: SecretStr

    def agent_kwargs(self) -> dict[str, str]:
        result = {
            "model": self.profile.model,
            "api_key": self.api_key.get_secret_value(),
        }
        if self.base_url:
            result["api_base"] = self.base_url
        return result


def _validate_url(value: str, *, variable: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{variable} must contain an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{variable} must not embed credentials")
    return value.rstrip("/")


def resolve_model(profile: ModelProfile, environ: dict[str, str] | None = None) -> ResolvedModel:
    env = os.environ if environ is None else environ
    key_name = profile.connection.api_key_env
    api_key = env.get(key_name)
    if not api_key:
        raise ValueError(f"required credential environment variable is missing: {key_name}")
    base_url = None
    if profile.connection.base_url_env:
        base_name = profile.connection.base_url_env
        value = env.get(base_name)
        if not value:
            raise ValueError(f"required endpoint environment variable is missing: {base_name}")
        base_url = _validate_url(value, variable=base_name)
    return ResolvedModel(profile=profile, base_url=base_url, api_key=api_key)


def load_model_profile(path: str | Path) -> ModelProfile:
    profile_path = Path(path)
    try:
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid model profile YAML {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"model profile must be a mapping: {profile_path}")
    return ModelProfile.model_validate(payload)
