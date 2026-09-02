"""Declarative benchmark profiles discovered from dataset packages."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComponentConfig(BaseModel):
    """One lazily imported benchmark component."""

    model_config = ConfigDict(extra="forbid")

    entrypoint: str
    defaults: dict[str, Any] = Field(default_factory=dict)
    accepts: list[str] | None = None
    argument_map: dict[str, str] = Field(default_factory=dict)

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint_has_attribute(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("entrypoint must use 'module:attribute' syntax")
        return value


class BenchmarkDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=8, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    llm_timeout: float = Field(default=180.0, gt=0)
    run_deadline: float = Field(default=1800.0, gt=0)
    wall: float | None = Field(default=None, gt=0)


class BenchmarkResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker: bool = False
    network: bool = False


class BenchmarkProfile(BaseModel):
    """Versioned, public configuration owned by one benchmark."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    aliases: list[str] = Field(default_factory=list)
    kind: Literal["qa", "tool", "sandbox"]
    loader: ComponentConfig
    binding: ComponentConfig
    grader: ComponentConfig
    session: ComponentConfig | None = None
    setup: ComponentConfig | None = None
    defaults: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)
    agent_overrides: dict[str, Any] = Field(default_factory=dict)
    resources: BenchmarkResources = Field(default_factory=BenchmarkResources)

    @field_validator("schema_version")
    @classmethod
    def _version_is_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported benchmark profile schema_version: {value}")
        return value

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("benchmark profile id must not be empty")
        return value

    def digest(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def default_benchmark_root() -> Path:
    """Return the repository dataset directory, with an explicit override."""
    configured = os.environ.get("A2E_BENCHMARKS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # .../task/runners/src/ageneval/task/runners/benchmark_profile.py -> task/
    return Path(__file__).resolve().parents[5] / "datasets"


def load_benchmark_profile(path: str | Path) -> BenchmarkProfile:
    profile_path = Path(path)
    try:
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid benchmark YAML {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark profile must be a mapping: {profile_path}")
    return BenchmarkProfile.model_validate(payload)


def discover_benchmark_profiles(
    root: str | Path | None = None,
) -> dict[str, tuple[BenchmarkProfile, Path]]:
    """Discover canonical profiles and aliases below ``task/datasets``."""
    base = Path(root).resolve() if root is not None else default_benchmark_root()
    paths = sorted({*base.glob("*/benchmark.yaml"), *base.glob("*/benchmarks/*.yaml")})
    if not paths:
        raise FileNotFoundError(f"no benchmark profiles found under {base}")
    profiles: dict[str, tuple[BenchmarkProfile, Path]] = {}
    for path in paths:
        profile = load_benchmark_profile(path)
        for key in (profile.id, *profile.aliases):
            if key in profiles:
                raise ValueError(
                    f"duplicate benchmark profile id or alias {key!r}: "
                    f"{profiles[key][1]} and {path}"
                )
            profiles[key] = (profile, path.resolve())
    return profiles


def resolve_entrypoint(reference: str) -> Any:
    module_name, attribute = reference.split(":", 1)
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def component_callable(component: ComponentConfig) -> Callable[..., Any]:
    """Return a lazy wrapper that merges defaults and filters keyword args."""

    def invoke(*args: Any, **kwargs: Any) -> Any:
        target = resolve_entrypoint(component.entrypoint)
        if not callable(target):
            if args or kwargs or component.defaults:
                raise TypeError(f"entrypoint {component.entrypoint!r} is not callable")
            return target
        merged = {**component.defaults, **kwargs}
        for source, destination in component.argument_map.items():
            if source in merged and destination not in merged:
                merged[destination] = merged.pop(source)
        if component.accepts is not None:
            accepted = set(component.accepts)
            merged = {key: value for key, value in merged.items() if key in accepted}
        return target(*args, **merged)

    return invoke


def resolve_component(component: ComponentConfig) -> Any:
    """Resolve a static object or invoke a configured factory."""
    target = resolve_entrypoint(component.entrypoint)
    if callable(target):
        return target(**component.defaults)
    if component.defaults:
        raise TypeError(f"static entrypoint {component.entrypoint!r} cannot have defaults")
    return target


def build_dataset_registry(
    profiles: Mapping[str, tuple[BenchmarkProfile, Path]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the legacy registry interface from declarative profiles."""
    discovered = dict(profiles or discover_benchmark_profiles())
    registry: dict[str, dict[str, Any]] = {}
    for key, (profile, path) in discovered.items():
        entry: dict[str, Any] = {
            "load": component_callable(profile.loader),
            "bind": component_callable(profile.binding),
            "kind": profile.kind,
            "official_settings": profile.defaults.model_dump(exclude_none=True),
            "agent_overrides": dict(profile.agent_overrides),
            "profile": profile,
            "profile_path": path,
            "profile_digest": profile.digest(),
        }
        if profile.setup is not None:
            entry["setup"] = component_callable(profile.setup)
        registry[key] = entry
    return registry


__all__ = [
    "BenchmarkDefaults",
    "BenchmarkProfile",
    "BenchmarkResources",
    "ComponentConfig",
    "build_dataset_registry",
    "component_callable",
    "default_benchmark_root",
    "discover_benchmark_profiles",
    "load_benchmark_profile",
    "resolve_component",
    "resolve_entrypoint",
]
