"""Deterministic model x benchmark x harness expansion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ageneval.model.gateway import ModelProfile, ModelProtocol

from .schema import CampaignConfig, MatrixExclude


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    model: str
    benchmark: str
    harness: str
    profile_digest: str


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    cell_id: str
    task_id: str
    repetition: int


@dataclass(frozen=True)
class CampaignPlan:
    campaign_id: str
    cells: tuple[CellSpec, ...]
    trials: tuple[TrialSpec, ...]


class UnsupportedCombination(ValueError):
    pass


def stable_id(prefix: str, payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(encoded.encode()).hexdigest()[:length]}"


def _excluded(rule: MatrixExclude, *, model: str, benchmark: str, harness: str) -> bool:
    return all(
        value is None or value == actual
        for value, actual in (
            (rule.model, model),
            (rule.benchmark, benchmark),
            (rule.harness, harness),
        )
    )


def _validate_compatibility(
    profile: ModelProfile, harness: str, requirements: Mapping[str, Any]
) -> None:
    protocols = set(
        requirements.get("protocols")
        or [
            ModelProtocol.OPENAI_CHAT_COMPLETIONS.value,
            ModelProtocol.ANTHROPIC_MESSAGES.value,
        ]
    )
    exposed_protocols = {protocol.value for protocol in profile.exposed_protocols()}
    if protocols.isdisjoint(exposed_protocols):
        raise UnsupportedCombination(
            f"harness {harness!r} accepts protocols {sorted(protocols)} but model "
            f"{profile.id!r} exposes {sorted(exposed_protocols)}"
        )
    required_caps = requirements.get("capabilities") or {}
    missing = [
        name
        for name, required in required_caps.items()
        if required and not bool(getattr(profile.capabilities, name, False))
    ]
    if missing:
        raise UnsupportedCombination(
            f"harness {harness!r} requires model capabilities {missing}; "
            f"profile {profile.id!r} does not provide them"
        )


def _round_robin_trials(per_cell: list[list[TrialSpec]]) -> tuple[TrialSpec, ...]:
    result: list[TrialSpec] = []
    index = 0
    while True:
        added = False
        for trials in per_cell:
            if index < len(trials):
                result.append(trials[index])
                added = True
        if not added:
            return tuple(result)
        index += 1


def expand_campaign(
    config: CampaignConfig,
    *,
    profiles: Mapping[str, ModelProfile],
    selected_task_ids: Mapping[str, Sequence[str]],
    harness_requirements: Mapping[str, Mapping[str, Any]],
) -> CampaignPlan:
    canonical = config.model_dump(mode="json", exclude_none=False)
    campaign_id = stable_id("campaign", canonical)
    cells: list[CellSpec] = []
    per_cell_trials: list[list[TrialSpec]] = []
    limits_by_group: dict[str, int] = {}

    for model_name in config.models:
        profile = profiles[model_name]
        previous = limits_by_group.setdefault(
            profile.concurrency.group, profile.concurrency.max_sessions
        )
        if previous != profile.concurrency.max_sessions:
            raise ValueError(
                f"model concurrency group {profile.concurrency.group!r} has conflicting "
                f"limits: {previous} and {profile.concurrency.max_sessions}"
            )
        for benchmark_config in config.benchmarks:
            benchmark = benchmark_config.id
            task_ids = selected_task_ids[benchmark]
            for harness in config.harnesses:
                if any(
                    _excluded(rule, model=model_name, benchmark=benchmark, harness=harness)
                    for rule in config.matrix.exclude
                ):
                    continue
                _validate_compatibility(profile, harness, harness_requirements.get(harness, {}))
                cell_payload = {
                    "model": model_name,
                    "benchmark": benchmark_config.model_dump(mode="json"),
                    "harness": harness,
                    "profile_digest": profile.digest(),
                }
                cell_id = stable_id("cell", cell_payload)
                cell = CellSpec(
                    cell_id=cell_id,
                    model=model_name,
                    benchmark=benchmark,
                    harness=harness,
                    profile_digest=profile.digest(),
                )
                cells.append(cell)
                trials: list[TrialSpec] = []
                for repetition in range(1, config.repetitions + 1):
                    for task_id in task_ids:
                        trial_id = stable_id(
                            "trial",
                            [campaign_id, cell_id, task_id, repetition],
                        )
                        trials.append(
                            TrialSpec(
                                trial_id=trial_id,
                                cell_id=cell_id,
                                task_id=task_id,
                                repetition=repetition,
                            )
                        )
                per_cell_trials.append(trials)
    if not cells:
        raise ValueError("campaign matrix is empty after exclusions")
    return CampaignPlan(
        campaign_id=campaign_id,
        cells=tuple(cells),
        trials=_round_robin_trials(per_cell_trials),
    )
