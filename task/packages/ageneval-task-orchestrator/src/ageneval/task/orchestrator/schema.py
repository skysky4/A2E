"""Versioned public configuration and result models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int | None = Field(default=None, gt=0)
    seed: int | None = None
    task_ids: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(default_factory=list)


class GraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    mode: Literal["inline", "posthoc"] = "posthoc"
    required: bool = True
    model: str | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str | None = None
    sample: SampleConfig = Field(default_factory=SampleConfig)
    graders: list[GraderConfig] = Field(default_factory=list)
    args: dict[str, Any] = Field(default_factory=dict)


class MatrixExclude(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    benchmark: str | None = None
    harness: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> MatrixExclude:
        if self.model is None and self.benchmark is None and self.harness is None:
            raise ValueError("matrix exclude entry must select at least one dimension")
        return self


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude: list[MatrixExclude] = Field(default_factory=list)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=0, ge=0)
    min_wait_seconds: float = Field(default=1.0, ge=0)
    max_wait_seconds: float = Field(default=30.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(
        default_factory=lambda: [
            "AuthenticationError",
            "InvalidConfiguration",
            "UnsupportedCombination",
        ]
    )

    @model_validator(mode="after")
    def _wait_bounds(self) -> RetryPolicy:
        if self.max_wait_seconds < self.min_wait_seconds:
            raise ValueError("max_wait_seconds must be >= min_wait_seconds")
        overlap = set(self.include) & set(self.exclude)
        if overlap:
            raise ValueError(f"retry include/exclude overlap: {sorted(overlap)}")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_concurrent_trials: int = Field(default=3, gt=0)
    n_active_cells: int = Field(default=2, gt=0)
    n_concurrent_sandboxes: int = Field(default=2, gt=0)
    n_concurrent_model_sessions: int | None = Field(default=None, gt=0)
    n_concurrent_graders: int = Field(default=4, gt=0)
    n_concurrent_uploads: int = Field(default=8, gt=0)
    queue_capacity: int = Field(default=16, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    cancellation_grace_seconds: float = Field(default=30.0, gt=0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    @model_validator(mode="after")
    def _queue_covers_workers(self) -> ExecutionConfig:
        if self.queue_capacity < self.n_concurrent_trials:
            raise ValueError("queue_capacity must be >= n_concurrent_trials")
        return self


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retain: Literal["all", "failures", "none"] = "failures"


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str
    models: list[str]
    benchmarks: list[BenchmarkConfig]
    harnesses: list[str]
    repetitions: int = Field(default=1, gt=0)
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)

    @field_validator("schema_version")
    @classmethod
    def _version_is_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported campaign schema_version: {value}")
        return value

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("campaign name must not be empty")
        return value.strip()

    @field_validator("models", "harnesses")
    @classmethod
    def _unique_non_empty(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if not normalized or any(not value for value in normalized):
            raise ValueError("list must contain non-empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("list values must be unique")
        return normalized


class LifecycleEvent(str, Enum):
    START = "START"
    ENVIRONMENT_START = "ENVIRONMENT_START"
    ENVIRONMENT_END = "ENVIRONMENT_END"
    AGENT_START = "AGENT_START"
    AGENT_END = "AGENT_END"
    VERIFICATION_START = "VERIFICATION_START"
    VERIFICATION_END = "VERIFICATION_END"
    UPLOAD_START = "UPLOAD_START"
    UPLOAD_END = "UPLOAD_END"
    END = "END"
    CANCEL = "CANCEL"


class LifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: LifecycleEvent
    timestamp: datetime
    attempt: int | None = None


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mode: Literal["inline", "posthoc"]
    annotator_kind: Literal["LLM", "CODE", "HUMAN"] = "CODE"
    required: bool = True
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    trace_id: str | None = None
    start_time: datetime
    end_time: datetime


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    cell_id: str
    task_id: str
    repetition: int
    attempt: int
    status: Literal[
        "pending",
        "running",
        "local_complete",
        "uploading",
        "completed",
        "failed",
        "cancelled",
        "incompatible",
    ]
    output: dict[str, Any] = Field(default_factory=dict)
    grades: list[GradeResult] = Field(default_factory=list)
    trace_id: str | None = None
    error: str | None = None
    error_type: str | None = None
    retryable: bool = False
    uploaded: bool = False
    experiment_run_id: str | None = None
    lifecycle: list[LifecycleRecord] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime


def load_campaign(path: str | Path) -> CampaignConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid campaign YAML {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"campaign must be a mapping: {config_path}")
    return CampaignConfig.model_validate(payload)
