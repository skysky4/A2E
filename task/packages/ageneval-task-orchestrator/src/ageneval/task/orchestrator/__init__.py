"""Campaign planning, execution and recovery for A2E."""

from typing import Any

from .matrix import CampaignPlan, CellSpec, TrialSpec, expand_campaign
from .schema import (
    CampaignConfig,
    GradeResult,
    LifecycleEvent,
    LifecycleRecord,
    TrialResult,
)
from .state import RunDirectory

__all__ = [
    "CampaignConfig",
    "CampaignController",
    "CampaignPlan",
    "CellSpec",
    "GradeResult",
    "LifecycleEvent",
    "LifecycleRecord",
    "RunDirectory",
    "TrialResult",
    "TrialSpec",
    "expand_campaign",
]


def __getattr__(name: str) -> Any:
    # Keep worker-side imports lightweight. In particular, the standalone
    # AutoGen environment can use schema/executor modules without importing
    # a2e-client and the main Controller dependency graph.
    if name == "CampaignController":
        from .controller import CampaignController

        return CampaignController
    raise AttributeError(name)
