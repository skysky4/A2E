"""DeepSearchQA dataset adapter for A2E."""

from ageneval.task.datasets.deepsearchqa.binding import build_deepsearchqa_binding
from ageneval.task.datasets.deepsearchqa.grader import (
    GRADER,
    GRADER_METADATA,
    deepsearch_grader,
)
from ageneval.task.datasets.deepsearchqa.loader import (
    DeepSearchQADataset,
    load_deepsearchqa_tasks,
)
from ageneval.task.datasets.deepsearchqa.session import wrap_dsqa_official_session

__all__ = [
    "DeepSearchQADataset",
    "GRADER",
    "GRADER_METADATA",
    "build_deepsearchqa_binding",
    "deepsearch_grader",
    "load_deepsearchqa_tasks",
    "wrap_dsqa_official_session",
]
