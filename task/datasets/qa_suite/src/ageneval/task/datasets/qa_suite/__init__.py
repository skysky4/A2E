"""QA Suite — 10 config-driven pure question-answering benchmarks for A2E."""

from ageneval.task.datasets.qa_suite.benchmarks import BBH_CONFIGS, BENCHMARKS, QABenchmark
from ageneval.task.datasets.qa_suite.binding import build_qa_binding
from ageneval.task.datasets.qa_suite.grader import (
    grader_for_benchmark,
    grade_qa_suite,
    normalize_bbh_answer,
    normalize_math_number,
)
from ageneval.task.datasets.qa_suite.loader import QADataset, load_qa_tasks

__all__ = [
    "BBH_CONFIGS",
    "BENCHMARKS",
    "QABenchmark",
    "QADataset",
    "build_qa_binding",
    "grade_qa_suite",
    "grader_for_benchmark",
    "load_qa_tasks",
    "normalize_bbh_answer",
    "normalize_math_number",
]
