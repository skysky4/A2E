from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.native_metrics import (
    extract_native_metric,
    merge_upstream_eval_annotations,
    prefer_trajectory_native_metric,
)


class NativeMetricsTest(unittest.TestCase):
    def test_extract_accuracy_as_correctness(self) -> None:
        result = extract_native_metric(
            "correctness",
            {"metrics": {"accuracy": 1.0}, "final_answer": "wrong on purpose"},
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "native:accuracy")

    def test_prefer_native_on_conflict(self) -> None:
        output = {"metrics": {"accuracy": 1.0}, "resolved": False}

        def compute(_output: dict, _expected: dict, _input: dict) -> dict:
            return {
                "score": 0.0,
                "label": "incorrect",
                "explanation": "post-hoc eval says incorrect",
            }

        result = prefer_trajectory_native_metric("correctness", compute, output, {}, {})
        self.assertEqual(result["score"], 1.0)
        self.assertIn("Overrode post-hoc eval", result["explanation"])

    def test_use_computed_when_no_native(self) -> None:
        def compute(_output: dict, _expected: dict, _input: dict) -> dict:
            return {"score": 0.0, "label": "incorrect", "explanation": "computed"}

        result = prefer_trajectory_native_metric("correctness", compute, {"final_answer": "x"}, {}, {})
        self.assertEqual(result["score"], 0.0)

    def test_use_computed_when_native_agrees(self) -> None:
        output = {"metrics": {"accuracy": 1.0}}

        def compute(_output: dict, _expected: dict, _input: dict) -> dict:
            return {"score": 1.0, "label": "correct", "explanation": "computed"}

        result = prefer_trajectory_native_metric("correctness", compute, output, {}, {})
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["explanation"], "computed")

    def test_merge_upstream_exact_match_into_metrics(self) -> None:
        experiment = {
            "task_runs": [
                {
                    "id": "run-1",
                    "output": {"final_answer": "B"},
                }
            ],
            "evaluation_runs": [
                {
                    "experiment_run_id": "run-1",
                    "name": "exact_match",
                    "result": {"score": 1.0, "label": "match", "explanation": "upstream"},
                }
            ],
        }
        merge_upstream_eval_annotations(experiment)
        metrics = experiment["task_runs"][0]["output"]["metrics"]
        self.assertEqual(metrics["correctness"]["score"], 1.0)
        self.assertEqual(metrics["exact_match"]["score"], 1.0)

    def test_resolved_bool_maps_to_correctness(self) -> None:
        result = extract_native_metric("correctness", {"resolved": True})
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
