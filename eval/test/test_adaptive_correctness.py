from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.deal_server import _detect_benchmark, _metrics_require_llm
from process_values.correct_eval import (
    correctness_rule_for_benchmark,
    make_adaptive_correctness,
    make_correctness,
    normalize_benchmark_name,
)


class FakeLLM:
    model = "deepseek-v4-pro"

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


class AdaptiveCorrectnessTest(unittest.TestCase):
    def test_normalizes_isolated_dataset_name(self) -> None:
        self.assertEqual(
            normalize_benchmark_name("ae2-mmlu-pro-20260724-123456"),
            "mmlu-pro",
        )

    def test_detects_benchmark_from_experiment_metadata(self) -> None:
        benchmark = _detect_benchmark(
            {"experiment_metadata": {"dataset": "gsm8k"}},
            {},
        )
        self.assertEqual(benchmark, "gsm8k")

    def test_detects_benchmark_from_actual_dataset_name(self) -> None:
        benchmark = _detect_benchmark(
            {"experiment_metadata": {"agent_framework": "agno"}},
            {"dataset_id": "RGF0YXNldDoyMw=="},
            dataset_name="ae2-qa-truthfulqa",
        )
        self.assertEqual(benchmark, "truthfulqa")

    def test_multiple_choice_rule(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="mmlu")
        result = evaluator(
            {"final_answer": '{"final_answer": "B"}'},
            {"expected_outputs": ["B"]},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:multiple_choice")

    def test_numeric_rule(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="gsm8k")
        result = evaluator(
            {"final_answer": "The answer is 1,200.0"},
            {"expected_outputs": ["1200"]},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:numeric")

    def test_exact_match_rule(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="bbh")
        result = evaluator(
            {"final_answer": " TRUE "},
            {"expected_outputs": ["true"]},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:exact_match")

    def test_rule_matches_any_valid_reference(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="bbh")
        result = evaluator(
            {"final_answer": '```json\n{"final_answer": "second"}\n```'},
            {"expected_outputs": ["first", "second"]},
            {},
        )
        self.assertEqual(result["score"], 1.0)

    def test_resolved_rule(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="terminal-bench-2")
        result = evaluator(
            {"task_output": {"resolved": True, "final_answer": "done"}},
            {},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:resolved")

    def test_missing_resolved_is_unscored(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="terminal-bench-2")
        result = evaluator({"task_output": {"status": "ok"}}, {}, {})
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")

    def test_unextractable_multiple_choice_is_unscored(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="mmlu")
        result = evaluator(
            {"final_answer": "The correct option is photosynthesis."},
            {"expected_outputs": ["B"]},
            {},
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")

    def test_unextractable_numeric_is_unscored(self) -> None:
        evaluator = make_adaptive_correctness(benchmark="gsm8k")
        result = evaluator(
            {"final_answer": "I cannot solve this yet."},
            {"expected_outputs": ["42"]},
            {},
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")

    def test_incomplete_action_sequence_falls_through_to_phoenix_llm_judge(self) -> None:
        llm = FakeLLM(
            "LABEL=incorrect; SCORE=0; EXPLANATION=Output does not achieve the ground-truth goal."
        )
        evaluator = make_adaptive_correctness(benchmark="tau-bench", llm=llm)
        result = evaluator(
            {"final_answer": '{"action": "lookup", "arguments": {"id": "1"}}'},
            {
                "expected_actions": [
                    {"name": "lookup", "arguments": {"id": "1"}},
                    {"name": "update", "arguments": {"id": "1"}},
                ]
            },
            {},
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "llm:phoenix_gt_correctness")
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("GROUND_TRUTH", llm.prompts[0])
        self.assertIn("OUTPUT", llm.prompts[0])
        self.assertIn("Phoenix Correctness", llm.prompts[0])

    def test_exact_action_sequence_is_deterministically_correct(self) -> None:
        llm = FakeLLM()
        evaluator = make_adaptive_correctness(benchmark="tau2", llm=llm)
        actions = [
            {"name": "lookup", "arguments": {"id": "1"}},
            {"name": "update", "arguments": {"id": "1", "value": "new"}},
        ]
        result = evaluator(
            {"tool_calls_full": actions, "final_answer": "done"},
            {"expected_actions": actions},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:action_sequence_exact")
        self.assertEqual(llm.prompts, [])

    def test_tool_call_records_are_used_as_actual_actions(self) -> None:
        llm = FakeLLM()
        evaluator = make_adaptive_correctness(benchmark="tau3", llm=llm)
        actions = [
            {"name": "lookup", "arguments": {"id": "1"}},
            {"name": "update", "arguments": {"id": "1", "value": "new"}},
        ]
        result = evaluator(
            {
                "tool_calls": ["lookup", "update"],
                "tool_call_records": actions,
                "final_answer": "done",
            },
            {"expected_actions": actions},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "rule:action_sequence_exact")
        self.assertEqual(llm.prompts, [])

    def test_llm_receives_all_ground_truth_and_actual_output(self) -> None:
        llm = FakeLLM(
            "LABEL=correct; SCORE=1; EXPLANATION=The second valid reference matches exactly."
        )
        evaluator = make_adaptive_correctness(benchmark="math", llm=llm)
        result = evaluator(
            {"final_answer": '{"final_answer": "beta"}'},
            {"expected_outputs": ["alpha", "beta"]},
            {"instruction": "This text must not be used to solve the task."},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertIn('"alpha"', llm.prompts[0])
        self.assertIn('"beta"', llm.prompts[0])
        self.assertIn('"final_answer": "beta"', llm.prompts[0])
        self.assertIn("GROUND_TRUTH", llm.prompts[0])
        self.assertIn("OUTPUT", llm.prompts[0])
        self.assertNotIn("This text must not be used", llm.prompts[0])
        self.assertEqual(result["metadata"]["evaluation_method"], "llm:phoenix_gt_correctness")
        self.assertEqual(len(result["metadata"]["judge_prompt_sha256"]), 64)

    def test_missing_ground_truth_uses_phoenix_native_correctness(self) -> None:
        llm = FakeLLM(
            "LABEL=correct; SCORE=1; EXPLANATION=The answer addresses the question accurately."
        )
        evaluator = make_adaptive_correctness(benchmark="persistbench", llm=llm)
        result = evaluator(
            {"final_answer": "Paris is the capital of France."},
            {},
            {"instruction": "What is the capital of France?"},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["label"], "correct")
        self.assertEqual(result["metadata"]["evaluation_method"], "llm:phoenix_correctness")
        self.assertEqual(result["metadata"]["ground_truth_type"], "missing")
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("factual accuracy and completeness", llm.prompts[0])
        self.assertIn("What is the capital of France?", llm.prompts[0])
        self.assertIn("Paris is the capital of France.", llm.prompts[0])
        self.assertNotIn("GROUND_TRUTH", llm.prompts[0])

    def test_make_llm_judge_without_gt_uses_phoenix_native(self) -> None:
        llm = FakeLLM(
            "LABEL=incorrect; SCORE=0; EXPLANATION=The response is incomplete."
        )
        evaluator = make_correctness(llm)
        result = evaluator(
            {"final_answer": "I don't know"},
            {},
            {"instruction": "Explain photosynthesis in detail."},
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metadata"]["evaluation_method"], "llm:phoenix_correctness")
        self.assertIn("Explain photosynthesis", llm.prompts[0])

    def test_tau_requires_llm(self) -> None:
        self.assertIsNone(correctness_rule_for_benchmark("tau-bench"))
        self.assertTrue(_metrics_require_llm(["correctness"], benchmark="tau-bench"))

    def test_rule_benchmark_does_not_require_llm(self) -> None:
        self.assertFalse(_metrics_require_llm(["correctness"], benchmark="mmlu"))


if __name__ == "__main__":
    unittest.main()
