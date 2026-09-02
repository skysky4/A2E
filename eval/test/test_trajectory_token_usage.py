"""Tests for trajectory-fair total_token_usage."""

from __future__ import annotations

import unittest

from core.trajectory_token_usage import (
    agent_turn_count,
    instruction_text,
    loop_estimated_total_tokens,
    trajectory_total_tokens,
)
from result_values.efficiency_eval import make_total_token_usage


class TrajectoryTokenUsageTests(unittest.TestCase):
    def test_tau_retail_counts_system_instruction_and_answer(self) -> None:
        total, source = trajectory_total_tokens(
            benchmark="tau3-agno-gpt-5.6-sol",
            input_payload={"instruction": "You are Yusuf Rossi in 19122.", "initial_state": {}},
            output={"task_output": {"final_answer": "Done.", "tool_calls": ["find_user_id_by_name_zip"]}},
            example_metadata={"task_id": "tau3-00000-retail-0", "domain": "retail"},
        )
        self.assertGreater(total, 1000)
        self.assertIn("trajectory_fair", source)
        self.assertIn("family=tau", source)

    def test_deepsearchqa_counts(self) -> None:
        total, _ = trajectory_total_tokens(
            benchmark="deepsearchqa-agno",
            input_payload={"instruction": "Which countries changed scores?", "initial_state": {}},
            output={"task_output": {"final_answer": '{"final_answer":"NZ"}', "tool_calls": ["web_search"]}},
            example_metadata={"task_id": "deepsearchqa-eval-0000"},
        )
        self.assertGreater(total, 300)

    def test_make_total_token_usage_prefers_span_sum(self) -> None:
        spans = [
            {
                "span_kind": "LLM",
                "attributes": {"llm.token_count.prompt": 10000, "llm.token_count.completion": 5000},
            },
            {
                "span_kind": "LLM",
                "attributes": {"llm.token_count.prompt": 8000, "llm.token_count.completion": 3000},
            },
        ]
        metric = make_total_token_usage({"ex-1": spans}, benchmark="tau-bench-agno")
        result = metric(
            {"task_output": {"final_answer": "ok"}},
            {},
            {"instruction": "help me", "initial_state": {}},
            example={"id": "ex-1", "metadata": {"task_id": "retail-0000", "domain": "retail"}},
        )
        self.assertEqual(result["score"], 26000.0)
        self.assertIn("span_llm_sum", result["explanation"])

    def test_loop_estimated_scales_with_turns(self) -> None:
        common = {
            "benchmark": "tau3-agno-glm-5.3",
            "input_payload": {"instruction": "You are Yusuf Rossi in 19122.", "initial_state": {}},
            "example_metadata": {"task_id": "tau3-00000-retail-0", "domain": "retail"},
        }
        one_turn, one_src = loop_estimated_total_tokens(
            **common,
            output={"task_output": {"final_answer": "Done.", "turns": 1}},
        )
        three_turn, three_src = loop_estimated_total_tokens(
            **common,
            output={"task_output": {"final_answer": "Done.", "turns": 3, "tool_calls": ["a", "b", "c"]}},
        )
        self.assertIn("loop_estimated", one_src)
        self.assertIn("loop_estimated", three_src)
        self.assertGreater(three_turn, one_turn)

    def test_agent_turn_count_falls_back_to_tool_calls(self) -> None:
        self.assertEqual(
            agent_turn_count({"task_output": {"final_answer": "ok", "tool_calls": ["a", "b"]}}),
            2,
        )

    def test_make_total_token_usage_falls_back_to_loop_estimated_without_spans(self) -> None:
        metric = make_total_token_usage({}, benchmark="tau-bench-agno")
        result = metric(
            {"task_output": {"final_answer": "ok", "turns": 3}},
            {},
            {"instruction": "help me", "initial_state": {}},
            example={"id": "1", "metadata": {"task_id": "retail-0000", "domain": "retail"}},
        )
        self.assertGreater(result["score"], 1000)
        self.assertIn("loop_estimated", result["explanation"])

    def test_instruction_text_includes_initial_state(self) -> None:
        text = instruction_text({"instruction": "hi", "initial_state": {"x": 1}})
        self.assertIn("initial_state", text)


if __name__ == "__main__":
    unittest.main()
