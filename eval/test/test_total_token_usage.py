"""Tests for total_token_usage: output first, LLM span sum fallback."""

from __future__ import annotations

import unittest

from result_values.efficiency_eval import make_total_token_usage


class TotalTokenUsageTests(unittest.TestCase):
    def test_trajectory_fair_counts_binding_system_plus_io(self) -> None:
        metric = make_total_token_usage({}, benchmark="tau-bench-agno")
        result = metric(
            {"task_output": {"final_answer": "Exchange complete."}},
            {},
            {"instruction": "You are Yusuf Rossi in 19122.", "initial_state": {}},
            example={"id": "ex-1", "metadata": {"task_id": "retail-0000", "domain": "retail"}},
        )
        self.assertGreater(result["score"], 1000)
        self.assertIn("loop_estimated", result["explanation"])

    def test_gdpval_long_instruction_increases_score(self) -> None:
        metric = make_total_token_usage({}, benchmark="gdpval-aa-agno")
        short = metric(
            {"task_output": {"final_answer": "memo"}},
            {},
            {"instruction": "Write a memo.", "initial_state": {}},
            example={"id": "1", "metadata": {"task_id": "uuid-1"}},
        )
        long_instr = "Write a memo. " + ("detail " * 500)
        long = metric(
            {"task_output": {"final_answer": "memo"}},
            {},
            {"instruction": long_instr, "initial_state": {}},
            example={"id": "2", "metadata": {"task_id": "uuid-2"}},
        )
        self.assertGreater(long["score"], short["score"])


if __name__ == "__main__":
    unittest.main()
