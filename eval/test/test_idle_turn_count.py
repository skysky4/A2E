"""Tests for idle_turn_count (RedundancyBench empty-tool-call agent steps)."""

from __future__ import annotations

import unittest

from core.trajectory_token_usage import (
    agent_tool_step_count,
    idle_turn_count,
    label_for_idle_turn_count,
)
from result_values.efficiency_eval import make_idle_turn_count


class IdleTurnCountTests(unittest.TestCase):
    def test_messages_path_counts_assistant_empty_toolcall(self) -> None:
        count, source = idle_turn_count(
            {
                "task_output": {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "Let me check.", "tool_calls": []},
                        {
                            "role": "assistant",
                            "content": "Calling tool",
                            "tool_calls": [{"name": "find_user", "arguments": {"x": 1}}],
                        },
                        {"role": "tool", "content": "ok"},
                        {"role": "assistant", "content": "Done."},
                    ]
                }
            }
        )
        self.assertEqual(count, 2)
        self.assertIn("messages", source)
        self.assertIn("agent_steps=3", source)

    def test_messages_path_ignores_user_steps(self) -> None:
        count, _ = idle_turn_count(
            {
                "task_output": {
                    "messages": [
                        {"role": "user", "content": "confirm?", "tool_calls": [{"name": "noop"}]},
                        {"role": "assistant", "tool_calls": [{"name": "search", "arguments": {}}]},
                    ]
                }
            }
        )
        self.assertEqual(count, 0)

    def test_turns_minus_tool_steps_proxy(self) -> None:
        count, source = idle_turn_count(
            {
                "task_output": {
                    "turns": 13,
                    "tool_spans": [
                        {"name": "a", "arguments": {}},
                        {"name": "b", "arguments": {}},
                        {"name": "c", "arguments": {}},
                        {"name": "d", "arguments": {}},
                        {"name": "e", "arguments": {}},
                        {"name": "f", "arguments": {}},
                    ],
                }
            }
        )
        self.assertEqual(count, 7)
        self.assertIn("Sec.3 proxy", source)
        self.assertIn("agent_turns=13", source)
        self.assertIn("agent_tool_steps=6", source)

    def test_tool_steps_without_turns_field_is_clean(self) -> None:
        count, source = idle_turn_count(
            {"task_output": {"tool_spans": [{"name": "web_search", "arguments": {"q": "x"}}]}}
        )
        self.assertEqual(count, 0)
        self.assertIn("turns field is missing", source)

    def test_unscored_without_turns_or_tools_or_messages(self) -> None:
        count, source = idle_turn_count({"task_output": {"final_answer": "ok"}})
        self.assertIsNone(count)
        self.assertIn("cannot score", source)

    def test_labels(self) -> None:
        self.assertEqual(label_for_idle_turn_count(0), "0")
        self.assertEqual(label_for_idle_turn_count(2), "2")
        self.assertEqual(label_for_idle_turn_count(6), "6")

    def test_make_idle_turn_count_evaluator(self) -> None:
        metric = make_idle_turn_count()
        result = metric(
            {"task_output": {"turns": 5, "tool_spans": [{"name": "t", "arguments": {}}]}},
            {},
            {},
        )
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["label"], "4")

    def test_agent_tool_step_count_prefers_tool_spans(self) -> None:
        self.assertEqual(
            agent_tool_step_count(
                {
                    "task_output": {
                        "tool_spans": [{"name": "a", "arguments": {}}],
                        "tool_calls": ["a", "b"],
                    }
                }
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
