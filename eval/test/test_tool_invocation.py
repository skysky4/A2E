from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from process_values.tool_eval import make_tool_invocation


class FakeLLM:
    model = "test-judge"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return "LABEL=correct; SCORE=1; EXPLANATION=Arguments match the schema and user request."


_BASH_SCHEMA = {
    "name": "bash",
    "description": "Run a shell command",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


class ToolInvocationLogicTest(unittest.TestCase):
    def test_no_calls_is_unscored(self) -> None:
        llm = FakeLLM()
        result = make_tool_invocation(llm)(
            {"final_answer": "B"},
            {},
            {"instruction": "What is 2+2?", "available_tools": [_BASH_SCHEMA]},
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")
        self.assertEqual(llm.prompts, [])

    def test_names_without_arguments_is_unscored(self) -> None:
        llm = FakeLLM()
        result = make_tool_invocation(llm)(
            {"tool_calls": ["bash"], "final_answer": "done"},
            {},
            {"instruction": "list files", "available_tools": [_BASH_SCHEMA]},
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")
        self.assertEqual(llm.prompts, [])

    def test_calls_without_parameter_schema_is_unscored(self) -> None:
        llm = FakeLLM()
        result = make_tool_invocation(llm)(
            {"tool_calls_full": [{"name": "bash", "arguments": {"command": "ls"}}]},
            {},
            {"instruction": "list files", "available_tools": [{"name": "bash"}]},
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unscored")
        self.assertEqual(llm.prompts, [])

    def test_calls_with_schema_are_judged(self) -> None:
        llm = FakeLLM()
        result = make_tool_invocation(llm)(
            {"tool_calls_full": [{"name": "bash", "arguments": {"command": "ls"}}]},
            {},
            {"instruction": "list files", "available_tools": [_BASH_SCHEMA]},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["label"], "correct")
        self.assertGreaterEqual(len(llm.prompts), 1)
        self.assertIn("command", llm.prompts[-1])

    def test_schema_from_llm_spans(self) -> None:
        llm = FakeLLM()
        example = type("E", (), {"id": "ex1"})()
        spans = {
            "ex1": [
                {
                    "span_kind": "LLM",
                    "attributes": {
                        "llm": {
                            "tools": [
                                {
                                    "name": "bash",
                                    "json_schema": {
                                        "name": "bash",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {"command": {"type": "string"}},
                                            "required": ["command"],
                                        },
                                    },
                                }
                            ]
                        }
                    },
                }
            ]
        }
        result = make_tool_invocation(llm, spans)(
            {"tool_calls_full": [{"name": "bash", "arguments": {"command": "ls"}}]},
            {},
            {"instruction": "list files"},
            example=example,
        )
        self.assertEqual(result["label"], "correct")
        self.assertGreaterEqual(len(llm.prompts), 1)


if __name__ == "__main__":
    unittest.main()
