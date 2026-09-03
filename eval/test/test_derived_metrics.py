"""Tests for metrics derived from span trajectories."""

from __future__ import annotations

import unittest

from process_values.tool_eval import _parameter_schemas_from_tool_spans, _tool_schema_menu_str
from result_values.efficiency_eval import make_wall_time


class DerivedElapsedTimeTests(unittest.TestCase):
    def test_wall_time_from_span_timestamps(self) -> None:
        spans = {
            "ex-1": [
                {
                    "span_kind": "LLM",
                    "start_time": "2026-08-17 15:31:08.127976",
                    "end_time": "2026-08-17 15:31:10.127976",
                    "attributes": {},
                },
                {
                    "span_kind": "TOOL",
                    "start_time": "2026-08-17 15:31:10.127976",
                    "end_time": "2026-08-17 15:36:06.084875",
                    "attributes": {},
                },
            ]
        }
        metric = make_wall_time(spans)
        result = metric({"task_output": {"status": "ok"}}, {}, {}, example={"id": "ex-1"})
        self.assertEqual(result["score"], 297.956899)
        self.assertIn("derived from", result["explanation"])

    def test_prefers_task_output_wall_time_fields(self) -> None:
        metric = make_wall_time({})
        result = metric({"task_output": {"elapsed_time": 12.5}}, {}, {}, example={"id": "ex-1"})
        self.assertEqual(result["score"], 12.5)
        self.assertIn("task output", result["explanation"])


class DerivedToolSchemaTests(unittest.TestCase):
    def test_infers_schema_from_tool_span_arguments(self) -> None:
        spans = [
            {
                "span_kind": "TOOL",
                "name": "bash",
                "attributes": {
                    "tool": {"name": "bash"},
                    "input": {"value": '{"command": "pwd"}', "mime_type": "application/json"},
                },
            }
        ]
        schemas = _parameter_schemas_from_tool_spans(spans)
        self.assertEqual(len(schemas), 1)
        menu = _tool_schema_menu_str({}, spans)
        self.assertIn("bash", menu)
        self.assertIn("command", menu)

    def test_uses_tool_span_description_when_present(self) -> None:
        spans = [
            {
                "span_kind": "TOOL",
                "name": "bash",
                "attributes": {
                    "tool": {
                        "name": "bash",
                        "description": "Run a bash command",
                        "parameters": {"command": "ls"},
                    }
                },
            }
        ]
        schemas = _parameter_schemas_from_tool_spans(spans)
        self.assertTrue(schemas)


if __name__ == "__main__":
    unittest.main()
