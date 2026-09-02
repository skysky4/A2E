from __future__ import annotations

from ageneval.task.agents.smolagents.agent import _make_smolagents_tool
from smolagents.models import get_tool_json_schema
from smolagents.tools import validate_tool_arguments


def _editor_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "View and edit files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "insert_line": {"type": "integer"},
                    "view_range": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["command", "path"],
            },
        },
    }


def test_smolagents_tool_preserves_optional_parameters() -> None:
    tool = _make_smolagents_tool(
        _editor_schema(),
        executor=lambda *_args: {"ok": True},
        initial_state={},
        recorder=[],
    )

    assert tool.inputs["command"].get("nullable") is not True
    assert tool.inputs["path"].get("nullable") is not True
    for name in ("file_text", "old_str", "new_str", "insert_line", "view_range"):
        assert tool.inputs[name]["nullable"] is True

    validate_tool_arguments(tool, {"command": "view", "path": "/app/example.txt"})
    schema = get_tool_json_schema(tool)
    assert schema["function"]["parameters"]["required"] == ["command", "path"]


def test_smolagents_tool_still_rejects_missing_required_parameter() -> None:
    tool = _make_smolagents_tool(
        _editor_schema(),
        executor=lambda *_args: {"ok": True},
        initial_state={},
        recorder=[],
    )

    try:
        validate_tool_arguments(tool, {"command": "view"})
    except ValueError as exc:
        assert str(exc) == "Argument path is required"
    else:
        raise AssertionError("missing required path was accepted")
