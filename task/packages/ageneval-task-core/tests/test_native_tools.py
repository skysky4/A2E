from __future__ import annotations

import inspect
import typing

from ageneval.task.core.native_tools import (
    attach_json_schema_signature,
    pydantic_args_model,
)


def test_json_schema_signature_marks_optional_parameters_nullable() -> None:
    def tool(**kwargs: object) -> str:
        return str(kwargs)

    wrapped = attach_json_schema_signature(
        tool,
        name="editor",
        description="Edit a file.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "file_text": {"type": "string"},
            },
            "required": ["command"],
        },
    )

    parameters = inspect.signature(wrapped).parameters
    assert parameters["command"].annotation is str
    assert parameters["command"].default is inspect.Parameter.empty
    assert parameters["file_text"].annotation == str | None
    assert typing.get_origin(parameters["file_text"].annotation) is typing.Union
    assert parameters["file_text"].default is None


def test_pydantic_args_model_preserves_array_item_types() -> None:
    model = pydantic_args_model(
        "ArrayTool",
        {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["numbers", "names", "records"],
        },
    )

    properties = model.model_json_schema()["properties"]
    assert properties["numbers"]["items"]["type"] == "integer"
    assert properties["names"]["items"]["type"] == "string"
    assert properties["records"]["items"] == {
        "additionalProperties": True,
        "type": "object",
    }
