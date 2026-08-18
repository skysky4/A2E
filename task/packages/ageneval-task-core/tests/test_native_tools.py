from __future__ import annotations

from ageneval.task.core.native_tools import pydantic_args_model


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
