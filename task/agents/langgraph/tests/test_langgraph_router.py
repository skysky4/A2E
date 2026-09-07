"""LangGraph router must execute leaked official tools, not treat them as finals."""

from __future__ import annotations

from ageneval.task.agents.langgraph.nodes import executor_run, router_node
from ageneval.task.core import TaskInput


class _LLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _msgs):  # noqa: ANN001
        return type("M", (), {"content": self.content})()


class _Binding:
    name = "gdpval"

    def __init__(self) -> None:
        self.tool_schemas = [
            {"function": {"name": "code_exec"}},
            {"function": {"name": "find_user_id_by_name_zip"}},
        ]
        self.calls: list[tuple[str, dict]] = []

    def render_system_prompt(self) -> str:
        return "sys"

    def tool_executor(self, name, args, state):  # noqa: ANN001
        self.calls.append((name, dict(args)))
        return {"ok": True}


def test_router_dispatches_leaked_code_exec():
    binding = _Binding()
    task = TaskInput(task_id="t", instruction="Build the workbook.", initial_state={})
    out = router_node(
        state={"task": task, "tool_calls": []},
        llm=_LLM('to=code_exec  code:\n{"code":"print(1)"}'),
        binding=binding,
    )
    assert out["next_action"]["name"] == "code_exec"
    assert "print(1)" in str(out["next_action"]["arguments"])


def test_parse_json_keeps_nested_code_braces():
    from ageneval.task.agents.langgraph.nodes import _parse_json

    parsed = _parse_json(
        '{"action":"code_exec","arguments":{"code":"d={1:2}; print(d)"}}'
    )
    assert parsed["action"] == "code_exec"
    assert "print(d)" in parsed["arguments"]["code"]


def test_router_dispatches_leftover_action_json():
    binding = _Binding()
    task = TaskInput(task_id="t", instruction="Build the workbook.", initial_state={})
    out = router_node(
        state={"task": task, "tool_calls": []},
        llm=_LLM('{"action":"code_exec","arguments":{"code":"d={1:2}; print(d)"}}'),
        binding=binding,
    )
    assert out["next_action"]["name"] == "code_exec"
    assert "print(d)" in str(out["next_action"]["arguments"])


def test_executor_aliases_zip_code_and_name():
    binding = _Binding()
    task = TaskInput(task_id="t", instruction="x", initial_state={})
    out = executor_run(
        state={
            "task": task,
            "next_action": {
                "name": "find_user_id_by_name_zip",
                "arguments": {"name": "Yara Silva", "zip_code": "77159"},
            },
            "tool_calls": [],
            "turns": 0,
        },
        binding=binding,
    )
    assert binding.calls == [
        (
            "find_user_id_by_name_zip",
            {"first_name": "Yara", "last_name": "Silva", "zip": "77159"},
        )
    ]
    assert out["tool_calls"][0]["arguments"]["zip"] == "77159"
