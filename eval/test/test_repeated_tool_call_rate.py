"""Tests for repeated_tool_call_rate (CODE locates TOOL windows, one LLM judge)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.eval_common import _tool_calls_from_spans
from process_values.tool_eval import (
    _collapse_monitor_clones,
    _locate_loop_windows,
    _loop_call_record,
    make_repeated_tool_call_rate,
)


class FakeLLM:
    model = "test-judge"

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.text


def _bash(command: str, result: str) -> dict:
    return {"name": "bash", "arguments": {"command": command}, "result": result}


def test_no_calls_is_unscored_without_llm():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=should not run")
    result = make_repeated_tool_call_rate(llm)(output={}, expected={}, input={})
    assert result["label"] == "unscored"
    assert llm.prompts == []


def test_names_only_is_unscored_without_llm():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=should not run")
    result = make_repeated_tool_call_rate(llm)(
        output={"tool_calls": ["bash", "bash", "bash"]},
        expected={},
        input={},
    )
    assert result["label"] == "unscored"
    assert llm.prompts == []


def test_no_tool_window_is_clean_without_llm():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=should not run")
    result = make_repeated_tool_call_rate(llm)(
        output={
            "tool_calls_full": [
                _bash("ls /app", "a"),
                _bash("ls /data", "b"),
                _bash("pwd", "/app"),
            ]
        },
        expected={},
        input={"instruction": "inspect files"},
    )
    assert result["label"] == "clean"
    assert result["score"] == 0.0
    assert llm.prompts == []


def test_llm_sees_only_located_tool_window():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=No new information; stuck repeating sleep.")
    filler = [_bash(f"echo {idx}", str(idx)) for idx in range(8)]
    loop = [_bash("sleep 290; cat /app/final.log", f"progress {idx}") for idx in range(3)]
    result = make_repeated_tool_call_rate(llm)(
        output={"tool_calls_full": filler + loop, "status": "ok"},
        expected={},
        input={"instruction": "train a fasttext model under 150MB"},
    )
    assert result["label"] == "looping"
    assert result["score"] == 1.0
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "sleep 290" in prompt
    assert "echo 0" not in prompt
    assert "echo 7" not in prompt
    assert "Candidate TOOL windows" in prompt


def test_llm_can_reject_polling_as_not_a_loop():
    llm = FakeLLM("LABEL=clean; SCORE=1; EXPLANATION=Outputs change from running to done.")
    result = make_repeated_tool_call_rate(llm)(
        output={
            "tool_calls_full": [
                _bash("ps", "running"),
                _bash("ps", "running 2"),
                _bash("ps", "done"),
            ]
        },
        expected={},
        input={"instruction": "wait until training finishes"},
    )
    assert result["label"] == "clean"
    assert result["score"] == 0.0
    assert llm.prompts


def test_near_duplicate_long_commands_are_located():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=trivial tail -2/-3 variation, no new info.")
    prefix = "tail -{n} /tmp/apt.log; which coqc ocaml menhir gcc make; pgrep -af apt-get"
    calls = [_bash("ls /app", "ok")] + [
        _bash(prefix.format(n=2 if idx % 2 == 0 else 3), "duplicate tool call")
        for idx in range(4)
    ]
    result = make_repeated_tool_call_rate(llm)(
        output={"tool_calls_full": calls},
        expected={},
        input={"instruction": "install coq"},
    )
    assert result["label"] == "looping"
    assert result["score"] == 1.0
    assert llm.prompts
    assert "apt.log" in llm.prompts[0]
    assert "ls /app" not in llm.prompts[0]


def test_alternating_cycle_is_localized():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=A-B-A-B with no progress.")
    result = make_repeated_tool_call_rate(llm)(
        output={
            "tool_calls_full": [
                _bash("ls", "1"),
                _bash("pwd", "2"),
                _bash("ls", "3"),
                _bash("pwd", "4"),
            ]
        },
        expected={},
        input={"instruction": "explore"},
    )
    assert result["label"] == "looping"
    assert result["score"] == 1.0
    assert llm.prompts


def test_two_tools_in_one_cycle_score_two():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=A-B-A-B across two tools.")
    result = make_repeated_tool_call_rate(llm)(
        output={
            "tool_calls_full": [
                _bash("ls", "1"),
                {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/app/x"}, "result": "2"},
                _bash("ls", "3"),
                {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/app/x"}, "result": "4"},
            ]
        },
        expected={},
        input={"instruction": "explore"},
    )
    assert result["label"] == "looping"
    assert result["score"] == 2.0
    assert llm.prompts


def _span(name: str, command: str, *, schema: dict, wrapper: str = "command", order: int = 0) -> dict:
    if wrapper == "kwargs":
        payload = {"kwargs": {"command": command}}
    elif wrapper == "smol":
        payload = {"args": [], "sanitize_inputs_outputs": True, "kwargs": {"command": command}}
    else:
        payload = {"command": command}
    return {
        "span_kind": "TOOL",
        "name": name,
        "start_time": f"{order:04d}",
        "attributes": {
            "tool": {"name": "bash", "parameters": schema},
            "input": {"value": json.dumps(payload), "mime_type": "application/json"},
            "output": {"value": f"out:{command}"},
        },
    }


_BASH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"title": "Command", "type": "string"}},
    "required": ["command"],
}


def test_span_args_use_invocation_not_schema():
    spans = [
        _span("FunctionTool.acall", "mkdir /app/polyglot", schema=_BASH_SCHEMA, wrapper="kwargs", order=0),
        _span("FunctionTool.acall", "python3 main.py.c", schema=_BASH_SCHEMA, wrapper="kwargs", order=1),
        _span("bash.run", "pwd && ls -la", schema=_BASH_SCHEMA, wrapper="command", order=2),
        _span("Tool_bash", "xxd -l 128 main.db", schema={"command": {"type": "string"}}, wrapper="smol", order=3),
    ]
    calls = _tool_calls_from_spans(spans)
    commands = [call["arguments"]["command"] for call in calls]
    assert commands == [
        "mkdir /app/polyglot",
        "python3 main.py.c",
        "pwd && ls -la",
        "xxd -l 128 main.db",
    ]
    records = _collapse_monitor_clones([rec for rec in (_loop_call_record(call) for call in calls) if rec])
    assert _locate_loop_windows(records) == []


def test_distinct_schema_wrapped_commands_skip_llm():
    llm = FakeLLM("LABEL=looping; SCORE=0; EXPLANATION=should not run")
    spans = [_span("bash.run", f"echo {idx}", schema=_BASH_SCHEMA, order=idx) for idx in range(4)]
    result = make_repeated_tool_call_rate(llm)(
        output={"tool_calls_full": _tool_calls_from_spans(spans)},
        expected={},
        input={"instruction": "explore"},
    )
    assert result["label"] == "clean"
    assert result["score"] == 0.0
    assert llm.prompts == []
