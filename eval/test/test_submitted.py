"""Tests for submitted (SWE-bench submitted vs empty_patch)."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from process_values.delivery_eval import make_submitted


def _metric():
    return make_submitted({})


def _bash(command: str) -> dict:
    return {"name": "bash", "arguments": {"command": command}, "result": {"exit_code": 0}}


def test_no_calls_and_no_answer_is_unscored():
    result = _metric()(output={}, expected={}, input={"instruction": "Write /app/out.py"})
    assert result["label"] == "unscored"
    assert result["score"] is None


def test_final_answer_without_tools_is_submitted():
    result = _metric()(
        output={"final_answer": "42"},
        expected={},
        input={"instruction": "What is 6*7?"},
    )
    assert result["label"] == "submitted"
    assert result["score"] == 1.0


def test_git_missing_string_is_not_a_patch():
    result = _metric()(
        output={
            "tool_calls_full": [_bash("ls /app")],
            "model_patch": (
                'OCI runtime exec failed: exec failed: unable to start container process: '
                'exec: "git": executable file not found in $PATH'
            ),
        },
        expected={},
        input={"instruction": "Fix the bug."},
    )
    assert result["label"] == "empty"
    assert result["score"] == 0.0

    result = _metric()(
        output={"tool_calls_full": [_bash("ls /app")], "model_patch": ""},
        expected={},
        input={"instruction": "Fix the bug."},
    )
    assert result["label"] == "empty"
    assert result["score"] == 0.0


def test_nonempty_patch_is_submitted():
    result = _metric()(
        output={"model_patch": "diff --git a/foo.py b/foo.py\n+print(1)\n"},
        expected={},
        input={"instruction": "Fix the bug."},
    )
    assert result["label"] == "submitted"
    assert result["score"] == 1.0


def test_stop_tool_is_submitted():
    result = _metric()(
        output={"tool_calls_full": [{"name": "stop", "arguments": {"answer": "done"}}]},
        expected={},
        input={"instruction": "Book a flight."},
    )
    assert result["label"] == "submitted"
    assert result["score"] == 1.0


def test_create_file_is_submitted():
    result = _metric()(
        output={
            "tool_calls_full": [
                {
                    "name": "str_replace_editor",
                    "arguments": {
                        "command": "create",
                        "path": "/app/run.py",
                        "file_text": "async def run_tasks(...): ...",
                    },
                }
            ]
        },
        expected={},
        input={"instruction": "Put the function in `/app/run.py`."},
    )
    assert result["label"] == "submitted"
    assert result["score"] == 1.0


def test_explore_only_is_empty():
    result = _metric()(
        output={
            "tool_calls_full": [
                _bash("cat /app/model_ref.xml /app/eval.py"),
                _bash("sed -n 40,120p /app/eval.py"),
            ]
        },
        expected={},
        input={"instruction": "Tuned mjcf should be saved as /app/model.xml."},
    )
    assert result["label"] == "empty"
    assert result["score"] == 0.0


def test_relative_copy_is_submitted():
    result = _metric()(
        output={"tool_calls_full": [_bash("cp model_ref.xml model.xml && python eval.py")]},
        expected={},
        input={"instruction": "Save as /app/model.xml."},
    )
    assert result["label"] == "submitted"
    assert result["score"] == 1.0
