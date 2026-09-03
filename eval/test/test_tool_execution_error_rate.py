"""Tests for tool_execution_error_rate (BFCL executable evaluation)."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from process_values.tool_eval import make_tool_execution_error_rate


def _metric():
    return make_tool_execution_error_rate({})


def test_no_calls_is_unscored():
    result = _metric()(output={}, expected={}, input={})
    assert result["label"] == "unscored"
    assert result["score"] is None


def test_all_zero_exit_is_clean():
    result = _metric()(
        output={
            "tool_calls_full": [
                {"name": "bash", "arguments": {"command": "ls"}, "result": {"stdout": "ok", "stderr": "", "exit_code": 0}},
                {"name": "bash", "arguments": {"command": "pwd"}, "result": {"stdout": "/app", "stderr": "", "exit_code": 0}},
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "clean"
    assert result["score"] == 0.0


def test_stdout_not_found_with_exit_zero_is_not_an_error():
    result = _metric()(
        output={
            "tool_calls_full": [
                {
                    "name": "bash",
                    "arguments": {"command": "ls /missing"},
                    "result": {"stdout": "ls: cannot access '/missing': No such file", "stderr": "", "exit_code": 0},
                }
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "clean"
    assert result["score"] == 0.0


def test_nonzero_exit_is_runtime():
    result = _metric()(
        output={
            "tool_calls_full": [
                {"name": "bash", "result": {"stdout": "", "stderr": "ls: No such file", "exit_code": 2}},
                {"name": "bash", "result": {"stdout": "ok", "stderr": "", "exit_code": 0}},
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "runtime"
    assert result["score"] == 1.0


def test_container_not_started_is_environment_dead():
    result = _metric()(
        output={"tool_calls_full": [{"name": "bash", "result": {"error": "container not started"}}]},
        expected={},
        input={},
    )
    assert result["label"] == "environment_dead"
    assert result["score"] == 1.0


def test_duplicate_tool_call_is_harness_reject():
    result = _metric()(
        output={
            "tool_calls_full": [
                {
                    "name": "bash",
                    "result": {"error": "duplicate tool call; reuse the previous result instead of calling again"},
                }
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "harness_reject"
    assert result["score"] == 1.0


def test_timeout_from_stderr():
    result = _metric()(
        output={
            "tool_calls_full": [
                {"name": "bash", "result": {"stdout": "", "stderr": "timed out after 300s", "exit_code": 137}}
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "timeout"
    assert result["score"] == 1.0


def test_python_traceback_is_exception():
    result = _metric()(
        output={
            "tool_calls_full": [
                {
                    "name": "bash",
                    "result": {
                        "stdout": "",
                        "stderr": 'Traceback (most recent call last):\n File "<stdin>", line 1, in <module>\nValueError: x',
                        "exit_code": 1,
                    },
                }
            ]
        },
        expected={},
        input={},
    )
    assert result["label"] == "exception"
    assert result["score"] == 1.0


def test_score_is_failed_call_count_not_rate():
    calls = [
        {"name": "bash", "result": {"stdout": "ok", "stderr": "", "exit_code": 0}},
        {"name": "bash", "result": {"error": "container not started"}},
        {"name": "bash", "result": {"stdout": "", "stderr": "timed out after 300s", "exit_code": 137}},
        {"name": "bash", "result": {"error": "container not started"}},
        {"name": "bash", "result": {"stdout": "ok", "stderr": "", "exit_code": 0}},
    ]
    result = _metric()(output={"tool_calls_full": calls}, expected={}, input={})
    assert result["score"] == 3.0
    assert result["label"] == "environment_dead"
