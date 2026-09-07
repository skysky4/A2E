"""Hidden Sierra script must never be the agent's first user message."""

from __future__ import annotations

import pytest

from ageneval.task.datasets.tau_bench.user_sim import (
    LLMUserSimulationEnv,
    STOP_TOKEN,
    hidden_script_from_example,
    looks_like_hidden_script,
    load_user,
    official_tau_example_input,
)

_SCRIPT = (
    "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish "
    "to exchange the mechanical keyboard."
)


def test_hidden_script_detector():
    assert looks_like_hidden_script(_SCRIPT)
    assert looks_like_hidden_script(
        "You name is Sophia Martin. You live in 77034 and want to exchange #W1603792."
    )
    assert not looks_like_hidden_script("Hi, I need help with an order I received.")
    assert not looks_like_hidden_script("My zip code is 19122.")
    assert not looks_like_hidden_script(
        "Hi, my name is Yara Silva and my ZIP code is 77159."
    )


def test_uploaded_example_keeps_script_out_of_instruction():
    payload = official_tau_example_input(_SCRIPT, {"__tau_domain__": "retail"})
    assert payload["hidden_instruction"] == _SCRIPT
    assert _SCRIPT not in payload["instruction"]
    assert "You are Yusuf" not in payload["instruction"]
    assert hidden_script_from_example(payload) == _SCRIPT
    assert hidden_script_from_example({"instruction": _SCRIPT}) == _SCRIPT


def test_naive_user_class_is_gone():
    import ageneval.task.datasets.tau_bench.user_sim as user_sim

    assert not hasattr(user_sim, "NaiveUserSimulationEnv")


def test_run_experiment_upload_hides_tau_script():
    import sys
    from types import SimpleNamespace

    examples = "/root/A2E/task/examples"
    if examples not in sys.path:
        sys.path.insert(0, examples)
    from run_experiment import _build_examples

    task = SimpleNamespace(
        task_id="retail-0111",
        instruction=_SCRIPT,
        initial_state={"__tau_domain__": "retail"},
        expected_outputs=(),
        expected_actions=(),
        metadata={},
        sandbox=None,
    )
    rows = _build_examples([task], dataset_key="tau-bench")
    assert rows[0]["input"]["hidden_instruction"] == _SCRIPT
    assert _SCRIPT not in rows[0]["input"]["instruction"]
    assert rows[0]["metadata"]["tau_hidden_instruction"] is True
    other = _build_examples([task], dataset_key="deepsearchqa")
    assert other[0]["input"]["instruction"] == _SCRIPT


def test_load_user_is_official_llm_only():
    user = load_user("naive")
    assert isinstance(user, LLMUserSimulationEnv)
    user2 = load_user("llm")
    assert isinstance(user2, LLMUserSimulationEnv)


def test_llm_retry_then_error_on_repeated_leak():
    user = LLMUserSimulationEnv(model="unused")
    calls = {"n": 0}

    def _gen():
        calls["n"] += 1
        return _SCRIPT

    user._generate = _gen  # type: ignore[method-assign]
    user.messages = [
        {"role": "system", "content": user.build_system_prompt(_SCRIPT)},
        {"role": "user", "content": "Hi! How can I help you today?"},
    ]
    with pytest.raises(RuntimeError, match="leaked"):
        user._official_utterance(_SCRIPT)
    assert calls["n"] == 1


def test_llm_retry_accepts_natural_line():
    user = LLMUserSimulationEnv(model="unused")
    replies = ["Hi, I need help with an order I received."]

    def _gen():
        return replies.pop(0) if replies else "Hi"

    user._generate = _gen  # type: ignore[method-assign]
    user.messages = [
        {"role": "system", "content": user.build_system_prompt(_SCRIPT)},
        {"role": "user", "content": "Hi! How can I help you today?"},
    ]
    out = user._official_utterance(_SCRIPT)
    assert out == "Hi, I need help with an order I received."
    assert not looks_like_hidden_script(out)


def test_empty_generate_retries_and_is_not_stop():
    user = LLMUserSimulationEnv(model="unused")
    replies = ["", "Hi, I need help with an order I received."]

    def _gen():
        return replies.pop(0) if replies else ""

    user._generate = _gen  # type: ignore[method-assign]
    out = user.reset(_SCRIPT)
    assert out == "Hi, I need help with an order I received."
    assert STOP_TOKEN not in out


def test_empty_after_retry_stays_empty():
    user = LLMUserSimulationEnv(model="unused")
    user._generate = lambda: ""  # type: ignore[method-assign]
    user.messages = [
        {"role": "system", "content": user.build_system_prompt(_SCRIPT)},
    ]
    out = user._official_utterance("")
    assert out == ""
    assert out != STOP_TOKEN
