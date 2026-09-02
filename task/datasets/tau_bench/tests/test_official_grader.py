from __future__ import annotations

from ageneval.task.datasets.tau_bench.grader import GRADER_METADATA, tau_grader
from ageneval.task.datasets.tau_bench.reward import (
    apply_gold_actions,
    data_hash,
    official_tau_reward,
)


def test_data_hash_is_key_order_invariant() -> None:
    assert data_hash({"b": 1, "a": [2, 3]}) == data_hash(
        {"a": [2, 3], "b": 1}
    )


def test_gold_replay_matches_same_write() -> None:
    actions = [
        {
            "name": "cancel_pending_order",
            "arguments": {
                "order_id": "#W2378156",
                "reason": "no longer needed",
            },
        }
    ]
    gold_db = apply_gold_actions("retail", actions)
    output = {
        "tau_data_hash": data_hash(gold_db),
        "tau_domain": "retail",
        "final_answer": "The order was cancelled.",
    }
    expected = {"expected_actions": actions, "expected_outputs": []}
    assert official_tau_reward(output=output, expected=expected) == 1.0
    assert tau_grader(output, expected) == 1.0


def test_required_output_must_be_communicated() -> None:
    gold_db = apply_gold_actions("retail", [])
    expected = {"expected_actions": [], "expected_outputs": ["$10"]}
    base = {"tau_data_hash": data_hash(gold_db), "tau_domain": "retail"}
    assert official_tau_reward(
        output={**base, "final_answer": "Done."},
        expected=expected,
    ) == 0.0
    assert official_tau_reward(
        output={**base, "final_answer": "Your credit is $10."},
        expected=expected,
    ) == 1.0
    assert official_tau_reward(
        output={**base, "tau_spoken": "Your credit is $10.", "final_answer": "Done."},
        expected=expected,
    ) == 1.0


def test_runtime_reuses_identical_successful_call(monkeypatch) -> None:
    from ageneval.task.datasets.tau_bench import runtime

    class _FakeTool:
        calls = 0

        @classmethod
        def invoke(cls, data, **kwargs):
            del data, kwargs
            cls.calls += 1
            return {"call": cls.calls}

    monkeypatch.setattr(runtime, "_tool_map", lambda domain: {"fake": _FakeTool})
    state = {"__tau_db__": {}}
    assert runtime.execute_tool("fake", {"value": 1}, state) == {"call": 1}
    assert runtime.execute_tool("fake", {"value": 1}, state) == {"call": 1}
    assert _FakeTool.calls == 1


def test_tau_bench_metadata_is_official() -> None:
    assert GRADER_METADATA["id"] == "tau_grader"
    assert GRADER_METADATA["official"] is True
