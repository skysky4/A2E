"""Official Sierra calculate_reward policy (pass^1)."""

from __future__ import annotations

from ageneval.task.datasets.tau_bench.reward import (
    apply_gold_actions,
    data_hash,
    official_tau_reward,
)


def test_execute_tool_caches_identical_calls():
    from ageneval.task.datasets.tau_bench.runtime import execute_tool

    state: dict = {}
    first = execute_tool(
        "find_user_id_by_name_zip",
        {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
        state,
        domain="retail",
    )
    second = execute_tool(
        "find_user_id_by_name_zip",
        {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
        state,
        domain="retail",
    )
    assert first == second
    assert isinstance(first, str) and "rossi" in str(first).lower()


def test_data_hash_is_key_order_invariant():
    assert data_hash({"b": 1, "a": [2, 3]}) == data_hash({"a": [2, 3], "b": 1})


def test_gold_replay_matches_same_writes():
    actions = [
        {
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
        },
        {"name": "get_order_details", "arguments": {"order_id": "#W2378156"}},
        {
            "name": "exchange_delivered_order_items",
            "arguments": {
                "order_id": "#W2378156",
                "item_ids": ["1151293680", "4983901480"],
                "new_item_ids": ["7706410293", "7747408585"],
                "payment_method_id": "credit_card_9513926",
            },
        },
    ]
    db_a = apply_gold_actions("retail", actions)
    db_b = apply_gold_actions("retail", actions)
    assert data_hash(db_a) == data_hash(db_b)
    assert official_tau_reward(
        output={"tau_data_hash": data_hash(db_a), "tau_domain": "retail", "final_answer": "done"},
        expected={"expected_actions": actions, "expected_outputs": []},
    ) == 1.0


def test_wrong_db_is_zero():
    actions = [
        {
            "name": "cancel_pending_order",
            "arguments": {"order_id": "#W2378156", "reason": "no longer needed"},
        }
    ]
    assert official_tau_reward(
        output={"tau_data_hash": data_hash({"orders": {}}), "tau_domain": "retail"},
        expected={"expected_actions": actions, "expected_outputs": []},
    ) == 0.0


def test_missing_required_output_is_zero():
    actions = [
        {
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
        }
    ]
    db = apply_gold_actions("retail", actions)
    assert official_tau_reward(
        output={
            "tau_data_hash": data_hash(db),
            "tau_domain": "retail",
            "final_answer": "I will look that up",
        },
        expected={"expected_actions": actions, "expected_outputs": ["10"]},
    ) == 0.0
    assert official_tau_reward(
        output={
            "tau_data_hash": data_hash(db),
            "tau_domain": "retail",
            "final_answer": "There are 10 options",
        },
        expected={"expected_actions": actions, "expected_outputs": ["10"]},
    ) == 1.0
