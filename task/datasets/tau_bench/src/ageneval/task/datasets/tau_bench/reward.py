"""Sierra τ-bench ``calculate_reward`` implementation (pass^1).

The score matches the final database against a replay of the reference tool
actions and, when present, checks that every required natural-language output
was communicated by the agent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

RESPOND_ACTION_NAME = "respond"


def to_hashable(item: Any) -> Any:
    """Convert nested containers to the key-sorted form used by τ-bench."""
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    return item


def consistent_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def data_hash(data: Mapping[str, Any] | None) -> str:
    return consistent_hash(to_hashable(dict(data or {})))


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        return str(action.get("name") or "")
    return str(getattr(action, "name", "") or "")


def _action_args(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        raw = action.get("arguments")
        if raw is None:
            raw = action.get("kwargs")
    else:
        raw = getattr(action, "kwargs", None)
        if raw is None:
            raw = getattr(action, "arguments", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _is_respond(name: str) -> bool:
    return name.strip().lower() in {RESPOND_ACTION_NAME, "communicate"}


def apply_gold_actions(domain: str, actions: Sequence[Any]) -> dict[str, Any]:
    """Replay reference tool calls on a fresh copy of the domain database."""
    from ageneval.task.datasets.tau_bench.runtime import execute_tool, load_domain_data

    resolved = "airline" if domain == "airline" else "retail"
    db = deepcopy(load_domain_data(resolved))
    state: dict[str, Any] = {
        "__tau_db__": db,
        "__tau_domain__": resolved,
    }
    for action in actions or ():
        name = _action_name(action)
        if not name or _is_respond(name):
            continue
        execute_tool(name, _action_args(action), state, domain=resolved)
    return db


def _agent_respond_text(output: Mapping[str, Any] | None) -> str:
    out = output or {}
    chunks = [
        str(out.get("tau_spoken") or ""),
        str(out.get("final_answer") or ""),
    ]
    for span in out.get("tool_spans") or ():
        if not isinstance(span, Mapping):
            continue
        if not _is_respond(str(span.get("name") or "")):
            continue
        args = span.get("arguments") or {}
        if isinstance(args, Mapping):
            chunks.append(
                str(args.get("content") or args.get("message") or args.get("text") or "")
            )
    return " ".join(chunks)


def _outputs_ok(gold_outputs: Sequence[Any], spoken: str) -> bool:
    if not gold_outputs:
        return True
    haystack = spoken.lower().replace(",", "")
    for item in gold_outputs:
        needle = str(item or "").lower().replace(",", "")
        if needle and needle not in haystack:
            return False
    return True


def _resolve_domain(
    output: Mapping[str, Any],
    input: Mapping[str, Any],
    explicit: str | None,
) -> str:
    initial_state = input.get("initial_state")
    state_domain = (
        initial_state.get("__tau_domain__") if isinstance(initial_state, Mapping) else None
    )
    resolved = explicit or output.get("tau_domain") or state_domain or "retail"
    return "airline" if resolved == "airline" else "retail"


def official_tau_reward(
    *,
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Return the official binary pass^1 outcome."""
    out = output or {}
    exp = expected or {}
    resolved = _resolve_domain(out, input or {}, domain)
    gold_actions = list(exp.get("expected_actions") or ())
    gold_outputs = list(exp.get("expected_outputs") or ())

    gold_db = apply_gold_actions(resolved, gold_actions)
    agent_hash = out.get("tau_data_hash")
    if not agent_hash:
        from ageneval.task.datasets.tau_bench.runtime import load_domain_data

        agent_hash = data_hash(load_domain_data(resolved))

    actions_ok = str(agent_hash) == data_hash(gold_db)
    outputs_ok = _outputs_ok(gold_outputs, _agent_respond_text(out))
    return 1.0 if actions_ok and outputs_ok else 0.0


def official_tau_reward_detail(
    *,
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Return the official score with its database and response checks."""
    out = output or {}
    exp = expected or {}
    resolved = _resolve_domain(out, input or {}, domain)
    gold_db = apply_gold_actions(resolved, list(exp.get("expected_actions") or ()))
    gold_hash = data_hash(gold_db)
    agent_hash = out.get("tau_data_hash")
    if not agent_hash:
        from ageneval.task.datasets.tau_bench.runtime import load_domain_data

        agent_hash = data_hash(load_domain_data(resolved))
    actions_ok = str(agent_hash) == gold_hash
    outputs_ok = _outputs_ok(
        list(exp.get("expected_outputs") or ()),
        _agent_respond_text(out),
    )
    return {
        "reward": 1.0 if actions_ok and outputs_ok else 0.0,
        "r_actions": actions_ok,
        "r_outputs": outputs_ok,
        "gold_hash": gold_hash,
        "agent_hash": agent_hash,
    }


__all__ = [
    "apply_gold_actions",
    "consistent_hash",
    "data_hash",
    "official_tau_reward",
    "official_tau_reward_detail",
    "to_hashable",
]
