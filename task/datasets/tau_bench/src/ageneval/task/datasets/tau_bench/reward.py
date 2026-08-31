"""Sierra τ-bench official ``calculate_reward`` (pass^1).

Policy copied from ``tau_bench.envs.base.Env.calculate_reward``
(sierra-research/tau-bench, MIT):

1. Hash the agent's final database (``to_hashable`` + SHA-256).
2. Replay gold **tool** actions on a fresh copy of the same domain DB.
3. ``r_actions`` is 1 iff the two hashes match.
4. If the task lists ``outputs``, each string must appear in an agent
   ``respond`` / final answer (case-insensitive, commas stripped).
   Missing any output zeroes the reward.

Reward is 1.0 only when both checks pass. This is the official pass^1
outcome, not A2E ``tool_recall``.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

RESPOND_ACTION_NAME = "respond"

ToHashable = Any
Hashable = Any


def to_hashable(item: ToHashable) -> Hashable:
    """Official Sierra ``to_hashable`` (key-sorted, list→tuple)."""
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    return item


def consistent_hash(value: Hashable) -> str:
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


def apply_gold_actions(
    domain: str,
    actions: Sequence[Any],
) -> dict[str, Any]:
    """Replay gold tool calls on a fresh official Sierra DB (no user-sim)."""
    from ageneval.task.datasets.tau_bench.runtime import (
        execute_tool,
        load_domain_data,
    )

    resolved = "airline" if domain == "airline" else "retail"
    state: dict[str, Any] = {}
    db = deepcopy(load_domain_data(resolved))
    state["__tau_db__"] = db
    state["__tau_domain__"] = resolved
    for action in actions or ():
        name = _action_name(action)
        if not name or _is_respond(name):
            continue
        execute_tool(name, _action_args(action), state, domain=resolved)
    return db


def _agent_respond_text(output: Mapping[str, Any] | None) -> str:
    """NL the official checker would see as ``respond`` content."""
    out = output or {}
    chunks = [str(out.get("final_answer") or "")]
    for span in out.get("tool_spans") or ():
        if not isinstance(span, Mapping):
            continue
        if not _is_respond(str(span.get("name") or "")):
            continue
        args = span.get("arguments") or {}
        if isinstance(args, Mapping):
            chunks.append(str(args.get("content") or args.get("message") or args.get("text") or ""))
    return " ".join(chunks)


def _outputs_ok(gold_outputs: Sequence[Any], spoken: str) -> bool:
    if not gold_outputs:
        return True
    hay = spoken.lower().replace(",", "")
    for item in gold_outputs:
        needle = str(item or "").lower().replace(",", "")
        if not needle:
            continue
        if needle not in hay:
            return False
    return True


def official_tau_reward(
    *,
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> float:
    """Return official pass^1 (1.0 or 0.0)."""
    out = output or {}
    exp = expected or {}
    meta_in = input or {}
    resolved = (
        domain
        or out.get("tau_domain")
        or (meta_in.get("initial_state") or {}).get("__tau_domain__")
        or "retail"
    )
    if resolved not in ("retail", "airline"):
        resolved = "retail"

    gold_actions = list(exp.get("expected_actions") or ())
    gold_outputs = list(exp.get("expected_outputs") or ())

    gold_db = apply_gold_actions(resolved, gold_actions)
    gt_hash = data_hash(gold_db)

    agent_hash = out.get("tau_data_hash")
    if not agent_hash:
        from ageneval.task.datasets.tau_bench.runtime import load_domain_data

        agent_hash = data_hash(load_domain_data(resolved))

    r_actions = str(agent_hash) == gt_hash
    r_outputs = _outputs_ok(gold_outputs, _agent_respond_text(out))
    return 1.0 if (r_actions and r_outputs) else 0.0


def official_tau_reward_detail(
    *,
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    input: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Same official policy, with the two Sierra checks split out."""
    reward = official_tau_reward(
        output=output, expected=expected, input=input, domain=domain
    )
    out = output or {}
    exp = expected or {}
    resolved = domain or out.get("tau_domain") or "retail"
    if resolved not in ("retail", "airline"):
        resolved = "retail"
    gold_db = apply_gold_actions(resolved, list(exp.get("expected_actions") or ()))
    agent_hash = out.get("tau_data_hash")
    if not agent_hash:
        from ageneval.task.datasets.tau_bench.runtime import load_domain_data

        agent_hash = data_hash(load_domain_data(resolved))
    return {
        "reward": reward,
        "r_actions": str(agent_hash) == data_hash(gold_db),
        "r_outputs": _outputs_ok(
            list(exp.get("expected_outputs") or ()), _agent_respond_text(out)
        ),
        "gold_hash": data_hash(gold_db),
        "agent_hash": agent_hash,
    }
