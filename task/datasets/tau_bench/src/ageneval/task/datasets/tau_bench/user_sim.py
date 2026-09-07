"""Official Sierra τ-bench LLM user simulator (``tau_bench.envs.user``).

``task.instruction`` is the *hidden customer script*, never the agent's first
user message. Official ``Env.reset`` asks this simulator for the opening
utterance; agent text is a ``respond`` action and ``step()`` continues until
``###STOP###``.

This module implements Sierra's ``LLMUserSimulationEnv`` only. There is no
naive / script / deterministic fallback on the official path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Protocol

STOP_TOKEN = "###STOP###"

# Sierra ``LLMUserSimulationEnv.build_system_prompt`` (MIT, sierra-research/tau-bench).
_OFFICIAL_USER_PROMPT = """You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '{stop}' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

_NAME_RE = re.compile(
    r"You are ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+)",
)
_ZIP_RE = re.compile(r"\b(\d{5})\b")
_ORDER_RE = re.compile(r"(#W\d+)")

_REMIND_NO_LEAK = (
    "That reply copied the hidden instruction. "
    "Generate one natural customer line only. "
    "Do not repeat the instruction, and do not volunteer zip codes or order "
    "IDs unless the agent just asked for them."
)

# Official Sierra only ends on an explicit ``###STOP###``. An empty model
# reply is not a stop. Retry once, then return empty so the session can
# continue; never map empty → STOP.
_REMIND_NONEMPTY = (
    "Reply with one natural customer line only, or "
    f"{STOP_TOKEN} if the instruction goal is already satisfied."
)


class UserSimulationEnv(Protocol):
    def reset(self, instruction: str | None = None) -> str: ...

    def step(self, content: str) -> str: ...


def looks_like_hidden_script(text: str) -> bool:
    """True if a customer utterance dumped the hidden Sierra script."""
    t = (text or "").strip()
    if not t:
        return False
    head = t[:80]
    has_you_are = (
        t.startswith("You are ")
        or "You are " in head
        or t.lower().startswith("you name is ")
        or "you name is " in head.lower()
    )
    has_zip = bool(_ZIP_RE.search(t))
    has_order = bool(_ORDER_RE.search(t))
    return has_you_are and (has_zip or has_order)


def official_tau_example_input(
    hidden_script: str, initial_state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A2E example fields. Hidden Sierra script is not the agent opening."""
    return {
        "instruction": (
            "Official Sierra τ conversation. "
            "The agent sees only LLM user-simulator utterances, "
            "never the hidden customer script."
        ),
        "hidden_instruction": hidden_script,
        "initial_state": dict(initial_state or {}),
    }


def hidden_script_from_example(
    payload: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Recover the hidden Sierra script from an uploaded example."""
    data = dict(payload or {})
    meta = dict(metadata or {})
    return str(
        data.get("hidden_instruction")
        or meta.get("hidden_instruction")
        or data.get("instruction")
        or ""
    )


class LLMUserSimulationEnv:
    """Sierra ``LLMUserSimulationEnv`` over the same OpenAI-compatible API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = (
            model
            or os.environ.get("A2E_TAU_USER_MODEL")
            or os.environ.get("A2E_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.messages: list[dict[str, Any]] = []

    def build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n") if instruction else ""
        )
        return _OFFICIAL_USER_PROMPT.format(
            instruction_display=instruction_display, stop=STOP_TOKEN
        )

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._official_utterance(self._generate())

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._official_utterance(self._generate())

    def _official_utterance(self, text: str) -> str:
        """Official policy: one customer line, never the hidden script."""
        text = (text or "").strip()
        if not text:
            self.messages.append({"role": "user", "content": _REMIND_NONEMPTY})
            text = (self._generate() or "").strip()
        if not looks_like_hidden_script(text):
            return text
        self.messages.append({"role": "user", "content": _REMIND_NO_LEAK})
        retry = self._generate()
        if looks_like_hidden_script(retry):
            raise RuntimeError(
                "official LLM user simulator leaked the hidden Sierra script"
            )
        return retry

    def _generate(self) -> str:
        from openai import OpenAI

        from ageneval.task.core.openai_compat import install_openai_compat, rewrite_token_kwargs

        install_openai_compat()
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE") or None,
        )
        kwargs = rewrite_token_kwargs(
            {"model": self.model, "messages": self.messages, "max_tokens": 256}
        )
        res = client.chat.completions.create(**kwargs)
        text = ""
        if res.choices:
            text = str(getattr(res.choices[0].message, "content", None) or "")
        self.messages.append({"role": "assistant", "content": text})
        return text.strip()


def load_user(strategy: str | None = None, model: str | None = None) -> UserSimulationEnv:
    """Official Sierra path is the LLM user only. ``strategy`` is ignored."""
    _ = strategy
    return LLMUserSimulationEnv(model=model)
