"""Sierra-style τ-bench user simulation.

The LLM strategy follows the upstream user prompt. ``naive`` is a deterministic
offline strategy intended for tests and local smoke runs.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

STOP_TOKEN = "###STOP###"

_OFFICIAL_USER_PROMPT = """You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '{stop}' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class UserSimulationEnv(Protocol):
    def reset(self, instruction: str | None = None) -> str: ...

    def step(self, content: str) -> str: ...


class NaiveUserSimulationEnv:
    """Deterministic user: brief opening, hidden script, then confirmations."""

    def __init__(self) -> None:
        self.instruction = ""
        self._turns = 0

    def reset(self, instruction: str | None = None) -> str:
        self.instruction = (instruction or "").strip()
        self._turns = 0
        return "Hi, I need help with an order I received."

    def step(self, content: str) -> str:
        self._turns += 1
        if STOP_TOKEN in (content or ""):
            return STOP_TOKEN
        if self._turns == 1 and self.instruction:
            return self.instruction
        if self._turns >= 6:
            return STOP_TOKEN
        return "Yes, please go ahead."


class LLMUserSimulationEnv:
    """Upstream-style user simulation over the configured OpenAI endpoint."""

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
            f"\n\nInstruction: {instruction}\n" if instruction else ""
        )
        return _OFFICIAL_USER_PROMPT.format(
            instruction_display=instruction_display,
            stop=STOP_TOKEN,
        )

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._generate()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._generate()

    def _generate(self) -> str:
        from openai import OpenAI

        from ageneval.task.core.openai_compat import (
            install_openai_compat,
            rewrite_token_kwargs,
        )

        install_openai_compat()
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE") or None,
        )
        kwargs = rewrite_token_kwargs(
            {"model": self.model, "messages": self.messages, "max_tokens": 256}
        )
        response = client.chat.completions.create(**kwargs)
        text = ""
        if response.choices:
            text = str(getattr(response.choices[0].message, "content", None) or "")
        self.messages.append({"role": "assistant", "content": text})
        return text.strip() or STOP_TOKEN


def load_user(
    strategy: str | None = None,
    model: str | None = None,
) -> UserSimulationEnv:
    """Load the official LLM simulator or deterministic offline substitute."""
    resolved = (
        strategy or os.environ.get("A2E_TAU_USER_STRATEGY") or "llm"
    ).strip().lower()
    if resolved in {"naive", "script", "deterministic"}:
        return NaiveUserSimulationEnv()
    return LLMUserSimulationEnv(model=model)


__all__ = [
    "LLMUserSimulationEnv",
    "NaiveUserSimulationEnv",
    "STOP_TOKEN",
    "UserSimulationEnv",
    "load_user",
]
