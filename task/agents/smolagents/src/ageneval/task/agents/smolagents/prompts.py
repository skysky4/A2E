"""Optional additional-instruction builder for smolagents.

smolagents ships with its own elaborate code-agent system prompt, so we
only pass a short *additional_instructions* string derived from the
binding's domain-specific system prompt. Keep the binding's tool docs
out of here — smolagents builds them automatically from each
``Tool.description`` / ``Tool.inputs``.
"""

from __future__ import annotations


_FALLBACK = (
    "Follow the user task. When you have the answer, call final_answer. "
    "Do not leave the final answer empty."
)


def build_additional_instructions(binding_prompt: str) -> str:
    """Dataset policy for smolagents ``instructions`` — never empty.

    The binding's full prompt typically appends an inline tool catalog (so
    that bare LLM agents can still use the tools without an MCP server).
    smolagents already exposes the tools natively, so the catalog would be
    duplicate noise. We keep the textual policy preamble and drop the
    "AVAILABLE TOOLS:" block. If that would wipe the prompt, keep the
    original text so the system prompt is never blank.
    """
    raw = (binding_prompt or "").strip()
    if not raw:
        return _FALLBACK
    marker = "AVAILABLE TOOLS:"
    head = raw.split(marker, 1)[0].strip()
    return head or raw or _FALLBACK
