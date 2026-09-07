"""GDPval binding — official tool roles, files on disk, no prompt-dump.

The agent's reply (or ``finish`` summary / written files) is the deliverable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ageneval.task.core import AgentBinding
from ageneval.task.datasets.gdpval.tools import (
    gdpval_tool_executor,
    get_gdpval_tool_schemas,
)

_SYSTEM_PROMPT = (
    "You are a top-tier professional completing a real-world, economically "
    "valuable work task inside an official GDPval computer-use sandbox "
    "(E2B when configured). Official tools: list_reference_files, read_file, "
    "view_image, web_search, web_fetch, code_exec, write_file, finish, abandon.\n"
    "- Reference files are already in the sandbox. Read them with tools.\n"
    "- Do not invent spreadsheet, PDF, slide, or image contents.\n"
    "- Use web_search / web_fetch when the task needs cited current facts.\n"
    "- Write one or more real deliverable files with write_file / code_exec, "
    "then call finish with those filenames.\n"
    "- finish requires real files on disk. Do not submit prompt-only text.\n"
    "- Match the format the task asks for. Do not ask clarifying questions."
)


def _build_system_prompt(tools: Sequence[Mapping[str, Any]]) -> str:
    tool_block = "\n".join(
        f"- {t['function']['name']}: {t['function'].get('description', '')}\n"
        f"  parameters: {json.dumps(t['function'].get('parameters', {}))}"
        for t in tools
    )
    return _SYSTEM_PROMPT + f"\nAVAILABLE TOOLS:\n{tool_block}\n"


def build_gdpval_binding() -> AgentBinding:
    tools = get_gdpval_tool_schemas()
    return AgentBinding(
        name="gdpval-aa",
        tool_schemas=tools,
        tool_executor=gdpval_tool_executor,
        system_prompt_builder=_build_system_prompt,
    )
