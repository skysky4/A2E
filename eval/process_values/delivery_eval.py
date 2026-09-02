"""submitted: SWE-bench submitted vs empty_patch, independent of resolved.

Binary CODE metric. A run is submitted (1.0) if the trajectory contains a
candidate terminal deliverable; empty (0.0) if tools ran but none was emitted.
Does not use the official verifier (that is correctness / resolved).

Deliverable locators follow the task-agnostic submit contract in:
- SWE-bench (ICLR 2024): non-empty patch vs empty_patch
- GAIA: non-empty FINAL ANSWER / final_answer
- WebArena (ICLR 2024): stop/submit action
- SPA-Bench (ICLR 2025): self-reported completion is submitted; success is separate
- Terminal-Bench: a written candidate file is the patch analogue
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.eval_common import (
    _as_dict,
    _enrich_output,
    _final_answer,
    _task_output,
    _unscored,
)

_SUBMIT_TOOLS = frozenset(
    {
        "stop",
        "submit",
        "final_answer",
        "finish",
        "terminate",
        "done",
    }
)
_PATCH_KEYS = (
    "model_patch",
    "patch",
    "git_diff",
    "diff",
    "prediction",
    "submission",
)
_FINAL_ANSWER_RE = re.compile(r"\bFINAL\s+ANSWER\s*:\s*(\S.+)", re.I)
_JSON_PATH = re.compile(
    r'"(?:path|file_path|filename|dest|destination|file)"\s*:\s*"([^"]+)"',
    re.I,
)
_REDIRECT = re.compile(
    r"(?:cat|tee|printf|echo|install)\b[^\n]{0,240}?(?:>{1,2}|tee(?:\s+-a)?)\s*"
    r"([^\s;|&<>]+)",
    re.I,
)
_SIMPLE_REDIRECT = re.compile(r"(?:^|[;&|\n])\s*(?:cat\s+>{1,2}|tee(?:\s+-a)?)\s*([^\s;|&<>]+)", re.I)
_CP_MV = re.compile(r"\b(?:cp|mv)\s+(?:-[a-zA-Z]+\s+)*(\S+)\s+(\S+)", re.I)
_SED_INPLACE = re.compile(r"\bsed\s+-i(?:\s+'[^']*'|\s+\"[^\"]*\")+\s+(\S+)", re.I)
_OPEN_WRITE = re.compile(r"""open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]w""")
_INPUT_BASENAMES = frozenset(
    {
        "eval.py",
        "eval.sh",
        "test.py",
        "tests.py",
        "instruction.md",
        "readme.md",
        "os-release",
    }
)
_SCRATCH_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/", "/proc/", "/etc/")
_EMPTY_MARKERS = frozenset({"", "none", "n/a", "null", "nil", "{}", "[]", "-", "n\\a"})


def _norm_path(raw: str) -> str:
    text = str(raw or "").strip().strip("`\"'")
    text = text.rstrip(".,;:)]}")
    if text.startswith("./"):
        text = text[2:]
    return text


def _basename(path: str) -> str:
    return _norm_path(path).rsplit("/", 1)[-1].lower()


def _is_scratch(path: str) -> bool:
    norm = _norm_path(path)
    if not norm or norm in {".", "/", "/app", "/workspace"}:
        return True
    lowered = norm.lower()
    return lowered.startswith(_SCRATCH_PREFIXES) or lowered in {"/dev/null", "/dev/stdout"}


def _is_artifact_path(path: str) -> bool:
    norm = _norm_path(path)
    if not norm or _is_scratch(norm):
        return False
    if norm.startswith("/app/") or norm.startswith("/workspace/") or "/" not in norm:
        return _basename(norm) not in _INPUT_BASENAMES and ("." in _basename(norm) or "/" in norm)
    return _basename(norm) not in _INPUT_BASENAMES


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list)):
        return bool(value)
    text = str(value).strip()
    return bool(text) and text.lower() not in _EMPTY_MARKERS


def _payload_text(call: Mapping[str, Any]) -> str:
    args = _as_dict(call.get("arguments"))
    chunks: list[str] = []

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                chunks.append(text)
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in {"mime_type"}:
                    continue
                visit(nested, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in list(value)[:30]:
                visit(item, depth + 1)

    visit(args)
    return "\n".join(chunks)


def _written_paths(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    written: list[str] = []

    def add(path: str) -> None:
        norm = _norm_path(path)
        if not norm or _is_scratch(norm):
            return
        if norm not in written:
            written.append(norm)

    for call in calls:
        payload = _payload_text(call)
        if not payload:
            continue
        args = _as_dict(call.get("arguments"))
        for key in ("path", "file_path", "filename", "dest", "destination", "file"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                command = str(args.get("command") or call.get("name") or "").lower()
                if command in {"create", "str_replace", "insert", "undo_edit", ""} or key == "path":
                    add(value)
        kwargs = args.get("kwargs")
        if isinstance(kwargs, Mapping):
            for key in ("path", "file_path", "filename"):
                value = kwargs.get(key)
                if isinstance(value, str):
                    add(value)
        for pattern in (_JSON_PATH, _REDIRECT, _SIMPLE_REDIRECT, _OPEN_WRITE, _SED_INPLACE):
            for match in pattern.finditer(payload):
                add(match.group(1))
        for match in _CP_MV.finditer(payload):
            add(match.group(2))
    return written


def _looks_like_patch(value: Any) -> bool:
    text = str(value or "").lstrip()
    if not text:
        return False
    return text.startswith(("diff --git", "--- a/", "+++ b/", "Index: "))


def _patch_from_output(task: Mapping[str, Any]) -> str:
    for key in _PATCH_KEYS:
        value = task.get(key)
        if _looks_like_patch(value):
            return f"{key}"
    return ""


def _answer_from_output(output: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    answer = _final_answer(output) or str(task.get("answer") or task.get("model_answer") or "")
    if _nonempty(answer):
        return "final_answer"
    match = _FINAL_ANSWER_RE.search(str(task.get("response") or task.get("output") or answer))
    if match and _nonempty(match.group(1)):
        return "gaia_final_answer"
    return ""


def _submit_tool(calls: Sequence[Mapping[str, Any]]) -> str:
    for call in calls:
        name = str(call.get("name") or "").strip().lower()
        if name not in _SUBMIT_TOOLS:
            continue
        payload = _payload_text(call)
        args = _as_dict(call.get("arguments"))
        content = (
            args.get("answer")
            or args.get("final_answer")
            or args.get("text")
            or args.get("content")
            or payload
        )
        if name == "stop" or _nonempty(content):
            return name
    return ""


def _file_artifacts(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        path
        for path in _written_paths(calls)
        if _is_artifact_path(path) or _is_artifact_path(path if path.startswith("/") else f"/app/{path}")
    ]


def make_submitted(
    spans_by_example_id: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """SWE-bench submitted vs empty_patch. Independent of verifier resolved/correctness."""

    spans_map = spans_by_example_id or {}

    def submitted(
        output: dict[str, Any],
        expected: dict[str, Any],
        input: dict[str, Any],
        example: Any = None,
    ) -> dict[str, Any]:
        del expected, input
        example_id = str(getattr(example, "id", "") or _as_dict(example).get("id") or "")
        enriched = _enrich_output(output, spans_map.get(example_id, []))
        task = _task_output(enriched)
        calls = [_as_dict(call) for call in (enriched.get("tool_calls_full") or [])]
        channels: list[str] = []
        answer_ch = _answer_from_output(enriched, task)
        if answer_ch:
            channels.append(answer_ch)
        patch_ch = _patch_from_output(task)
        if patch_ch:
            channels.append(patch_ch)
        stop_ch = _submit_tool(calls)
        if stop_ch:
            channels.append(f"tool:{stop_ch}")
        files = _file_artifacts(calls)
        if files:
            channels.append(f"file:{files[0]}")
        if channels:
            return {
                "score": 1.0,
                "label": "submitted",
                "explanation": (
                    "Candidate terminal deliverable present (SWE-bench submitted / GAIA FINAL "
                    f"ANSWER / WebArena stop analogue); channels={channels[:4]}. "
                    "Independent of correctness/resolved."
                )[:1000],
            }
        if not calls:
            return _unscored(
                "No tool invocations and no final_answer/patch/stop deliverable were observed; "
                "submitted is not applicable (cannot distinguish empty submission from a missing trace)."
            )
        return {
            "score": 0.0,
            "label": "empty",
            "explanation": (
                f"Tools ran ({len(calls)} call(s)) but no candidate submit was found "
                "(SWE-bench empty_patch analogue): no non-empty final_answer, patch, "
                "stop/submit tool, or written task file. score=0.0."
            )[:1000],
        }

    submitted.__name__ = "submitted"
    submitted.__qualname__ = "submitted"
    return submitted
