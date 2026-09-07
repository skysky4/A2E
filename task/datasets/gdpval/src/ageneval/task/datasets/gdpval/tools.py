"""Official GDPval computer-use tools.

Roles match the official agent: reference files on disk, Web Search / Fetch,
View Image, Code Exec, Finish, Abandon, plus write/list so the agent can
submit real files. Execution goes through ``GDPSandbox`` (E2B when
``E2B_API_KEY`` is set). Contents are never dumped into the prompt.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ageneval.task.datasets.gdpval.sandbox import GDPSandbox, sandbox_from_state

_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_TEXT_SUFFIX = {".txt", ".md", ".csv", ".tsv", ".json", ".py", ".xml", ".html", ".log"}


def get_gdpval_tool_schemas() -> list[dict[str, Any]]:
    return [
        _fn("list_reference_files", "List reference and workspace files in the sandbox.", {}),
        _fn(
            "read_file",
            "Read a sandbox file. Page long files with offset/limit. Office files are extracted, not truncated to 12k in the prompt.",
            {
                "path": {"type": "string", "description": "Basename or relative path"},
                "offset": {"type": "integer", "description": "Character offset"},
                "limit": {"type": "integer", "description": "Max characters (default 80000)"},
            },
            required=["path"],
        ),
        _fn(
            "web_search",
            "Official Web Search. Use when the task needs current or cited facts.",
            {"query": {"type": "string"}},
            required=["query"],
        ),
        _fn(
            "web_fetch",
            "Official Web Fetch. Open a URL returned by web_search or named in the task.",
            {"url": {"type": "string"}},
            required=["url"],
        ),
        _fn(
            "view_image",
            "Official View Image. Inspect an image file in the sandbox.",
            {"path": {"type": "string"}},
            required=["path"],
        ),
        _fn(
            "code_exec",
            "Official Code Exec. Run Python in the sandbox workspace (cwd has the reference files).",
            {"code": {"type": "string"}},
            required=["code"],
        ),
        _fn(
            "write_file",
            "Write a deliverable file into the sandbox workspace.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            required=["path", "content"],
        ),
        _fn(
            "finish",
            "Submit one or more real workspace files and stop. Official GDPval requires real files, not prompt text.",
            {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace basenames to submit",
                },
                "summary": {"type": "string"},
            },
            required=["files"],
        ),
        _fn(
            "abandon",
            "Give up on this task when it cannot be completed.",
            {"reason": {"type": "string"}},
            required=["reason"],
        ),
    ]


def _fn(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    params: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def _box(state: Mapping[str, Any]) -> GDPSandbox:
    st = state if isinstance(state, dict) else {}
    return sandbox_from_state(st)


def _local_path(name: str, state: Mapping[str, Any]) -> Path | None:
    raw = (name or "").strip()
    if not raw:
        return None
    base = os.path.basename(raw)
    refs = (state or {}).get("reference_files") or {}
    if isinstance(refs, dict):
        if raw in refs and Path(str(refs[raw])).is_file():
            return Path(str(refs[raw]))
        if base in refs and Path(str(refs[base])).is_file():
            return Path(str(refs[base]))
        for path in refs.values():
            p = Path(str(path))
            if p.name == base and p.is_file():
                return p
    ws = (state or {}).get("workspace")
    if ws:
        cand = Path(str(ws)) / base
        if cand.is_file():
            return cand
    return None


def _extract_text(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIX:
        return {
            "path": path.name,
            "type": "image",
            "bytes": path.stat().st_size,
            "note": "Call view_image to inspect pixels. File is on disk in the sandbox.",
        }
    try:
        if suffix in _TEXT_SUFFIX:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix in {".xlsx", ".xlsm", ".xls"}:
            text = _xlsx_text(path)
        elif suffix == ".pdf":
            text = _pdf_text(path)
        elif suffix in {".docx"}:
            text = _docx_text(path)
        elif suffix in {".pptx", ".ppt"}:
            text = _pptx_text(path)
        else:
            # Keep the bytes on disk; do not inject a "binary attachment" prompt dump.
            return {
                "path": path.name,
                "type": "file",
                "bytes": path.stat().st_size,
                "suffix": suffix,
                "abs": str(path),
                "note": "File is in the sandbox. Use code_exec to parse it.",
            }
    except Exception as exc:  # noqa: BLE001
        return {"path": path.name, "error": str(exc)[:300], "abs": str(path), "bytes": path.stat().st_size}
    return {"path": path.name, "total_chars": len(text), "text": text}


def _xlsx_text(path: Path) -> str:
    import openpyxl  # type: ignore

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    chunks: list[str] = []
    for sheet in wb.worksheets:
        chunks.append(f"# sheet {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(cells):
                chunks.append("\t".join(cells))
    return "\n".join(chunks)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(path: Path) -> str:
    import docx  # type: ignore

    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def _pptx_text(path: Path) -> str:
    from pptx import Presentation  # type: ignore

    pres = Presentation(str(path))
    chunks: list[str] = []
    for i, slide in enumerate(pres.slides, start=1):
        chunks.append(f"# slide {i}")
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _page(extracted: dict[str, Any], *, offset: int, limit: int) -> dict[str, Any]:
    text = extracted.get("text")
    if not isinstance(text, str):
        return extracted
    start = max(0, int(offset or 0))
    cap = max(1, int(limit or 80000))
    sliced = text[start : start + cap]
    return {
        **{k: v for k, v in extracted.items() if k != "text"},
        "offset": start,
        "chars": len(sliced),
        "total_chars": len(text),
        "truncated": start + cap < len(text),
        "text": sliced,
    }


def _view_image(path: Path) -> dict[str, Any]:
    meta = {
        "path": path.name,
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "abs": str(path),
    }
    try:
        from openai import OpenAI

        from ageneval.task.core.openai_compat import install_openai_compat, rewrite_token_kwargs

        install_openai_compat()
        raw = path.read_bytes()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(raw).decode("ascii")
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE") or None,
        )
        model = os.environ.get("A2E_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.6-sol"
        kwargs = rewrite_token_kwargs(
            {
                "model": model,
                "max_tokens": 800,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image for a professional work task. "
                                "Transcribe visible text. Note layout, tables, and figures.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
            }
        )
        res = client.chat.completions.create(**kwargs)
        desc = ""
        if res.choices:
            desc = str(getattr(res.choices[0].message, "content", None) or "")
        meta["description"] = desc
        return meta
    except Exception as exc:  # noqa: BLE001
        meta["note"] = f"Image is in the sandbox ({path}). Vision describe failed: {exc}"[:300]
        return meta


def gdpval_tool_executor(name: str, arguments: Mapping[str, Any], state: Mapping[str, Any]) -> Any:
    args = dict(arguments or {})
    st = state if isinstance(state, dict) else {}
    box = _box(st)
    st["sandbox_backend"] = box.backend
    st["workspace"] = str(box.workspace)
    if name == "list_reference_files":
        items = box.list_files()
        return {"files": items, "n": len(items), "backend": box.backend}
    if name == "read_file":
        path = _local_path(str(args.get("path") or ""), st)
        if path is None:
            try:
                data = box.read_bytes(str(args.get("path") or ""))
            except Exception:
                return {"error": "file not found", "path": args.get("path")}
            tmp = box.workspace / os.path.basename(str(args.get("path") or "file.bin"))
            tmp.write_bytes(data)
            path = tmp
        return _page(
            _extract_text(path),
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 80000),
        )
    if name == "view_image":
        path = _local_path(str(args.get("path") or ""), st)
        if path is None:
            return {"error": "image not found", "path": args.get("path")}
        return _view_image(path)
    if name == "web_search":
        from ageneval.task.datasets.deepsearchqa.tools import _web_search

        return _web_search(str(args.get("query") or ""))
    if name == "web_fetch":
        from ageneval.task.datasets.deepsearchqa.tools import _open_url

        return _open_url(str(args.get("url") or ""))
    if name == "code_exec":
        code = str(args.get("code") or "")
        if not code.strip():
            return {"error": "empty code"}
        return box.exec_python(code, timeout=30)
    if name == "write_file":
        rel = os.path.basename(str(args.get("path") or "deliverable.txt"))
        data = str(args.get("content") or "").encode("utf-8")
        dest = box.write_bytes(rel, data)
        written = list(st.get("written_files") or [])
        written.append(rel)
        st["written_files"] = written
        refs = st.get("reference_files")
        if isinstance(refs, dict):
            refs[rel] = str(box.workspace / rel)
        return {"path": rel, "bytes": len(data), "abs": dest}
    if name == "finish":
        names = [os.path.basename(str(x)) for x in (args.get("files") or [])]
        submitted: list[dict[str, Any]] = []
        for rel in names:
            path = _local_path(rel, st) or (box.workspace / rel)
            if path.is_file():
                submitted.append({"name": rel, "path": str(path), "bytes": path.stat().st_size})
        if not submitted:
            return {
                "error": "finish requires one or more real files in the sandbox workspace",
                "files": names,
            }
        st["finished"] = True
        st["submitted_files"] = [row["name"] for row in submitted]
        st["submitted_file_meta"] = submitted
        st["finish_summary"] = str(args.get("summary") or "")
        return {"ok": True, "submitted": submitted, "backend": box.backend}
    if name == "abandon":
        st["abandoned"] = True
        st["abandon_reason"] = str(args.get("reason") or "")
        return {"ok": True, "abandoned": True, "reason": st["abandon_reason"]}
    return {
        "error": f"unknown tool '{name}'",
        "available": [s["function"]["name"] for s in get_gdpval_tool_schemas()],
    }
