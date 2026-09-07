#!/usr/bin/env python3
"""Live official-path checks. Prints evidence, never prints API keys."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from ageneval.task.core.native_tools import execute_unique_recorded
from ageneval.task.core.result import ToolCall
from ageneval.task.datasets.deepsearchqa.binding import build_deepsearchqa_binding
from ageneval.task.datasets.deepsearchqa.tools import _web_search
from ageneval.task.datasets.tau_bench.user_sim import (
    LLMUserSimulationEnv,
    hidden_script_from_example,
    looks_like_hidden_script,
    load_user,
    official_tau_example_input,
)

SCRIPT = (
    "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish "
    "to exchange the mechanical keyboard."
)


def _ok(name: str, cond: bool, detail: str) -> bool:
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")
    return cond


def check_upload_and_naive() -> bool:
    user = load_user("naive")
    payload = official_tau_example_input(SCRIPT, {"__tau_domain__": "retail"})
    recovered = hidden_script_from_example(payload)
    ok = True
    ok &= _ok("load_user(naive)", type(user).__name__ == "LLMUserSimulationEnv", type(user).__name__)
    ok &= _ok("upload instruction is not script", SCRIPT not in payload["instruction"], payload["instruction"][:80])
    ok &= _ok("hidden recovered", recovered == SCRIPT, recovered[:60])
    ok &= _ok("detector", looks_like_hidden_script(SCRIPT), "script flagged")
    return ok


def check_live_user_sim() -> bool:
    user = LLMUserSimulationEnv()
    opening = user.reset(SCRIPT)
    leaked = looks_like_hidden_script(opening)
    empty = not (opening or "").strip()
    ok = True
    ok &= _ok("live opening nonempty", not empty, repr(opening[:160]))
    ok &= _ok("live opening not script", not leaked, repr(opening[:160]))
    ok &= _ok("live opening != script", opening.strip() != SCRIPT, "compared")
    return ok


def check_live_search() -> bool:
    prompt = build_deepsearchqa_binding().render_system_prompt()
    ok = True
    ok &= _ok("prompt no 2+2", "at most twice" not in prompt.lower() and "at most two" not in prompt.lower(), "prompt clean")
    recorder: list[ToolCall] = []
    state: dict = {}

    def executor(name, args, _state):
        return _web_search(str(args.get("query") or ""))

    queries = (
        "site:federalreserve.gov/releases/h10 annual average exchange rates 2023",
        "Federal Reserve G.5A 2023 India Japan",
        "site:federalreserve.gov/releases/g5a/current annual averages",
    )
    sources = []
    for q in queries:
        text = execute_unique_recorded(
            tool_name="web_search",
            kwargs={"query": q},
            executor=executor,
            initial_state=state,
            recorder=recorder,
        )
        ok &= _ok(f"search allowed {q[:40]}", "budget exhausted" not in text.lower(), text[:120].replace("\n", " "))
        try:
            data = json.loads(text) if text.startswith("{") else {}
        except json.JSONDecodeError:
            data = {}
        sources.append(str(data.get("source") or ""))
        hits = data.get("results") or []
        ok &= _ok(f"search hits {q[:40]}", bool(hits), f"source={data.get('source')} n={len(hits)}")
    ok &= _ok("three distinct searches recorded", len(recorder) == 3, str([tc.name for tc in recorder]))
    ok &= _ok(
        "live index attempted",
        any("bing" in s or "brave" in s or "named_page" in s for s in sources),
        str(sources),
    )
    return ok


def check_live_gdp() -> bool:
    from ageneval.task.datasets.gdpval.loader import (
        _MAX_RUBRIC_CHARS,
        _build_instruction,
        _fetch_reference_file,
        _file_names,
        _load_gdpval_split,
    )
    from ageneval.task.datasets.gdpval.sandbox import open_gdp_sandbox

    task_id = "ee09d943-5a11-430a-b7a2-971b4e9b01b5"
    task = None
    for i, row in enumerate(_load_gdpval_split("train")):
        tid = str(row.get("task_id") or f"gdpval-{i:04d}")
        if tid != task_id:
            continue
        prompt = str(row.get("prompt", "") or "")
        ref_rels = [str(p) for p in (row.get("reference_files") or [])]
        ref_names = _file_names(ref_rels)
        found_src: dict[str, str] = {}
        missing: list[str] = []
        for rel, name in zip(ref_rels, ref_names):
            fetched = _fetch_reference_file(rel)
            if fetched is None:
                missing.append(name)
                continue
            found_src[name] = str(fetched)
        box = open_gdp_sandbox()
        found: dict[str, str] = {}
        for name, src in found_src.items():
            dest = box.put_file(name, src)
            local = box.workspace / name
            found[name] = str(local if local.is_file() else dest)
        from ageneval.task.core.dataset import TaskInput

        task = TaskInput(
            task_id=tid,
            instruction=_build_instruction(prompt, found, missing),
            initial_state={
                "reference_files": found,
                "reference_names": ref_names,
                "missing_reference_files": missing,
                "workspace": str(box.workspace),
                "sandbox_backend": box.backend,
            },
            expected_outputs=(),
            metadata={"n_attachments_loaded": len(found)},
        )
        break
    if task is None:
        return _ok("gdp task found", False, task_id)
    inst = task.instruction or ""
    state = task.initial_state if isinstance(task.initial_state, dict) else {}
    refs = state.get("reference_files") or {}
    ok = True
    ok &= _ok("gdp no dump", "[Attached input file contents]" not in inst and "binary attachment" not in inst.lower(), inst[-200:].replace("\n", " "))
    ok &= _ok("gdp no text-only placeholder", "NOT included in this text-only context" not in inst, "clean")
    ok &= _ok("gdp files listed not dumped", "read_file" in inst and "id,name" not in inst, "listed")
    ok &= _ok("gdp attachments loaded", len(refs) > 0, f"n={len(refs)}")
    existing = 0
    for name, path in list(refs.items())[:5]:
        if Path(str(path)).is_file():
            existing += 1
    ok &= _ok("gdp files on disk", existing == min(5, len(refs)), f"{existing}/{min(5, len(refs))}")
    from ageneval.task.datasets.gdpval.tools import gdpval_tool_executor

    listed = gdpval_tool_executor("list_reference_files", {}, state)
    ok &= _ok("list_reference_files", int(listed.get("n") or 0) == len(refs), str(listed.get("n")))
    first = next(iter(refs))
    read = gdpval_tool_executor("read_file", {"path": first, "limit": 80}, state)
    ok &= _ok("read_file works", bool(read.get("text") or read.get("path")), str({k: read.get(k) for k in ("path", "truncated", "total_chars")}))
    return ok


def main() -> int:
    print("model", os.environ.get("A2E_MODEL"))
    print("base", os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    results = {
        "upload": check_upload_and_naive(),
        "user_sim": check_live_user_sim(),
        "search": check_live_search(),
        "gdp": check_live_gdp(),
    }
    print("SUMMARY", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
