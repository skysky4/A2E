"""GDPval official tools: files on disk, no 12k prompt dump."""

from __future__ import annotations

from pathlib import Path

from ageneval.task.datasets.gdpval.binding import build_gdpval_binding
from ageneval.task.datasets.gdpval.loader import _build_instruction
from ageneval.task.datasets.gdpval.tools import gdpval_tool_executor, get_gdpval_tool_schemas


def test_binding_exposes_official_tool_roles():
    names = {s["function"]["name"] for s in get_gdpval_tool_schemas()}
    from ageneval.task.core.native_tools import parameters_block, pydantic_args_model

    empty = next(s for s in get_gdpval_tool_schemas() if s["function"]["name"] == "list_reference_files")
    schema = pydantic_args_model(
        "list_reference_files", parameters_block(empty)
    ).model_json_schema()
    assert schema.get("additionalProperties") is False

    assert names >= {
        "list_reference_files",
        "read_file",
        "web_search",
        "web_fetch",
        "view_image",
        "code_exec",
        "write_file",
        "finish",
        "abandon",
    }
    prompt = build_gdpval_binding().render_system_prompt()
    assert "read_file" in prompt
    assert "NOT included in this text-only context" not in prompt


def test_instruction_lists_files_without_dumping_contents(tmp_path: Path):
    csv = tmp_path / "pop.csv"
    csv.write_text("id,name\n1,a\n" * 2000, encoding="utf-8")
    inst = _build_instruction("Make a memo.", {"pop.csv": str(csv)}, [])
    assert "pop.csv" in inst
    assert "[Attached input file contents]" not in inst
    assert "binary attachment" not in inst
    assert "id,name" not in inst


def test_read_file_pages_without_12k_wall(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello-world-" * 2000, encoding="utf-8")
    state = {"reference_files": {"notes.txt": str(path)}, "workspace": str(tmp_path)}
    listed = gdpval_tool_executor("list_reference_files", {}, state)
    assert listed["n"] == 1
    first = gdpval_tool_executor("read_file", {"path": "notes.txt", "limit": 20}, state)
    assert first["text"].startswith("hello-world-")
    assert first["truncated"] is True
    assert first["total_chars"] > 20


def test_local_reference_cache_resolves():
    from ageneval.task.datasets.gdpval.loader import _fetch_reference_file

    root = Path("/data/agenteval/a2e-data-full-20260817/gdpval-files/reference_files")
    if not root.is_dir():
        return
    sample = next(root.rglob("*"), None)
    if sample is None or not sample.is_file():
        return
    found = _fetch_reference_file(sample.name)
    assert found is not None
    assert Path(found).is_file()


def test_code_exec_and_finish(tmp_path: Path):
    state = {"reference_files": {}, "workspace": str(tmp_path)}
    out = gdpval_tool_executor("code_exec", {"code": "print(2+2)"}, state)
    assert out["returncode"] == 0
    assert "4" in out["stdout"]
    gdpval_tool_executor("write_file", {"path": "out.txt", "content": "done"}, state)
    empty = gdpval_tool_executor("finish", {"files": [], "summary": "ok"}, state)
    assert empty.get("error")
    fin = gdpval_tool_executor("finish", {"files": ["out.txt"], "summary": "ok"}, state)
    assert fin["ok"] is True
    assert state["finished"] is True
    assert state["submitted_files"] == ["out.txt"]


def test_sandbox_copies_attachment(tmp_path: Path):
    from ageneval.task.datasets.gdpval.sandbox import open_gdp_sandbox

    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    box = open_gdp_sandbox()
    box.put_file("src.csv", src)
    listed = {row["name"] for row in box.list_files()}
    assert "src.csv" in listed
    assert (box.workspace / "src.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert box.backend in {"local", "e2b"}


def test_crewai_gdp_presentation_requires_official_tools():
    from ageneval.task.agents.crewai.agent import _task_presentation
    from ageneval.task.datasets.gdpval.binding import build_gdpval_binding

    goal, expected, description = _task_presentation(
        build_gdpval_binding(), "Build April financials."
    )
    assert "list_reference_files" in description
    assert "finish" in description
    assert "Do not answer from memory" in description
    assert "finish" in expected.lower() or "files" in expected.lower()
    assert "sandbox" in goal.lower() or "GDPval" in goal


def test_crewai_implied_plan_dispatches_list_and_continues():
    from ageneval.task.agents.crewai.agent import (
        _dispatch_implied_official_start,
        _gdp_continue_prompt,
        _needs_gdp_continue,
    )
    from ageneval.task.core.result import ToolCall

    called: list[str] = []

    class _ListTool:
        name = "list_reference_files"

        def _run(self, **kwargs):
            called.append("list")
            return '{"n":1}'

    class _ReadTool:
        name = "read_file"

        def _run(self, **kwargs):
            called.append("read")
            return "ok"

    tools = [_ListTool(), _ReadTool()]
    _dispatch_implied_official_start(
        tools,
        "Thought: I'll first inventory the sandbox files, then inspect every attachment.",
    )
    assert called == ["list"]
    listed = [
        ToolCall(
            name="list_reference_files",
            arguments={},
            result={"n": 1, "files": [{"name": "a.xlsx"}]},
        )
    ]
    assert _needs_gdp_continue(tools, listed) is True
    prompt = _gdp_continue_prompt("Build April financials.", listed)
    assert "read_file" in prompt
    assert "function calls" in prompt
    assert _needs_gdp_continue(tools, []) is False
    from ageneval.task.agents.crewai.agent import (
        _dispatch_gdp_reads,
        _run_named_tool,
    )

    called.clear()
    assert _run_named_tool(tools, "list_reference_files") is True
    assert called == ["list"]
    _dispatch_gdp_reads(
        tools,
        [
            ToolCall(
                name="list_reference_files",
                arguments={},
                result={"files": [{"name": "a.xlsx"}, {"name": "b.xlsx"}]},
            )
        ],
    )
    assert called == ["list", "read", "read"]
    from ageneval.task.agents.crewai.agent import _tool_calls_to_react

    react = _tool_calls_to_react("read_file", {"path": "a.xlsx"})
    assert "Action: read_file" in react
    assert "a.xlsx" in react

    class _LLM:
        def call(self, messages, tools=None, available_functions=None, **kwargs):
            if available_functions and "list_reference_files" in available_functions:
                return available_functions["list_reference_files"]()
            return "Thought: plan"

    class _BoundList:
        name = "list_reference_files"

        def _run(self, **kwargs):
            called.append("native-list")
            return '{"n":1}'

    from ageneval.task.agents.crewai.agent import _bind_crewai_native_tools

    wrapped = _bind_crewai_native_tools(
        _LLM(),
        [
            {
                "type": "function",
                "function": {"name": "list_reference_files", "parameters": {"type": "object"}},
            }
        ],
        [_BoundList()],
    )
    out = wrapped.call("list files")
    assert "native-list" in called
    assert "n" in out
    again = wrapped.call("list files")
    assert "duplicate" in again


def test_gdp_grader_sees_submitted_files():
    from ageneval.task.datasets.gdpval.grader import make_gdp_grader

    seen: dict[str, str] = {}

    class _LLM:
        def generate_text(self, prompt: str) -> str:
            seen["prompt"] = prompt
            return "SCORE=1; EXPLANATION=Workbook was submitted."

    grade = make_gdp_grader(_LLM())(
        {
            "final_answer": "see attached workbook",
            "gdp_submitted": ["Aurisic_Financials_4-25-1.xlsx"],
            "gdp_submitted_meta": [
                {"name": "Aurisic_Financials_4-25-1.xlsx", "bytes": 12000}
            ],
            "gdp_n_attachments": 17,
            "gdp_finish_summary": "April financials",
        },
        {"expected_outputs": ["Produce the April workbook"]},
        {"instruction": "Build April financials from the attachments."},
    )
    prompt = seen["prompt"]
    assert "Aurisic_Financials_4-25-1.xlsx" in prompt
    assert "12000 bytes" in prompt
    assert "Reference attachments loaded: 17" in prompt
    assert "pairwise Elo" in prompt
    assert grade["score"] == 1.0
