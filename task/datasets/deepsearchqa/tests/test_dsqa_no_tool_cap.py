"""Official DeepSearchQA has no 2-search / 2-open cap."""

from __future__ import annotations

from ageneval.task.datasets.deepsearchqa.binding import build_deepsearchqa_binding
from ageneval.task.core.native_tools import execute_unique_recorded
from ageneval.task.core.result import ToolCall


def test_prompt_has_no_two_call_cap():
    prompt = build_deepsearchqa_binding().render_system_prompt()
    assert "at most twice" not in prompt.lower()
    assert "at most two" not in prompt.lower()


def test_hf_import_ignores_local_datasets_dir(tmp_path, monkeypatch):
    from ageneval.task.datasets.deepsearchqa.loader import _import_hf_load_dataset

    local = tmp_path / "datasets"
    local.mkdir()
    (local / "__init__.py").write_text("raise RuntimeError('local datasets package')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    load_dataset = _import_hf_load_dataset()
    assert callable(load_dataset)
    import datasets as hf

    path = str(getattr(hf, "__file__", "") or "")
    assert "site-packages" in path.replace("\\", "/")


def test_executor_allows_three_distinct_web_searches():
    recorder: list[ToolCall] = []
    state: dict = {}

    def executor(name, args, _state):
        return {"query": args.get("query"), "results": [{"url": "https://example.com"}]}

    for i, q in enumerate(("alpha", "beta", "gamma")):
        text = execute_unique_recorded(
            tool_name="web_search",
            kwargs={"query": q},
            executor=executor,
            initial_state=state,
            recorder=recorder,
        )
        assert "budget exhausted" not in text
    assert [tc.name for tc in recorder] == ["web_search", "web_search", "web_search"]


def test_named_page_does_not_skip_live_index(monkeypatch):
    from ageneval.task.datasets.deepsearchqa import tools as dsqa_tools

    called: list[str] = []

    monkeypatch.setattr(
        dsqa_tools,
        "_named_page_search",
        lambda q: [
            {
                "title": "Federal Reserve H.10",
                "snippet": "Official H.10",
                "url": "https://www.federalreserve.gov/releases/h10/hist/",
            }
        ],
    )

    def _bing(q):
        called.append("bing")
        return [
            {
                "title": "Federal Reserve G.5A",
                "snippet": "Annual averages from Bing",
                "url": "https://www.federalreserve.gov/releases/g5a/current/",
            }
        ]

    monkeypatch.setattr(dsqa_tools, "_bing_search", _bing)
    monkeypatch.setattr(dsqa_tools, "_brave_search", lambda q: called.append("brave") or [])
    out = dsqa_tools._web_search(
        "site:federalreserve.gov/releases/h10 annual average exchange rates 2023"
    )
    assert "bing" in called
    assert "bing" in out["source"]
    assert any("g5a" in h["url"] for h in out["results"])
