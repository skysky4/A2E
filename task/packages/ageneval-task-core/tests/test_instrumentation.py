from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ageneval.task.core import instrumentation


@pytest.mark.parametrize(
    ("framework", "expected_kwargs"),
    [
        (
            "crewai",
            {
                "use_event_listener": True,
                "create_llm_spans": True,
            },
        ),
        ("langchain", {}),
    ],
)
def test_install_instrumentor_enables_crewai_event_listener(
    monkeypatch: pytest.MonkeyPatch,
    framework: Any,
    expected_kwargs: dict[str, object],
) -> None:
    calls: list[dict[str, object]] = []

    class FakeInstrumentor:
        def instrument(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda _module_path: SimpleNamespace(
            CrewAIInstrumentor=FakeInstrumentor,
            LangChainInstrumentor=FakeInstrumentor,
        ),
    )
    provider = object()

    instrumentation._install_instrumentor(framework, provider)  # type: ignore[arg-type]

    assert calls == [{"tracer_provider": provider, **expected_kwargs}]
