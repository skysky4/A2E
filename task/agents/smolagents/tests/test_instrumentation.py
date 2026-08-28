from __future__ import annotations

from types import SimpleNamespace

import pytest
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from smolagents import OpenAIModel, OpenAIServerModel


def test_openai_model_aliases_emit_one_llm_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert OpenAIModel is OpenAIServerModel

    def fake_generate(
        _self: object,
        messages: list[object],
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert messages == []
        return SimpleNamespace(
            token_usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                total_tokens=18,
            ),
            role="assistant",
            content="ok",
            tool_calls=[],
        )

    monkeypatch.setattr(OpenAIModel, "generate", fake_generate)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = SmolagentsInstrumentor()

    try:
        instrumentor.instrument(tracer_provider=provider)
        model = object.__new__(OpenAIModel)
        model.model_id = "openai/smoke-model"
        model.kwargs = {}
        model.api_base = "http://localhost"
        model.last_input_token_count = 0
        model.last_output_token_count = 0

        model.generate([])
    finally:
        instrumentor.uninstrument()

    spans = list(exporter.get_finished_spans())
    llm_spans = [
        span
        for span in spans
        if span.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert len(llm_spans) == 1
    assert llm_spans[0].name == "OpenAIModel.generate"
    assert llm_spans[0].attributes["llm.token_count.prompt"] == 11
    assert llm_spans[0].attributes["llm.token_count.completion"] == 7
    assert llm_spans[0].attributes["llm.token_count.total"] == 18
    assert llm_spans[0].parent is None
    assert OpenAIModel.generate is fake_generate
