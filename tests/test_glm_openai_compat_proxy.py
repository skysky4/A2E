from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from ageneval.model.gateway.glm_compat import (
    ProxyConfig,
    ProxyMetrics,
    build_upstream_url,
    create_server,
    normalize_request_payload,
    normalize_response_payload,
    normalize_sse_body,
    repair_tool_arguments,
)


def test_repair_known_leading_empty_object() -> None:
    value = '{}{"command":"pwd; ls -la"}'
    repaired, changed, valid = repair_tool_arguments(value)
    assert repaired == '{"command":"pwd; ls -la"}'
    assert changed is True
    assert valid is True


@pytest.mark.parametrize(
    "value",
    [
        '{"command":"pwd"}',
        '{} {"command":"pwd"} {"extra":true}',
        '{"command":"pwd"}{"extra":true}',
        "not-json",
        "{}{}",
        "[]",
    ],
)
def test_repair_does_not_guess_ambiguous_or_unrelated_values(value: str) -> None:
    repaired, changed, _ = repair_tool_arguments(value)
    assert repaired == value
    assert changed is False


def test_request_normalization_is_model_scoped() -> None:
    payload = {
        "model": "glm-5.3",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        ],
    }
    normalized, eligible, counts = normalize_request_payload(
        payload, target_models=frozenset({"glm-5.3"})
    )
    assert eligible is True
    assert normalized["messages"][1]["content"] == ""
    assert payload["messages"][1]["content"] is None
    assert counts.request_content_null_normalized == 1

    untouched, eligible, counts = normalize_request_payload(
        {**payload, "model": "another-model"}, target_models=frozenset({"glm-5.3"})
    )
    assert eligible is False
    assert untouched["messages"][1]["content"] is None
    assert counts.request_content_null_normalized == 0


def test_nonstreaming_response_normalization() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{}{"command":"pwd"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    normalized, counts = normalize_response_payload(payload)
    arguments = normalized["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert arguments == '{"command":"pwd"}'
    assert counts.response_arguments_repaired == 1
    assert counts.response_invalid_arguments == 0


def test_streaming_response_repairs_arguments_split_across_chunks() -> None:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"command":"pwd"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    body = (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    ).encode()
    normalized, counts = normalize_sse_body(body)
    argument_fragments: list[str] = []
    for event in normalized.decode().split("\n\n"):
        if not event.startswith("data: ") or event == "data: [DONE]":
            continue
        parsed = json.loads(event[len("data: ") :])
        for choice in parsed.get("choices", []):
            for tool_call in choice.get("delta", {}).get("tool_calls", []):
                function = tool_call.get("function") or {}
                if "arguments" in function:
                    argument_fragments.append(function["arguments"])
    assert "".join(argument_fragments) == '{"command":"pwd"}'
    assert counts.response_arguments_repaired == 1


def test_build_upstream_url_avoids_duplicate_v1() -> None:
    assert (
        build_upstream_url("https://example.test/openai/v1", "/v1/chat/completions?x=1")
        == "https://example.test/openai/v1/chat/completions?x=1"
    )


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    received_payload: dict[str, Any] | None = None

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        type(self).received_payload = json.loads(self.rfile.read(length))
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": '{}{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_end_to_end_proxy_normalizes_request_and_response() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    _serve_in_thread(upstream)
    upstream_port = upstream.server_address[1]
    metrics = ProxyMetrics()
    proxy = create_server(
        host="127.0.0.1",
        port=0,
        config=ProxyConfig(upstream_base_url=f"http://127.0.0.1:{upstream_port}/v1"),
        metrics=metrics,
    )
    _serve_in_thread(proxy)
    proxy_port = proxy.server_address[1]
    request_payload = {
        "model": "glm-5.3",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "type": "function"}],
            }
        ],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=json.dumps(request_payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert _FakeUpstreamHandler.received_payload is not None
    assert _FakeUpstreamHandler.received_payload["messages"][0]["content"] == ""
    arguments = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"command":"pwd"}'
    assert metrics.snapshot()["request_content_null_normalized"] == 1
    assert metrics.snapshot()["response_arguments_repaired"] == 1


class _RetryUpstreamHandler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        type(self).calls += 1
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if type(self).calls == 1:
            body = b'{"error":"busy"}'
            self.send_response(429)
            self.send_header("Retry-After", "0")
        else:
            body = b'{"choices":[]}'
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_proxy_retries_transient_upstream_status() -> None:
    _RetryUpstreamHandler.calls = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _RetryUpstreamHandler)
    _serve_in_thread(upstream)
    metrics = ProxyMetrics()
    proxy = create_server(
        host="127.0.0.1",
        port=0,
        config=ProxyConfig(
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            retry_backoff_seconds=0,
        ),
        metrics=metrics,
    )
    _serve_in_thread(proxy)
    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
        data=b'{"model":"glm-5.3","messages":[]}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
    assert _RetryUpstreamHandler.calls == 2
    assert metrics.snapshot()["upstream_retries"] == 1
