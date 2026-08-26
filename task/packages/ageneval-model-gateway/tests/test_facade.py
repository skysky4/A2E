from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from ageneval.model.gateway import (
    GatewayServerConfig,
    GatewayServerMetrics,
    anthropic_request_to_openai,
    anthropic_sse_body,
    build_chat_completions_url,
    create_gateway_server,
    openai_response_to_anthropic,
)


def test_gateway_metrics_track_real_inflight_high_water() -> None:
    metrics = GatewayServerMetrics()
    metrics.begin_inflight()
    metrics.begin_inflight()
    metrics.record_request(stream=True)
    assert metrics.snapshot()["inflight_requests"] == 2
    assert metrics.snapshot()["inflight_high_water"] == 2
    metrics.end_inflight()
    metrics.end_inflight()
    assert metrics.snapshot()["inflight_requests"] == 0


def test_anthropic_request_converts_system_tools_and_tool_results() -> None:
    converted = anthropic_request_to_openai(
        {
            "model": "glm-5.3",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Use the terminal."}],
            "tools": [
                {
                    "name": "terminal",
                    "description": "Run a command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "List files"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "terminal",
                            "input": {"command": "ls"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "README.md",
                        }
                    ],
                },
            ],
        }
    )
    assert converted["messages"][0] == {
        "role": "system",
        "content": "Use the terminal.",
    }
    assistant = converted["messages"][2]
    assert assistant["content"] == "Checking."
    assert assistant["tool_calls"][0]["function"] == {
        "name": "terminal",
        "arguments": '{"command":"ls"}',
    }
    assert converted["messages"][3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "README.md",
    }
    assert converted["tools"][0]["function"]["parameters"]["type"] == "object"
    assert converted["max_tokens"] == 1024


def test_openai_tool_response_becomes_anthropic_tool_use() -> None:
    converted = openai_response_to_anthropic(
        {
            "id": "chatcmpl_1",
            "model": "glm-5.3",
            "choices": [
                {
                    "finish_reason": "tool_calls",
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
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
    )
    assert converted["stop_reason"] == "tool_use"
    assert converted["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "terminal",
            "input": {"command": "pwd"},
        }
    ]
    assert converted["usage"] == {"input_tokens": 12, "output_tokens": 7}


def test_anthropic_sse_contains_messages_events() -> None:
    body = anthropic_sse_body(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "gpt-5.6-sol",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    ).decode()
    assert "event: message_start" in body
    assert '"type":"text_delta","text":"done"' in body
    assert '"stop_reason":"end_turn"' in body
    assert body.endswith('event: message_stop\ndata: {"type":"message_stop"}\n\n')


def test_chat_completions_url_handles_v1_suffix() -> None:
    assert (
        build_chat_completions_url("https://models.test/openai/v1")
        == "https://models.test/openai/v1/chat/completions"
    )
    assert (
        build_chat_completions_url("https://models.test/openai")
        == "https://models.test/openai/v1/chat/completions"
    )


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    request_path = ""
    authorization = ""
    payload: dict[str, Any] | None = None

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        type(self).request_path = self.path
        type(self).authorization = self.headers.get("Authorization", "")
        length = int(self.headers.get("Content-Length") or 0)
        type(self).payload = json.loads(self.rfile.read(length))
        response = {
            "id": "chatcmpl_test",
            "model": type(self).payload["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "I will check.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeStreamingOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length))
        assert request["stream"] is True
        chunks = [
            {
                "id": "chatcmpl_stream",
                "model": request["model"],
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello "}}],
            },
            {
                "id": "chatcmpl_stream",
                "model": request["model"],
                "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
        ]
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_one_gateway_port_serves_openai_and_anthropic_protocols() -> None:
    _FakeOpenAIHandler.payload = None
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    _serve(upstream)
    proxy = create_gateway_server(
        host="127.0.0.1",
        port=0,
        config=GatewayServerConfig(
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            upstream_api_key="upstream-secret",
            models=frozenset({"glm-5.3"}),
            normalize_openai_tool_calls=True,
        ),
    )
    _serve(proxy)
    gateway_port = proxy.server_address[1]
    openai_request = urllib.request.Request(
        f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "glm-5.3",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "previous_call"}],
                    }
                ],
            }
        ).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer client-key"},
        method="POST",
    )
    with urllib.request.urlopen(openai_request, timeout=5) as response:
        openai_result = json.load(response)
    assert _FakeOpenAIHandler.request_path == "/v1/chat/completions"
    assert _FakeOpenAIHandler.authorization == "Bearer upstream-secret"
    assert _FakeOpenAIHandler.payload is not None
    assert _FakeOpenAIHandler.payload["messages"][0]["content"] == ""
    assert openai_result["choices"][0]["finish_reason"] == "tool_calls"

    request = urllib.request.Request(
        f"http://127.0.0.1:{gateway_port}/v1/messages",
        data=json.dumps(
            {
                "model": "glm-5.3",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "pwd"}],
                "tools": [
                    {
                        "name": "terminal",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": "facade-client-key",
            "anthropic-version": "2023-06-01",
        },
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

    assert _FakeOpenAIHandler.request_path == "/v1/chat/completions"
    assert _FakeOpenAIHandler.authorization == "Bearer upstream-secret"
    assert _FakeOpenAIHandler.payload is not None
    assert _FakeOpenAIHandler.payload["messages"] == [{"role": "user", "content": "pwd"}]
    assert result["type"] == "message"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][1]["input"] == {"command": "pwd"}


def test_official_openai_and_anthropic_sdks_share_one_gateway_port() -> None:
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    _serve(upstream)
    proxy = create_gateway_server(
        host="127.0.0.1",
        port=0,
        config=GatewayServerConfig(
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            upstream_api_key="upstream-secret",
            models=frozenset({"glm-5.3"}),
            normalize_openai_tool_calls=True,
        ),
    )
    _serve(proxy)
    root = f"http://127.0.0.1:{proxy.server_address[1]}"
    openai_client = openai.OpenAI(api_key="client-key", base_url=f"{root}/v1")
    anthropic_client = anthropic.Anthropic(api_key="client-key", base_url=root)
    try:
        openai_result = openai_client.chat.completions.create(
            model="glm-5.3",
            messages=[{"role": "user", "content": "pwd"}],
        )
        anthropic_result = anthropic_client.messages.create(
            model="glm-5.3",
            max_tokens=100,
            messages=[{"role": "user", "content": "pwd"}],
        )
    finally:
        openai_client.close()
        anthropic_client.close()
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert openai_result.choices[0].finish_reason == "tool_calls"
    assert anthropic_result.stop_reason == "tool_use"
    assert _FakeOpenAIHandler.authorization == "Bearer upstream-secret"


def test_official_anthropic_sdk_accepts_facade_response() -> None:
    anthropic = pytest.importorskip("anthropic")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    _serve(upstream)
    proxy = create_gateway_server(
        host="127.0.0.1",
        port=0,
        config=GatewayServerConfig(
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            upstream_api_key="upstream-secret",
            models=frozenset({"glm-5.3"}),
        ),
    )
    _serve(proxy)
    client = anthropic.Anthropic(
        api_key="facade-client-key",
        base_url=f"http://127.0.0.1:{proxy.server_address[1]}",
    )
    try:
        result = client.messages.create(
            model="glm-5.3",
            max_tokens=100,
            messages=[{"role": "user", "content": "pwd"}],
            tools=[
                {
                    "name": "terminal",
                    "description": "Run a command",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
    finally:
        client.close()
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert result.stop_reason == "tool_use"
    assert result.content[0].type == "text"
    assert result.content[1].type == "tool_use"
    assert result.content[1].input == {"command": "pwd"}


def test_official_anthropic_sdk_accepts_translated_stream() -> None:
    anthropic = pytest.importorskip("anthropic")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeStreamingOpenAIHandler)
    _serve(upstream)
    proxy = create_gateway_server(
        host="127.0.0.1",
        port=0,
        config=GatewayServerConfig(
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            upstream_api_key="upstream-secret",
            models=frozenset({"gpt-5.6-sol"}),
        ),
    )
    _serve(proxy)
    client = anthropic.Anthropic(
        api_key="facade-client-key",
        base_url=f"http://127.0.0.1:{proxy.server_address[1]}",
    )
    try:
        with client.messages.stream(
            model="gpt-5.6-sol",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            assert stream.get_final_text() == "hello world"
            final = stream.get_final_message()
    finally:
        client.close()
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 2
    assert final.usage.output_tokens == 2
