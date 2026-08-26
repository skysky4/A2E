"""Multi-protocol facade for OpenAI-compatible chat-completions APIs.

One managed loopback port exposes native OpenAI Chat Completions and translated
Anthropic Messages routes for the same upstream model. It translates request,
response, tool-use, error, and SSE event shapes without logging request bodies
or credentials.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .glm_compat import (
    is_chat_completions_path,
    normalize_request_payload,
    normalize_response_payload,
    normalize_sse_body,
    repair_tool_arguments,
)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_ANTHROPIC_ONLY_HEADERS = {
    "anthropic-beta",
    "anthropic-version",
    "x-api-key",
}


@dataclass(frozen=True)
class GatewayServerConfig:
    upstream_base_url: str
    upstream_api_key: str = field(repr=False)
    models: frozenset[str] = frozenset()
    interfaces: frozenset[str] = frozenset({"openai_chat_completions", "anthropic_messages"})
    normalize_openai_tool_calls: bool = False
    timeout_seconds: float = 900.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5


@dataclass
class GatewayServerMetrics:
    requests_total: int = 0
    stream_requests: int = 0
    inflight_requests: int = 0
    inflight_high_water: int = 0
    upstream_http_errors: int = 0
    upstream_retries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, *, stream: bool) -> None:
        with self._lock:
            self.requests_total += 1
            if stream:
                self.stream_requests += 1

    def begin_inflight(self) -> None:
        with self._lock:
            self.inflight_requests += 1
            self.inflight_high_water = max(
                self.inflight_high_water, self.inflight_requests
            )

    def end_inflight(self) -> None:
        with self._lock:
            if self.inflight_requests <= 0:
                raise RuntimeError("gateway inflight request accounting underflow")
            self.inflight_requests -= 1

    def record_retry(self) -> None:
        with self._lock:
            self.upstream_retries += 1

    def record_upstream_error(self) -> None:
        with self._lock:
            self.upstream_http_errors += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "stream_requests": self.stream_requests,
                "inflight_requests": self.inflight_requests,
                "inflight_high_water": self.inflight_high_water,
                "upstream_http_errors": self.upstream_http_errors,
                "upstream_retries": self.upstream_retries,
            }


def is_messages_path(path: str) -> bool:
    return urllib.parse.urlsplit(path).path.rstrip("/").endswith("/messages")


def build_chat_completions_url(upstream_base_url: str) -> str:
    parsed = urllib.parse.urlsplit(upstream_base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    path += "/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("thinking") or block.get("reasoning_content")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_content(content)
        return text if text else json.dumps(content, ensure_ascii=False, default=str)
    return json.dumps(content, ensure_ascii=False, default=str)


def _convert_anthropic_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content", "")
    if role == "assistant":
        if isinstance(content, str):
            return [{"role": "assistant", "content": content}]
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or "tool_call"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or "tool"),
                            "arguments": json.dumps(
                                block.get("input") or {},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
            else:
                text = _text_content([block])
                if text:
                    text_parts.append(text)
        converted: dict[str, Any] = {
            "role": "assistant",
            # Some OpenAI-compatible providers reject null alongside tool calls.
            "content": "\n".join(text_parts),
        }
        if tool_calls:
            converted["tool_calls"] = tool_calls
        return [converted]

    if role != "user":
        return [{"role": str(role or "user"), "content": _text_content(content)}]
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    converted_messages: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush_text() -> None:
        if pending_text:
            converted_messages.append({"role": "user", "content": "\n".join(pending_text)})
            pending_text.clear()

    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            flush_text()
            converted_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or "tool_call"),
                    "content": _tool_result_content(block.get("content", "")),
                }
            )
            continue
        text = _text_content([block])
        if text:
            pending_text.append(text)
    flush_text()
    return converted_messages or [{"role": "user", "content": ""}]


def anthropic_request_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate one Anthropic Messages request into Chat Completions."""
    model = payload.get("model")
    if not model:
        raise ValueError("Anthropic request is missing model")
    messages: list[dict[str, Any]] = []
    system = _text_content(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Anthropic request messages must be a list")
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Anthropic request messages must contain objects")
        messages.extend(_convert_anthropic_message(message))

    result: dict[str, Any] = {"model": str(model), "messages": messages}
    mappings = {
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "stop_sequences": "stop",
        "stream": "stream",
    }
    for source, target in mappings.items():
        if source in payload and payload[source] is not None:
            result[target] = payload[source]

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "tool"),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
            if isinstance(tool, dict)
        ]
    choice = payload.get("tool_choice")
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "auto":
            result["tool_choice"] = "auto"
        elif choice_type == "any":
            result["tool_choice"] = "required"
        elif choice_type == "none":
            result["tool_choice"] = "none"
        elif choice_type == "tool" and choice.get("name"):
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": str(choice["name"])},
            }
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        result["user"] = str(metadata["user_id"])
    return result


def _parse_tool_input(value: Any) -> dict[str, Any]:
    repaired, _, valid = repair_tool_arguments(value)
    if not valid:
        return {}
    if isinstance(repaired, dict):
        return repaired
    try:
        parsed = json.loads(repaired)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def openai_response_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate one non-streaming Chat Completions response."""
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if not text:
        text = message.get("reasoning_content")
    if isinstance(text, list):
        text = _text_content(text)
    if text is not None and str(text):
        content.append({"type": "text", "text": str(text)})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            function = function if isinstance(function, dict) else {}
            content.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or f"tool_call_{position}"),
                    "name": str(function.get("name") or "tool"),
                    "input": _parse_tool_input(function.get("arguments", "{}")),
                }
            )
    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choice.get("finish_reason")
    stop_reason = {
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
        "stop": "end_turn",
        None: None,
    }.get(finish_reason, "end_turn")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {
        "id": str(payload.get("id") or "msg_openai_compat"),
        "type": "message",
        "role": "assistant",
        "model": str(payload.get("model") or "unknown"),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _openai_sse_to_response(body: bytes) -> dict[str, Any]:
    normalized, _ = normalize_sse_body(body)
    response: dict[str, Any] = {"choices": [{"message": {"role": "assistant"}}]}
    message = response["choices"][0]["message"]
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    for event in normalized.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        for key in ("id", "model", "usage"):
            if chunk.get(key) is not None:
                response[key] = chunk[key]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason") is not None:
            response["choices"][0]["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if delta.get("content"):
            text_parts.append(str(delta["content"]))
        if delta.get("reasoning_content"):
            reasoning_parts.append(str(delta["reasoning_content"]))
        for position, call in enumerate(delta.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            index = int(call.get("index", position))
            target = tool_calls.setdefault(index, {"id": None, "type": "function", "function": {}})
            if call.get("id"):
                target["id"] = call["id"]
            function = call.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    target["function"]["name"] = function["name"]
                if "arguments" in function:
                    target["function"]["arguments"] = str(
                        target["function"].get("arguments") or ""
                    ) + str(function.get("arguments") or "")
    if text_parts:
        message["content"] = "".join(text_parts)
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return response


def anthropic_sse_body(payload: dict[str, Any]) -> bytes:
    """Encode a complete Anthropic response as valid Messages SSE events."""
    message = dict(payload)
    blocks = list(message.pop("content", []))
    stop_reason = message.pop("stop_reason", None)
    stop_sequence = message.pop("stop_sequence", None)
    usage = message.pop("usage", {})
    message["content"] = []
    message["stop_reason"] = None
    message["stop_sequence"] = None
    message["usage"] = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": message})
    ]
    for index, block in enumerate(blocks):
        if block.get("type") == "tool_use":
            start_block = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    block.get("input") or {}, ensure_ascii=False, separators=(",", ":")
                ),
            }
        else:
            start_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": str(block.get("text") or "")}
        events.extend(
            [
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": start_block},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": index}),
            ]
        )
    events.extend(
        [
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    return "".join(
        f"event: {name}\ndata: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
        for name, event in events
    ).encode("utf-8")


def translate_openai_error(body: bytes, status: int) -> bytes:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or f"upstream returned HTTP {status}")
    else:
        message = str(error or f"upstream returned HTTP {status}")
    error_type = (
        "authentication_error"
        if status in {401, 403}
        else "rate_limit_error"
        if status == 429
        else "invalid_request_error"
        if status == 400
        else "api_error"
    )
    return json.dumps(
        {"type": "error", "error": {"type": error_type, "message": message}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class _GatewayServerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    config: GatewayServerConfig
    metrics: GatewayServerMetrics

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[model-gateway] {self.address_string()} {format % args}\n")

    def do_GET(self) -> None:
        route = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if route == "/healthz":
            self._send_json(200, {"status": "ok", "models": sorted(self.config.models)})
            return
        if route == "/metrics":
            self._send_json(200, self.metrics.snapshot())
            return
        if route.endswith("/models") and "openai_chat_completions" in self.config.interfaces:
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "owned_by": "a2e-gateway"}
                        for model in sorted(self.config.models)
                    ],
                },
            )
            return
        self._send_json(
            404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}}
        )

    def do_POST(self) -> None:
        self.metrics.begin_inflight()
        try:
            if is_messages_path(self.path):
                if "anthropic_messages" not in self.config.interfaces:
                    self._send_not_found()
                    return
                self._proxy_anthropic_messages()
                return
            if is_chat_completions_path(self.path):
                if "openai_chat_completions" not in self.config.interfaces:
                    self._send_not_found()
                    return
                self._proxy_openai_chat_completions()
                return
            self._send_not_found()
        finally:
            self.metrics.end_inflight()

    def _proxy_anthropic_messages(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            incoming = json.loads(body)
            if not isinstance(incoming, dict):
                raise ValueError("request body must be an object")
            if self.config.models and str(incoming.get("model")) not in self.config.models:
                raise ValueError(f"model {incoming.get('model')!r} is not served by this runtime")
            outgoing = anthropic_request_to_openai(incoming)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(
                400,
                {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            )
            return

        stream = bool(incoming.get("stream"))
        self.metrics.record_request(stream=stream)
        request_body = json.dumps(outgoing, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower()
            not in _HOP_BY_HOP_HEADERS
            | _ANTHROPIC_ONLY_HEADERS
            | {"host", "content-length", "accept-encoding", "authorization"}
        }
        headers["Accept-Encoding"] = "identity"
        headers["Authorization"] = f"Bearer {self.config.upstream_api_key}"
        request = urllib.request.Request(
            build_chat_completions_url(self.config.upstream_base_url),
            data=request_body,
            headers=headers,
            method="POST",
        )
        response = None
        upstream_error = False
        for attempt in range(self.config.max_retries + 1):
            try:
                response = urllib.request.urlopen(request, timeout=self.config.timeout_seconds)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.config.max_retries:
                    response = exc
                    upstream_error = True
                    break
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                self.metrics.record_retry()
                time.sleep(min(30.0, max(delay, self.config.retry_backoff_seconds * (2**attempt))))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.config.max_retries:
                    self._send_json(
                        502,
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"compat proxy upstream error: {exc}",
                            },
                        },
                    )
                    return
                self.metrics.record_retry()
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        assert response is not None
        with response:
            status = response.status
            response_headers = response.headers
            response_body = response.read()
        if upstream_error:
            self.metrics.record_upstream_error()
            response_body = translate_openai_error(response_body, status)
            content_type = "application/json"
        else:
            try:
                if stream:
                    openai_payload = _openai_sse_to_response(response_body)
                    anthropic_payload = openai_response_to_anthropic(openai_payload)
                    response_body = anthropic_sse_body(anthropic_payload)
                    content_type = "text/event-stream"
                else:
                    openai_payload = json.loads(response_body)
                    if not isinstance(openai_payload, dict):
                        raise ValueError("upstream response body must be an object")
                    anthropic_payload = openai_response_to_anthropic(openai_payload)
                    response_body = json.dumps(
                        anthropic_payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    content_type = "application/json"
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                status = 502
                response_body = translate_openai_error(
                    json.dumps(
                        {"error": {"message": f"invalid upstream response: {exc}"}}
                    ).encode(),
                    status,
                )
                content_type = "application/json"
        self.send_response(status)
        for name, value in response_headers.items():
            if name.lower() in _HOP_BY_HOP_HEADERS | {
                "content-length",
                "content-encoding",
                "content-type",
                "etag",
            }:
                continue
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response_body)

    def _proxy_openai_chat_completions(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request_body = self.rfile.read(length) if length else b""
        eligible = False
        parsed: Any = None
        if request_body:
            try:
                parsed = json.loads(request_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                model = str(parsed.get("model") or "")
                if self.config.models and model not in self.config.models:
                    self._send_json(
                        400,
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "message": f"model {model!r} is not served by this runtime",
                            }
                        },
                    )
                    return
                if self.config.normalize_openai_tool_calls:
                    parsed, eligible, _ = normalize_request_payload(
                        parsed, target_models=self.config.models
                    )
                    request_body = json.dumps(
                        parsed, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
        stream = bool(isinstance(parsed, dict) and parsed.get("stream"))
        self.metrics.record_request(stream=stream)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower()
            not in _HOP_BY_HOP_HEADERS
            | _ANTHROPIC_ONLY_HEADERS
            | {"host", "content-length", "accept-encoding", "authorization"}
        }
        headers["Accept-Encoding"] = "identity"
        headers["Authorization"] = f"Bearer {self.config.upstream_api_key}"
        request = urllib.request.Request(
            build_chat_completions_url(self.config.upstream_base_url),
            data=request_body,
            headers=headers,
            method="POST",
        )
        response = None
        upstream_error = False
        for attempt in range(self.config.max_retries + 1):
            try:
                response = urllib.request.urlopen(request, timeout=self.config.timeout_seconds)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.config.max_retries:
                    response = exc
                    upstream_error = True
                    break
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                self.metrics.record_retry()
                time.sleep(min(30.0, max(delay, self.config.retry_backoff_seconds * (2**attempt))))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.config.max_retries:
                    self._send_json(
                        502,
                        {
                            "error": {
                                "type": "api_error",
                                "message": f"model gateway upstream error: {exc}",
                            }
                        },
                    )
                    return
                self.metrics.record_retry()
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        assert response is not None
        with response:
            status = response.status
            response_headers = response.headers
            response_body = response.read()
        if upstream_error:
            self.metrics.record_upstream_error()
        elif eligible and 200 <= status < 300:
            content_type = response_headers.get("Content-Type", "")
            if "text/event-stream" in content_type.lower():
                response_body, _ = normalize_sse_body(response_body)
            else:
                try:
                    response_payload = json.loads(response_body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    response_payload = None
                if response_payload is not None:
                    response_payload, _ = normalize_response_payload(response_payload)
                    response_body = json.dumps(
                        response_payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
        self.send_response(status)
        for name, value in response_headers.items():
            if name.lower() in _HOP_BY_HOP_HEADERS | {
                "content-length",
                "content-encoding",
                "etag",
            }:
                continue
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response_body)

    def _send_not_found(self) -> None:
        self._send_json(
            404,
            {"type": "error", "error": {"type": "not_found_error", "message": "not found"}},
        )

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def create_gateway_server(
    *,
    host: str,
    port: int,
    config: GatewayServerConfig,
    metrics: GatewayServerMetrics | None = None,
) -> ThreadingHTTPServer:
    handler = type(
        "GatewayServerHandler",
        (_GatewayServerHandler,),
        {"config": config, "metrics": metrics or GatewayServerMetrics()},
    )
    return ThreadingHTTPServer((host, port), handler)
