#!/usr/bin/env python3
"""Local compatibility proxy for GLM's OpenAI chat-completions endpoint.

The proxy deliberately applies only two narrow workarounds:

* assistant messages that contain ``tool_calls`` get ``content: ""`` when
  their content is null; and
* function-call arguments with one or more leading empty JSON objects are
  reduced to the single following non-empty JSON object.

Everything else, including the Authorization header, is forwarded unchanged.
The default listener is loopback-only and the proxy never logs request bodies
or credentials.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


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


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str
    models: frozenset[str] = frozenset({"glm-5.3"})
    timeout_seconds: float = 900.0


@dataclass
class MutationCounts:
    request_content_null_normalized: int = 0
    response_arguments_repaired: int = 0
    response_invalid_arguments: int = 0

    def add(self, other: "MutationCounts") -> None:
        self.request_content_null_normalized += other.request_content_null_normalized
        self.response_arguments_repaired += other.response_arguments_repaired
        self.response_invalid_arguments += other.response_invalid_arguments


@dataclass
class ProxyMetrics:
    requests_total: int = 0
    eligible_requests: int = 0
    stream_responses: int = 0
    nonstream_responses: int = 0
    upstream_http_errors: int = 0
    mutations: MutationCounts = field(default_factory=MutationCounts)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, *, eligible: bool, mutations: MutationCounts) -> None:
        with self._lock:
            self.requests_total += 1
            if eligible:
                self.eligible_requests += 1
            self.mutations.add(mutations)

    def record_response(
        self,
        *,
        stream: bool,
        mutations: MutationCounts,
        upstream_error: bool,
    ) -> None:
        with self._lock:
            if stream:
                self.stream_responses += 1
            else:
                self.nonstream_responses += 1
            if upstream_error:
                self.upstream_http_errors += 1
            self.mutations.add(mutations)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "eligible_requests": self.eligible_requests,
                "stream_responses": self.stream_responses,
                "nonstream_responses": self.nonstream_responses,
                "upstream_http_errors": self.upstream_http_errors,
                "request_content_null_normalized": (
                    self.mutations.request_content_null_normalized
                ),
                "response_arguments_repaired": self.mutations.response_arguments_repaired,
                "response_invalid_arguments": self.mutations.response_invalid_arguments,
            }


def is_chat_completions_path(path: str) -> bool:
    return urllib.parse.urlsplit(path).path.rstrip("/").endswith("/chat/completions")


def normalize_request_payload(
    payload: dict[str, Any], *, target_models: frozenset[str]
) -> tuple[dict[str, Any], bool, MutationCounts]:
    """Return a copied, normalized payload and whether the model is targeted."""
    counts = MutationCounts()
    eligible = str(payload.get("model") or "") in target_models
    if not eligible:
        return payload, False, counts

    normalized = copy.deepcopy(payload)
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized, True, counts

    for message in messages:
        if not isinstance(message, dict):
            continue
        if (
            message.get("role") == "assistant"
            and message.get("content") is None
            and message.get("tool_calls")
        ):
            message["content"] = ""
            counts.request_content_null_normalized += 1
    return normalized, True, counts


def repair_tool_arguments(value: Any) -> tuple[Any, bool, bool]:
    """Conservatively repair the observed ``{}{...}`` GLM argument shape.

    Returns ``(value, changed, valid_object)``. Arbitrary malformed JSON and
    concatenated non-empty objects are never guessed at or rewritten.
    """
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")), True, True
    if not isinstance(value, str):
        return value, False, False

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None
    else:
        return value, False, isinstance(decoded, dict)

    decoder = json.JSONDecoder()
    position = 0
    parsed: list[tuple[Any, int, int]] = []
    length = len(value)
    try:
        while position < length:
            while position < length and value[position].isspace():
                position += 1
            if position >= length:
                break
            item, end = decoder.raw_decode(value, position)
            parsed.append((item, position, end))
            position = end
    except json.JSONDecodeError:
        return value, False, False

    if position != length or len(parsed) < 2:
        return value, False, False
    prefix = [item for item, _, _ in parsed[:-1]]
    final, final_start, final_end = parsed[-1]
    if not prefix or any(item != {} for item in prefix):
        return value, False, False
    if not isinstance(final, dict) or final == {}:
        return value, False, False
    repaired = value[final_start:final_end]
    return repaired, True, True


def _normalize_function_container(
    container: Any, counts: MutationCounts
) -> None:
    if not isinstance(container, dict) or "arguments" not in container:
        return
    repaired, changed, valid = repair_tool_arguments(container.get("arguments"))
    if changed:
        container["arguments"] = repaired
        counts.response_arguments_repaired += 1
    elif not valid:
        counts.response_invalid_arguments += 1


def normalize_response_payload(payload: Any) -> tuple[Any, MutationCounts]:
    """Normalize non-streaming Chat Completions tool arguments in place."""
    counts = MutationCounts()
    if not isinstance(payload, dict):
        return payload, counts
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload, counts
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for message_key in ("message", "delta"):
            message = choice.get(message_key)
            if not isinstance(message, dict):
                continue
            _normalize_function_container(message.get("function_call"), counts)
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    _normalize_function_container(tool_call.get("function"), counts)
    return payload, counts


def normalize_sse_body(body: bytes) -> tuple[bytes, MutationCounts]:
    """Buffer and normalize arguments split across Chat Completions SSE chunks."""
    counts = MutationCounts()
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        counts.response_invalid_arguments += 1
        return body, counts

    newline = "\r\n" if "\r\n" in text else "\n"
    canonical = text.replace("\r\n", "\n")
    raw_events = canonical.split("\n\n")
    events: list[dict[str, Any] | None] = []
    argument_locations: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for raw_event in raw_events:
        data_lines = [line[5:].lstrip() for line in raw_event.split("\n") if line.startswith("data:")]
        data = "\n".join(data_lines)
        if not data or data == "[DONE]":
            events.append(None)
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            events.append(None)
            continue
        events.append(event if isinstance(event, dict) else None)
        if not isinstance(event, dict):
            continue
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice_position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            choice_index = choice.get("index", choice_position)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_position, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                tool_index = tool_call.get("index", tool_position)
                function = tool_call.get("function")
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    argument_locations.setdefault(
                        (str(choice_index), str(tool_index)), []
                    ).append(function)

    for locations in argument_locations.values():
        combined = "".join(str(location.get("arguments") or "") for location in locations)
        repaired, changed, valid = repair_tool_arguments(combined)
        if changed:
            for location in locations:
                location["arguments"] = ""
            locations[0]["arguments"] = repaired
            counts.response_arguments_repaired += 1
        elif not valid:
            counts.response_invalid_arguments += 1

    rebuilt: list[str] = []
    for index, raw_event in enumerate(raw_events):
        event = events[index]
        if event is None:
            rebuilt.append(raw_event)
            continue
        non_data = [line for line in raw_event.split("\n") if not line.startswith("data:")]
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        rebuilt.append("\n".join([*non_data, f"data: {serialized}"]).lstrip("\n"))

    # Preserve the upstream event terminator; OpenAI clients accept either LF style.
    normalized = "\n\n".join(rebuilt)
    if newline == "\r\n":
        normalized = normalized.replace("\n", "\r\n")
    return normalized.encode("utf-8"), counts


def build_upstream_url(upstream_base_url: str, incoming_path: str) -> str:
    base = urllib.parse.urlsplit(upstream_base_url)
    incoming = urllib.parse.urlsplit(incoming_path)
    base_path = base.path.rstrip("/")
    incoming_route = incoming.path
    if base_path.endswith("/v1") and incoming_route.startswith("/v1/"):
        incoming_route = incoming_route[len("/v1") :]
    joined_path = f"{base_path}/{incoming_route.lstrip('/')}"
    query = incoming.query or base.query
    return urllib.parse.urlunsplit((base.scheme, base.netloc, joined_path, query, ""))


class _CompatProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    config: ProxyConfig
    metrics: ProxyMetrics

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[glm-compat] {self.address_string()} {format % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        route = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if route == "/healthz":
            self._send_json(200, {"status": "ok", "models": sorted(self.config.models)})
            return
        if route == "/metrics":
            self._send_json(200, self.metrics.snapshot())
            return
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _proxy_request(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request_body = self.rfile.read(length) if length else b""
        chat_request = is_chat_completions_path(self.path)
        eligible = False
        request_counts = MutationCounts()
        request_payload: dict[str, Any] | None = None

        if chat_request and request_body:
            try:
                parsed = json.loads(request_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                request_payload, eligible, request_counts = normalize_request_payload(
                    parsed, target_models=self.config.models
                )
                if eligible:
                    request_body = json.dumps(
                        request_payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
        self.metrics.record_request(eligible=eligible, mutations=request_counts)

        upstream_url = build_upstream_url(self.config.upstream_base_url, self.path)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower()
            not in _HOP_BY_HOP_HEADERS | {"host", "content-length", "accept-encoding"}
        }
        headers["Accept-Encoding"] = "identity"
        upstream_request = urllib.request.Request(
            upstream_url,
            data=request_body if self.command not in {"GET", "HEAD"} else None,
            headers=headers,
            method=self.command,
        )

        upstream_error = False
        try:
            upstream_response = urllib.request.urlopen(
                upstream_request, timeout=self.config.timeout_seconds
            )
        except urllib.error.HTTPError as exc:
            upstream_response = exc
            upstream_error = True
        except (urllib.error.URLError, TimeoutError) as exc:
            self._send_json(502, {"error": {"message": f"compat proxy upstream error: {exc}"}})
            return

        with upstream_response:
            status = upstream_response.status
            response_headers = upstream_response.headers
            response_body = upstream_response.read()

        content_type = response_headers.get("Content-Type", "")
        is_stream = "text/event-stream" in content_type.lower()
        response_counts = MutationCounts()
        if eligible and 200 <= status < 300:
            if is_stream:
                response_body, response_counts = normalize_sse_body(response_body)
            else:
                try:
                    response_payload = json.loads(response_body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    response_payload = None
                if response_payload is not None:
                    response_payload, response_counts = normalize_response_payload(response_payload)
                    response_body = json.dumps(
                        response_payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
        self.metrics.record_response(
            stream=is_stream,
            mutations=response_counts,
            upstream_error=upstream_error,
        )

        self.send_response(status)
        for name, value in response_headers.items():
            lower = name.lower()
            if lower in _HOP_BY_HOP_HEADERS | {"content-length", "content-encoding", "etag"}:
                continue
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response_body)


def create_server(
    *, host: str, port: int, config: ProxyConfig, metrics: ProxyMetrics | None = None
) -> ThreadingHTTPServer:
    handler = type(
        "CompatProxyHandler",
        (_CompatProxyHandler,),
        {"config": config, "metrics": metrics or ProxyMetrics()},
    )
    return ThreadingHTTPServer((host, port), handler)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-base-url",
        default=os.environ.get("GLM_COMPAT_UPSTREAM_BASE_URL"),
        help="real OpenAI-compatible API base URL (or GLM_COMPAT_UPSTREAM_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="model to normalize; repeatable (default: glm-5.3)",
    )
    parser.add_argument("--host", default=os.environ.get("GLM_COMPAT_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GLM_COMPAT_PORT", "8011"))
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("GLM_COMPAT_TIMEOUT", "900")),
        help="upstream timeout in seconds",
    )
    args = parser.parse_args(argv)
    if not args.upstream_base_url:
        parser.error("--upstream-base-url or GLM_COMPAT_UPSTREAM_BASE_URL is required")
    parsed = urllib.parse.urlsplit(args.upstream_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("upstream base URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        parser.error("put credentials in request headers, not in the upstream URL")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    models = frozenset(args.models or ["glm-5.3"])
    config = ProxyConfig(
        upstream_base_url=args.upstream_base_url,
        models=models,
        timeout_seconds=args.timeout,
    )
    server = create_server(host=args.host, port=args.port, config=config)
    host, port = server.server_address[:2]
    print(
        f"GLM compatibility proxy listening on http://{host}:{port}/v1 "
        f"for models {', '.join(sorted(models))}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
