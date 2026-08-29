"""Gateway-compat patches for OpenAI-compatible chat APIs.

Some gateways (Claude-shaped validators in front of /v1/chat/completions)
reject assistant messages with ``content: null`` even when ``tool_calls``
are present. Reasoning models often emit exactly that. Coerce null content
to an empty string on every Completions.create call.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)
_INSTALLED = False


def _placeholder_content(msg: Any) -> str:
    """Non-empty placeholder.

    The official OpenAI SDK drops ``content=""`` on assistant+tool_calls
    messages (serializes as JSON null). This gateway then 422s. A single
    space survives serialization and is ignored by the model.
    """
    return " "


def sanitize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    out = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is None or content == "" or content == []:
                msg = dict(msg)
                msg["content"] = _placeholder_content(msg)
            out.append(msg)
            continue
        content = getattr(msg, "content", "MISSING")
        if content is None or content == "" or content == []:
            try:
                msg.content = _placeholder_content(msg)
            except Exception:  # noqa: BLE001
                pass
        out.append(msg)
    return out


def sanitize_tool_arguments(raw: Any) -> Any:
    """Repair concatenated JSON like ``{}{"a":1}`` or ``{}{}`` that some models emit."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return "{}"
    # Repeated empty / concatenated objects: {}{}  {}{"k":1}
    while text.startswith("{}{") or text == "{}{}":
        text = text[2:]
    if text.startswith("{}") and len(text) > 2:
        rest = text[2:].strip()
        if rest.startswith("{"):
            text = rest
    if not text:
        text = "{}"
    return text


def _sanitize_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        payload = dict(payload)
        payload["messages"] = sanitize_messages(msgs)
    return payload


def _rewrite_httpx_request(request: Any) -> Any:
    try:
        import json

        import httpx
    except Exception:  # noqa: BLE001
        return request
    url_s = str(getattr(request, "url", ""))
    if "chat/completions" not in url_s:
        return request
    raw = getattr(request, "content", None)
    if not raw:
        return request
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return request
    fixed = _sanitize_payload(payload)
    if fixed is payload:
        return request
    body = json.dumps(fixed).encode("utf-8")
    headers = dict(request.headers)
    headers["content-length"] = str(len(body))
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=headers,
        content=body,
        extensions=request.extensions,
    )


def _install_httpx_patch() -> None:
    """Last-resort: coerce null message content on the wire.

    The OpenAI SDK calls ``httpx.Client.send``, not ``request``.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return

    if not getattr(httpx.Client.send, "_a2e_compat", False):
        orig_send = httpx.Client.send

        def send(self, request, *args, **kwargs):  # noqa: ANN001
            import time

            request = _rewrite_httpx_request(request)
            last = None
            for attempt in range(6):
                last = orig_send(self, request, *args, **kwargs)
                if getattr(last, "status_code", 0) != 503:
                    return last
                time.sleep(min(30.0, 1.5 * (2**attempt)))
            return last

        send._a2e_compat = True  # type: ignore[attr-defined]
        httpx.Client.send = send  # type: ignore[method-assign]

    if getattr(httpx.AsyncClient.send, "_a2e_compat", False):
        return

    orig_asend = httpx.AsyncClient.send

    async def asend(self, request, *args, **kwargs):  # noqa: ANN001
        import asyncio

        request = _rewrite_httpx_request(request)
        last = None
        for attempt in range(6):
            last = await orig_asend(self, request, *args, **kwargs)
            if getattr(last, "status_code", 0) != 503:
                return last
            await asyncio.sleep(min(30.0, 1.5 * (2**attempt)))
        return last

    asend._a2e_compat = True  # type: ignore[attr-defined]
    httpx.AsyncClient.send = asend  # type: ignore[method-assign]


def _install_json_loads_patch() -> None:
    """Some models emit ``{}{"k":1}``; SDKs then ``json.loads`` and crash."""
    import json

    if getattr(json.loads, "_a2e_compat", False):
        return
    orig = json.loads

    def loads(s, *args, **kwargs):  # noqa: ANN001
        if isinstance(s, (str, bytes, bytearray)):
            text = s.decode() if isinstance(s, (bytes, bytearray)) else s
            stripped = text.strip()
            if stripped.startswith("{}{") or stripped == "{}{}":
                text = sanitize_tool_arguments(text)
                s = text.encode() if isinstance(s, (bytes, bytearray)) else text
        return orig(s, *args, **kwargs)

    loads._a2e_compat = True  # type: ignore[attr-defined]
    json.loads = loads  # type: ignore[method-assign]


def install_openai_compat() -> None:
    """Idempotent monkeypatch of the OpenAI SDK chat.completions entrypoints."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception as exc:  # noqa: BLE001
        _log.warning("openai_compat: openai SDK not patchable (%s)", exc)
    else:

        def _is_503(exc: BaseException) -> bool:
            text = str(exc).lower()
            return "503" in text or "no available accounts" in text

        def _wrap_sync(orig):
            def create(self, *args, **kwargs):
                import time

                if "messages" in kwargs:
                    kwargs["messages"] = sanitize_messages(kwargs["messages"])
                last = None
                for attempt in range(6):
                    try:
                        return orig(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        if not _is_503(exc) or attempt == 5:
                            raise
                        time.sleep(min(30.0, 1.5 * (2**attempt)))
                raise last  # pragma: no cover

            return create

        def _wrap_async(orig):
            async def create(self, *args, **kwargs):
                import asyncio

                if "messages" in kwargs:
                    kwargs["messages"] = sanitize_messages(kwargs["messages"])
                last = None
                for attempt in range(6):
                    try:
                        return await orig(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        if not _is_503(exc) or attempt == 5:
                            raise
                        await asyncio.sleep(min(30.0, 1.5 * (2**attempt)))
                raise last  # pragma: no cover

            return create

        Completions.create = _wrap_sync(Completions.create)  # type: ignore[method-assign]
        AsyncCompletions.create = _wrap_async(AsyncCompletions.create)  # type: ignore[method-assign]
    _install_httpx_patch()
    _install_json_loads_patch()
    try:
        import agno.utils.functions as agno_fn

        if not getattr(agno_fn.get_function_call, "_a2e_compat", False):
            _orig_gfc = agno_fn.get_function_call

            def get_function_call(name, arguments=None, call_id=None, functions=None):  # noqa: ANN001
                return _orig_gfc(
                    name,
                    arguments=sanitize_tool_arguments(arguments),
                    call_id=call_id,
                    functions=functions,
                )

            get_function_call._a2e_compat = True  # type: ignore[attr-defined]
            agno_fn.get_function_call = get_function_call  # type: ignore[method-assign]
    except Exception as exc:  # noqa: BLE001
        _log.warning("openai_compat: agno get_function_call not patchable (%s)", exc)
    try:
        import agents.tool as agents_tool

        if not getattr(agents_tool._parse_function_tool_json_input, "_a2e_compat", False):
            _orig_parse = agents_tool._parse_function_tool_json_input

            def _parse(*, tool_name, input_json):  # noqa: ANN001
                return _orig_parse(
                    tool_name=tool_name,
                    input_json=sanitize_tool_arguments(input_json),
                )

            _parse._a2e_compat = True  # type: ignore[attr-defined]
            agents_tool._parse_function_tool_json_input = _parse  # type: ignore[method-assign]
    except Exception as exc:  # noqa: BLE001
        _log.warning("openai_compat: agents tool parser not patchable (%s)", exc)
    _INSTALLED = True
    _log.info("openai_compat: installed null-content sanitizer")
