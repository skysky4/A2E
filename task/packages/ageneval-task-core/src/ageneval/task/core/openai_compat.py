"""Gateway-compat patches for OpenAI-compatible chat APIs.

Some gateways (Claude-shaped validators in front of /v1/chat/completions)
reject assistant messages with ``content: null`` even when ``tool_calls``
are present. Reasoning models often emit exactly that. Coerce null content
to an empty string on every Completions.create call.
"""

from __future__ import annotations

import json
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


def _needs_completion_tokens(model: Any) -> bool:
    name = str(model or "").lower()
    return name.startswith("gpt-5") or "gpt-5.6" in name


def rewrite_token_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """gpt-5.x on this gateway rejects ``max_tokens``; use ``max_completion_tokens``."""
    if "max_tokens" not in kwargs or "max_completion_tokens" in kwargs:
        return kwargs
    if not _needs_completion_tokens(kwargs.get("model")):
        return kwargs
    out = dict(kwargs)
    out["max_completion_tokens"] = out.pop("max_tokens")
    return out


def coerce_json_object(raw: Any) -> dict[str, Any]:
    """Accept the JSON shapes models emit; always return one object.

    CrewAI's ``repair_json`` turns concatenated ``{...}{...}`` into a list.
    The official tool schema is a single object, so take the first dict.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return dict(item)
        return {}
    if not isinstance(raw, str):
        return {}
    text = sanitize_tool_arguments(raw.strip())
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:  # noqa: BLE001
            return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return dict(item)
    return {}


def is_anthropic_messages_url(url: Any) -> bool:
    """True for a gateway ``/v1/messages`` that is not api.anthropic.com."""
    text = str(url or "")
    if "/v1/messages" not in text:
        return False
    if "api.anthropic.com" in text:
        return False
    return True


def anthropic_to_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate Anthropic Messages body → OpenAI chat.completions body."""
    model = payload.get("model")
    out: dict[str, Any] = {"model": model}
    if "max_tokens" in payload:
        out["max_tokens"] = payload["max_tokens"]
    elif "max_completion_tokens" in payload:
        out["max_completion_tokens"] = payload["max_completion_tokens"]
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        bits = []
        for block in system:
            if isinstance(block, dict) and block.get("text"):
                bits.append(str(block["text"]))
            elif isinstance(block, str):
                bits.append(block)
        if bits:
            messages.append({"role": "system", "content": "\n".join(bits)})
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        messages.extend(_anthropic_message_to_openai(msg))
    out["messages"] = messages
    tools = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "")
        if not name:
            continue
        params = tool.get("input_schema") or tool.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or name),
                    "parameters": params,
                },
            }
        )
    if tools:
        out["tools"] = tools
    return rewrite_token_kwargs(out)


def openai_to_anthropic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate OpenAI chat.completions JSON → Anthropic Messages JSON."""
    import uuid

    choice = {}
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        choice = choices[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []
    text = msg.get("content")
    if isinstance(text, str) and text.strip():
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("text"):
                content_blocks.append({"type": "text", "text": str(part["text"])})
            elif isinstance(part, str) and part.strip():
                content_blocks.append({"type": "text", "text": part})
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            parsed = raw_args
        else:
            try:
                parsed = json.loads(sanitize_tool_arguments(str(raw_args)))
            except Exception:  # noqa: BLE001
                parsed = coerce_json_object(raw_args)
            if not isinstance(parsed, dict):
                parsed = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": str(fn.get("name") or "tool"),
                "input": parsed,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    usage = payload.get("usage") or {}
    finish = str(choice.get("finish_reason") or "")
    stop = "tool_use" if tool_calls or finish == "tool_calls" else "end_turn"
    if finish == "length":
        stop = "max_tokens"
    return {
        "id": str(payload.get("id") or f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": payload.get("model") or "",
        "content": content_blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            ),
        },
    }


def _anthropic_message_to_openai(msg: dict[str, Any]) -> list[dict[str, Any]]:
    role = str(msg.get("role") or "user")
    content = msg.get("content")
    if isinstance(content, str):
        return [{"role": role if role in {"user", "assistant", "system"} else "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": str(content or "")}]
    out: list[dict[str, Any]] = []
    text_bits: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type") or "")
        if kind == "text" or (not kind and block.get("text")):
            if block.get("text"):
                text_bits.append(str(block["text"]))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or "tool"),
                        "arguments": json.dumps(block.get("input") or {}, default=str),
                    },
                }
            )
        elif kind == "tool_result":
            raw = block.get("content")
            if isinstance(raw, list):
                raw = "".join(
                    str(p.get("text") or p) if isinstance(p, dict) else str(p) for p in raw
                )
            elif not isinstance(raw, str):
                raw = json.dumps(raw, default=str)
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": raw or " ",
                }
            )
    if role == "assistant" or tool_calls:
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_bits) or " ",
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        out.append(assistant)
    elif text_bits:
        out.append({"role": "user" if role != "system" else "system", "content": "\n".join(text_bits)})
    out.extend(tool_results)
    return out or [{"role": "user", "content": " "}]


def _sanitize_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    payload = rewrite_token_kwargs(payload)
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        payload = dict(payload)
        payload["messages"] = sanitize_messages(msgs)
    return payload


def _rewrite_httpx_request(request: Any) -> Any:
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return request
    url_s = str(getattr(request, "url", ""))
    raw = getattr(request, "content", None)
    if not raw:
        return request
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return request
    if is_anthropic_messages_url(url_s) and isinstance(payload, dict):
        fixed = _sanitize_payload(anthropic_to_openai_payload(payload))
        body = json.dumps(fixed).encode("utf-8")
        headers = dict(request.headers)
        headers["content-length"] = str(len(body))
        headers["content-type"] = "application/json"
        new_url = url_s.replace("/v1/messages", "/v1/chat/completions")
        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=body,
            extensions=request.extensions,
        )
    if "chat/completions" not in url_s:
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


def _openai_http_to_anthropic(response: Any, *, original_url: str) -> Any:
    """Rewrite a chat.completions HTTP body back to Anthropic Messages."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return response
    if getattr(response, "status_code", 0) != 200:
        return response
    raw = getattr(response, "content", None)
    if not raw:
        return response
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return response
    if not isinstance(payload, dict) or "choices" not in payload:
        return response
    try:
        converted = openai_to_anthropic_payload(payload)
        body = json.dumps(converted).encode("utf-8")
    except Exception:  # noqa: BLE001
        return response
    headers = dict(response.headers)
    headers["content-length"] = str(len(body))
    headers["content-type"] = "application/json"
    return httpx.Response(
        status_code=200,
        headers=headers,
        content=body,
        request=getattr(response, "request", None),
        extensions=getattr(response, "extensions", {}),
    )


def _install_httpx_patch() -> None:
    """Last-resort: coerce null message content on the wire.

    The OpenAI SDK calls ``httpx.Client.send``, not ``request``.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return

    if not getattr(httpx.Client.send, "_a2e_compat_v2", False):
        orig_send = httpx.Client.send

        def send(self, request, *args, **kwargs):  # noqa: ANN001
            import time

            bridged = is_anthropic_messages_url(getattr(request, "url", ""))
            original_url = str(getattr(request, "url", ""))
            request = _rewrite_httpx_request(request)
            last = None
            for attempt in range(6):
                last = orig_send(self, request, *args, **kwargs)
                if getattr(last, "status_code", 0) != 503:
                    if bridged and last is not None:
                        return _openai_http_to_anthropic(last, original_url=original_url)
                    return last
                time.sleep(min(30.0, 1.5 * (2**attempt)))
            if bridged and last is not None:
                return _openai_http_to_anthropic(last, original_url=original_url)
            return last

        send._a2e_compat_v2 = True  # type: ignore[attr-defined]
        httpx.Client.send = send  # type: ignore[method-assign]

    if getattr(httpx.AsyncClient.send, "_a2e_compat_v2", False):
        return

    orig_asend = httpx.AsyncClient.send

    async def asend(self, request, *args, **kwargs):  # noqa: ANN001
        import asyncio

        bridged = is_anthropic_messages_url(getattr(request, "url", ""))
        original_url = str(getattr(request, "url", ""))
        request = _rewrite_httpx_request(request)
        last = None
        for attempt in range(6):
            last = await orig_asend(self, request, *args, **kwargs)
            if getattr(last, "status_code", 0) != 503:
                if bridged and last is not None:
                    return _openai_http_to_anthropic(last, original_url=original_url)
                return last
            await asyncio.sleep(min(30.0, 1.5 * (2**attempt)))
        if bridged and last is not None:
            return _openai_http_to_anthropic(last, original_url=original_url)
        return last

    asend._a2e_compat_v2 = True  # type: ignore[attr-defined]
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

                kwargs = rewrite_token_kwargs(kwargs)
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

                kwargs = rewrite_token_kwargs(kwargs)
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
    _install_crewai_arg_patch()
    _INSTALLED = True
    _log.info("openai_compat: installed null-content sanitizer")


def _install_crewai_arg_patch() -> None:
    """Accept list/concatenated tool JSON that CrewAI otherwise rejects."""
    try:
        from crewai.tools.tool_usage import ToolUsage
    except Exception as exc:  # noqa: BLE001
        _log.warning("openai_compat: crewai tool input not patchable (%s)", exc)
        return
    if getattr(ToolUsage._validate_tool_input, "_a2e_compat", False):
        return
    orig = ToolUsage._validate_tool_input

    def _validate(self, tool_input):  # noqa: ANN001
        obj = coerce_json_object(tool_input)
        if obj:
            tool_input = json.dumps(obj)
        return orig(self, tool_input)

    _validate._a2e_compat = True  # type: ignore[attr-defined]
    ToolUsage._validate_tool_input = _validate  # type: ignore[method-assign]
    try:
        from crewai.llm import LLM

        if not getattr(LLM.supports_function_calling, "_a2e_compat", False):
            orig_fc = LLM.supports_function_calling

            def supports_function_calling(self) -> bool:  # noqa: ANN001
                name = str(getattr(self, "model", "") or "").lower()
                if (
                    name.startswith("openai/")
                    or name.startswith("gpt-")
                    or "gpt-5" in name
                ):
                    return True
                return bool(orig_fc(self))

            supports_function_calling._a2e_compat = True  # type: ignore[attr-defined]
            LLM.supports_function_calling = supports_function_calling  # type: ignore[method-assign]
    except Exception as exc:  # noqa: BLE001
        _log.warning("openai_compat: crewai FC probe not patchable (%s)", exc)
