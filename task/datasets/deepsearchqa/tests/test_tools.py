from __future__ import annotations

from ageneval.task.datasets.deepsearchqa import tools


def _clear_proxy_env(monkeypatch) -> None:
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "A2E_WEB_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_web_tools_default_to_direct_egress(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    configured: list[dict[str, str]] = []
    monkeypatch.setattr(
        tools.urllib.request,
        "ProxyHandler",
        lambda proxies: configured.append(proxies) or object(),
    )
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda _handler: object())

    assert tools._proxy_url() is None
    tools._opener()
    assert configured == [{}]


def test_web_tools_honor_explicit_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("A2E_WEB_PROXY", "http://proxy.example:3128")

    assert tools._proxy_url() == "http://proxy.example:3128"
