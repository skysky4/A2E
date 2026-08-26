"""Lifecycle management for optional model compatibility middleware."""

from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

from .facade import GatewayServerConfig, create_gateway_server
from .glm_compat import ProxyConfig, create_server
from .profile import ResolvedModel


@dataclass
class ModelRuntime:
    resolved: ResolvedModel
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    def start(self) -> ResolvedModel:
        middleware = self.resolved.profile.middleware
        gateway = self.resolved.profile.gateway
        if not middleware and gateway is None:
            return self.resolved
        if not self.resolved.base_url:
            feature = "gateway" if gateway is not None else middleware[0]
            raise ValueError(f"{feature} requires a configured upstream base URL")
        if gateway is not None:
            self._server = create_gateway_server(
                host="127.0.0.1",
                port=0,
                config=GatewayServerConfig(
                    upstream_base_url=self.resolved.base_url,
                    upstream_api_key=self.resolved.api_key.get_secret_value(),
                    models=frozenset({self.resolved.profile.model}),
                    interfaces=frozenset(item.value for item in gateway.interfaces),
                    normalize_openai_tool_calls="glm_tool_call_compat" in middleware,
                ),
            )
        elif "glm_tool_call_compat" in middleware:
            self._server = create_server(
                host="127.0.0.1",
                port=0,
                config=ProxyConfig(
                    upstream_base_url=self.resolved.base_url,
                    models=frozenset({self.resolved.profile.model}),
                ),
            )
        else:  # guarded by ModelProfile validation
            raise ValueError(f"unsupported middleware: {middleware[0]}")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        port = int(self._server.server_address[1])
        health_url = f"http://127.0.0.1:{port}/healthz"
        try:
            with urllib.request.urlopen(health_url, timeout=3) as response:
                if response.status != 200:
                    raise RuntimeError(f"middleware health check returned {response.status}")
        except Exception:
            self.close()
            raise
        return self.resolved.model_copy(update={"base_url": f"http://127.0.0.1:{port}/v1"})

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None

    def metrics(self) -> dict[str, int]:
        """Return a secret-free snapshot for Campaign runtime reporting."""
        if self._server is None:
            return {
                "requests_total": 0,
                "stream_requests": 0,
                "inflight_requests": 0,
                "inflight_high_water": 0,
                "upstream_http_errors": 0,
                "upstream_retries": 0,
            }
        metrics = getattr(self._server, "metrics", None) or getattr(
            self._server.RequestHandlerClass, "metrics", None
        )
        return metrics.snapshot() if metrics is not None else {}

    def __enter__(self) -> ResolvedModel:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
