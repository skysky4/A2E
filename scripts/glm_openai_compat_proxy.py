#!/usr/bin/env python3
"""Backward-compatible, stdlib-only CLI for the model gateway GLM middleware."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL_PATH = (
    Path(__file__).resolve().parents[1]
    / "task/packages/ageneval-model-gateway/src/ageneval/model/gateway/glm_compat.py"
)
_SPEC = importlib.util.spec_from_file_location("a2e_glm_compat_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load GLM compatibility implementation: {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

ProxyConfig = _IMPL.ProxyConfig
ProxyMetrics = _IMPL.ProxyMetrics
build_upstream_url = _IMPL.build_upstream_url
create_server = _IMPL.create_server
normalize_request_payload = _IMPL.normalize_request_payload
normalize_response_payload = _IMPL.normalize_response_payload
normalize_sse_body = _IMPL.normalize_sse_body
repair_tool_arguments = _IMPL.repair_tool_arguments
main = _IMPL.main

if __name__ == "__main__":
    raise SystemExit(main())
