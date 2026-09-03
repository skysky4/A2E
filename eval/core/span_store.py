"""Load spans from the local A2E SQLite DB when the span HTTP API is unavailable."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlparse

LOGGER = logging.getLogger(__name__)

_SPAN_COLUMNS = (
    "span_id",
    "name",
    "span_kind",
    "start_time",
    "end_time",
    "attributes",
    "llm_token_count_prompt",
    "llm_token_count_completion",
)


def sqlite_database_path() -> str | None:
    url = os.getenv("A2E_SQL_DATABASE_URL") or os.getenv("SQL_DATABASE_URL") or ""
    if not url.startswith("sqlite:"):
        return None
    parsed = urlparse(url)
    if parsed.path:
        return unquote(parsed.path)
    if parsed.netloc:
        return unquote(parsed.netloc)
    return None


def _parse_attributes(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _flatten_llm_token_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    merged = dict(attributes)
    llm = merged.get("llm")
    if not isinstance(llm, Mapping):
        return merged
    token_count = llm.get("token_count")
    if isinstance(token_count, Mapping):
        for key, value in token_count.items():
            if value is not None:
                merged[f"llm.token_count.{key}"] = value
    cost = llm.get("cost")
    if isinstance(cost, Mapping):
        for key, value in cost.items():
            if value is not None:
                merged[f"llm.cost.{key}"] = value
    return merged


def span_dict_from_sqlite_row(row: sqlite3.Row) -> dict[str, Any]:
    attributes = _flatten_llm_token_attrs(_parse_attributes(row["attributes"]))
    prompt = row["llm_token_count_prompt"]
    completion = row["llm_token_count_completion"]
    if prompt is not None and "llm.token_count.prompt" not in attributes:
        attributes["llm.token_count.prompt"] = prompt
    if completion is not None and "llm.token_count.completion" not in attributes:
        attributes["llm.token_count.completion"] = completion
    if (
        prompt is not None
        and completion is not None
        and "llm.token_count.total" not in attributes
    ):
        attributes["llm.token_count.total"] = float(prompt) + float(completion)

    span_id = str(row["span_id"] or "")
    span_kind = str(row["span_kind"] or "")
    return {
        "id": span_id,
        "span_id": span_id,
        "name": row["name"],
        "span_kind": span_kind,
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "attributes": attributes,
    }


def fetch_spans_by_trace_id(
    db_path: str,
    trace_id: str,
    *,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
              s.span_id,
              s.name,
              s.span_kind,
              s.start_time,
              s.end_time,
              s.attributes,
              s.llm_token_count_prompt,
              s.llm_token_count_completion
            FROM spans s
            JOIN traces t ON s.trace_rowid = t.rowid
            WHERE t.trace_id = ?
            ORDER BY s.start_time
            LIMIT ?
            """,
            (trace_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [span_dict_from_sqlite_row(row) for row in rows]


def fetch_spans_by_trace_ids(
    trace_ids: Sequence[str],
    *,
    limit_per_trace: int = 1000,
    db_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    path = db_path or sqlite_database_path()
    if not path or not trace_ids:
        return {}
    if not os.path.isfile(path):
        LOGGER.warning("sqlite span fallback: database not found at %s", path)
        return {}

    spans_by_trace_id: dict[str, list[dict[str, Any]]] = {}
    for trace_id in trace_ids:
        if not trace_id:
            continue
        spans_by_trace_id[str(trace_id)] = fetch_spans_by_trace_id(
            path,
            str(trace_id),
            limit=limit_per_trace,
        )
    return spans_by_trace_id
