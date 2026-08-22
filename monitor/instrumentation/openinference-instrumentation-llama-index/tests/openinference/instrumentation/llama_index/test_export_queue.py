from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from openinference.instrumentation.llama_index._handler import (
    _ExportQueue,
)


class _FakeStreamingSpan:
    def __init__(self, id_: str) -> None:
        self.id_ = id_
        self.active = True
        self._last_updated_at = 0.0
        self.end_count = 0

    def end(self, exception: BaseException | None = None) -> None:
        self.active = False
        self.end_count += 1
        self.end_exception = exception


def _queue_without_thread(*, max_pending_spans: int = 1024) -> _ExportQueue:
    return _ExportQueue(
        max_pending_spans=max_pending_spans,
        start_sweeper=False,
    )


def test_silent_streams_are_not_evicted_by_elapsed_time() -> None:
    queue = _queue_without_thread()
    spans = [_FakeStreamingSpan(f"stream-{index}") for index in range(32)]
    for span in spans:
        queue.put(span)  # type: ignore[arg-type]

    swept_at = 24 * 60 * 60
    assert queue._sweep_once(queue.queue, swept_at)

    with ThreadPoolExecutor(max_workers=16) as executor:
        found = list(executor.map(queue.find, (span.id_ for span in spans)))

    assert found == spans
    assert all(span.active for span in spans)
    assert all(span.end_count == 0 for span in spans)


def test_finished_stream_is_removed_on_next_sweep() -> None:
    queue = _queue_without_thread()
    span = _FakeStreamingSpan("stream")
    queue.put(span)  # type: ignore[arg-type]
    span.end()

    assert queue._sweep_once(queue.queue, swept_at=1)

    assert queue.find(span.id_) is None
    assert not span.active
    assert span.end_count == 1


def test_pending_span_limit_bounds_abandoned_streams() -> None:
    queue = _queue_without_thread(max_pending_spans=2)
    oldest = _FakeStreamingSpan("oldest")
    queue.put(oldest)  # type: ignore[arg-type]
    queue.put(_FakeStreamingSpan("second"))  # type: ignore[arg-type]

    queue.put(_FakeStreamingSpan("third"))  # type: ignore[arg-type]

    assert queue.find("oldest") is None
    assert not oldest.active
    assert oldest.end_count == 1
    assert isinstance(oldest.end_exception, RuntimeError)


def test_close_ends_and_removes_pending_streams() -> None:
    queue = _queue_without_thread()
    spans = [_FakeStreamingSpan(f"stream-{index}") for index in range(3)]
    for span in spans:
        queue.put(span)  # type: ignore[arg-type]

    queue.close()

    assert all(queue.find(span.id_) is None for span in spans)
    assert all(not span.active for span in spans)
    assert all(span.end_count == 1 for span in spans)
    assert all(isinstance(span.end_exception, RuntimeError) for span in spans)


@pytest.mark.parametrize(
    ("max_pending_spans", "sweep_interval_seconds"),
    [(0, 0.1), (-1, 0.1), (1, 0), (1, -1)],
)
def test_export_queue_rejects_non_positive_timing_configuration(
    max_pending_spans: int,
    sweep_interval_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        _ExportQueue(
            max_pending_spans=max_pending_spans,
            sweep_interval_seconds=sweep_interval_seconds,
            start_sweeper=False,
        )
