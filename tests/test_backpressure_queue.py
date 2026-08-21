"""Tests for app/ingestion/backpressure_queue.py (Lesson 5.4).

Unit tests are pure in-memory / asyncio.Queue based and run without any
external service. Integration tests describe end-to-end wiring into
AsyncIngestionPipeline (Lesson 5.2) against a real embedding backend and
are skipped by default.

Run unit tests only:
    pytest tests/test_backpressure_queue.py -v -m unit

Run all tests:
    pytest tests/test_backpressure_queue.py -v
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.ingestion.backpressure_queue import BoundedIngestionQueue, DocumentBatcher


# ---------------------------------------------------------------------------
# DocumentBatcher
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDocumentBatcherAdd:
    def test_returns_none_below_batch_size(self):
        batcher = DocumentBatcher(batch_size=3)
        assert batcher.add("doc-1") is None
        assert batcher.add("doc-2") is None

    def test_returns_full_batch_when_reached(self):
        batcher = DocumentBatcher(batch_size=3)
        batcher.add("doc-1")
        batcher.add("doc-2")
        batch = batcher.add("doc-3")
        assert batch == ["doc-1", "doc-2", "doc-3"]

    def test_starts_new_buffer_after_full_batch(self):
        batcher = DocumentBatcher(batch_size=2)
        batcher.add("doc-1")
        first_batch = batcher.add("doc-2")
        assert first_batch == ["doc-1", "doc-2"]
        assert batcher.add("doc-3") is None
        second_batch = batcher.add("doc-4")
        assert second_batch == ["doc-3", "doc-4"]

    def test_batch_size_one_returns_batch_every_call(self):
        batcher = DocumentBatcher(batch_size=1)
        assert batcher.add("doc-1") == ["doc-1"]
        assert batcher.add("doc-2") == ["doc-2"]

    def test_returned_batch_is_not_mutated_by_later_adds(self):
        """A batch handed back by add() must be a snapshot, not a live
        reference into a buffer that keeps growing on subsequent calls."""
        batcher = DocumentBatcher(batch_size=2)
        batcher.add("doc-1")
        first_batch = batcher.add("doc-2")
        batcher.add("doc-3")
        assert first_batch == ["doc-1", "doc-2"]


@pytest.mark.unit
class TestDocumentBatcherFlush:
    def test_flush_returns_partial_buffer(self):
        batcher = DocumentBatcher(batch_size=5)
        batcher.add("doc-1")
        batcher.add("doc-2")
        remainder = batcher.flush()
        assert remainder == ["doc-1", "doc-2"]

    def test_flush_on_empty_buffer_returns_empty_list(self):
        batcher = DocumentBatcher(batch_size=5)
        assert batcher.flush() == []

    def test_flush_clears_buffer(self):
        batcher = DocumentBatcher(batch_size=5)
        batcher.add("doc-1")
        batcher.flush()
        assert batcher.flush() == []

    def test_flush_after_full_batches_only_returns_tail(self):
        batcher = DocumentBatcher(batch_size=2)
        batcher.add("doc-1")
        batcher.add("doc-2")  # full batch, buffer resets
        batcher.add("doc-3")  # partial tail
        assert batcher.flush() == ["doc-3"]


@pytest.mark.unit
class TestDocumentBatcherPendingCount:
    def test_pending_count_tracks_buffer_size(self):
        batcher = DocumentBatcher(batch_size=4)
        assert batcher.pending_count() == 0
        batcher.add("doc-1")
        assert batcher.pending_count() == 1
        batcher.add("doc-2")
        assert batcher.pending_count() == 2

    def test_pending_count_resets_after_full_batch(self):
        batcher = DocumentBatcher(batch_size=2)
        batcher.add("doc-1")
        batcher.add("doc-2")
        assert batcher.pending_count() == 0


# ---------------------------------------------------------------------------
# BoundedIngestionQueue — depth / utilization / throttle_delay
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestDepthAndUtilization:
    async def test_empty_queue_has_zero_depth_and_utilization(self):
        queue = BoundedIngestionQueue(max_size=10)
        assert queue.depth() == 0
        assert queue.utilization() == 0.0

    async def test_depth_reflects_items_put(self):
        queue = BoundedIngestionQueue(max_size=10)
        await queue.put("doc-1")
        await queue.put("doc-2")
        assert queue.depth() == 2

    async def test_utilization_is_depth_over_max_size(self):
        queue = BoundedIngestionQueue(max_size=4)
        await queue.put("doc-1")
        assert queue.utilization() == pytest.approx(0.25)


@pytest.mark.unit
class TestThrottleDelay:
    def _queue_with_depth(self, depth: int, max_size: int, **kwargs) -> BoundedIngestionQueue:
        queue = BoundedIngestionQueue(max_size=max_size, **kwargs)
        for i in range(depth):
            queue._queue.put_nowait(f"doc-{i}")
        return queue

    def test_zero_delay_below_low_watermark(self):
        queue = self._queue_with_depth(
            depth=3, max_size=10, low_watermark=0.5, high_watermark=0.9, max_delay=2.0
        )
        assert queue.throttle_delay() == 0.0

    def test_max_delay_above_high_watermark(self):
        queue = self._queue_with_depth(
            depth=9, max_size=10, low_watermark=0.5, high_watermark=0.8, max_delay=2.0
        )
        assert queue.throttle_delay() == pytest.approx(2.0)

    def test_linear_interpolation_at_midpoint(self):
        # utilization = 0.6, low=0.4, high=0.8 -> midpoint of the ramp
        queue = self._queue_with_depth(
            depth=6, max_size=10, low_watermark=0.4, high_watermark=0.8, max_delay=2.0
        )
        assert queue.throttle_delay() == pytest.approx(1.0)

    def test_delay_at_exact_low_watermark_is_zero(self):
        queue = self._queue_with_depth(
            depth=5, max_size=10, low_watermark=0.5, high_watermark=0.9, max_delay=2.0
        )
        assert queue.throttle_delay() == 0.0

    def test_delay_at_exact_high_watermark_is_max(self):
        queue = self._queue_with_depth(
            depth=9, max_size=10, low_watermark=0.5, high_watermark=0.9, max_delay=2.0
        )
        assert queue.throttle_delay() == pytest.approx(2.0)


@pytest.mark.unit
class TestConstructorValidation:
    def test_rejects_low_watermark_not_below_high(self):
        with pytest.raises(ValueError):
            BoundedIngestionQueue(max_size=10, low_watermark=0.9, high_watermark=0.5)

    def test_rejects_equal_watermarks(self):
        with pytest.raises(ValueError):
            BoundedIngestionQueue(max_size=10, low_watermark=0.5, high_watermark=0.5)


# ---------------------------------------------------------------------------
# BoundedIngestionQueue — put() / get() backpressure behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestPutGetRoundTrip:
    async def test_get_returns_put_item_in_order(self):
        queue = BoundedIngestionQueue(max_size=5)
        await queue.put("doc-1")
        await queue.put("doc-2")
        assert await queue.get() == "doc-1"
        assert await queue.get() == "doc-2"

    async def test_get_reduces_depth(self):
        queue = BoundedIngestionQueue(max_size=5)
        await queue.put("doc-1")
        await queue.get()
        assert queue.depth() == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestHardBlocking:
    """max_delay is kept tiny here on purpose: these tests isolate the HARD
    block on a full asyncio.Queue from the soft throttle_delay() sleep --
    soft throttling is covered separately in TestSoftThrottling."""

    async def test_put_blocks_when_queue_is_full(self):
        queue = BoundedIngestionQueue(
            max_size=1, low_watermark=0.99, high_watermark=1.0, max_delay=0.01
        )
        await queue.put("doc-1")  # fills the queue to max_size

        task = asyncio.create_task(queue.put("doc-2"))
        await asyncio.sleep(0.1)
        assert not task.done()  # still blocked -- queue is full

        await queue.get()  # frees up a slot
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()

    async def test_blocked_put_completes_after_slot_freed(self):
        queue = BoundedIngestionQueue(
            max_size=2, low_watermark=0.99, high_watermark=1.0, max_delay=0.01
        )
        await queue.put("doc-1")
        await queue.put("doc-2")

        task = asyncio.create_task(queue.put("doc-3"))
        await asyncio.sleep(0.1)
        assert not task.done()

        first = await queue.get()
        assert first == "doc-1"
        await asyncio.wait_for(task, timeout=1.0)
        assert queue.depth() == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestSoftThrottling:
    async def test_put_sleeps_when_above_low_watermark(self):
        queue = BoundedIngestionQueue(
            max_size=2, low_watermark=0.0, high_watermark=0.5, max_delay=0.2
        )
        # First put: utilization before put is 0.0 (<= low_watermark) -> no delay.
        start = time.monotonic()
        await queue.put("doc-1")
        elapsed_first = time.monotonic() - start

        # Second put: utilization before put is 0.5 (>= high_watermark) -> max delay.
        start = time.monotonic()
        await queue.put("doc-2")
        elapsed_second = time.monotonic() - start

        assert elapsed_first < 0.1
        assert elapsed_second >= 0.15


# ---------------------------------------------------------------------------
# Integration tests — require wiring into a real ingestion pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBackpressureIntegration:
    """Требует запущенный Qdrant и рабочий get_embedding() для проверки
    полного пути producer -> DocumentBatcher -> BoundedIngestionQueue ->
    AsyncIngestionPipeline воркеры на реалистичном объёме документов.
    Скип по умолчанию: pytest -m 'not integration'."""

    def test_large_archive_memory_stays_bounded(self):
        pytest.skip(
            "Требует запущенный Qdrant и рабочий embedding backend — "
            "прогон на реалистичном объёме документов (десятки-сотни тысяч) "
            "с измерением пиковой памяти процесса. Запускайте вручную:\n"
            "  pytest tests/test_backpressure_queue.py -m integration -v"
        )

    def test_batch_embedding_call_reduces_total_api_calls(self):
        pytest.skip(
            "Сравнение числа вызовов embedding API с DocumentBatcher и без "
            "него на одной и той же выборке документов — запускайте вручную."
        )
