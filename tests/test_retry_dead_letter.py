"""
Tests for retry_dead_letter: classify_exception, compute_backoff_delay,
DeadLetterQueue, RetryingProcessor.

Covers:
- Classification of transient vs. permanent failures (known and unknown
  exception types)
- Exponential backoff growth, cap at max_delay, and equal-jitter bounds
- Permanent failures move to DLQ immediately, without any sleep/backoff
- Transient failures retry with backoff up to max_attempts, then move to DLQ
- DeadLetterQueue add / list_entries (copy semantics) / requeue
"""

import pytest

from app.ingestion.retry_dead_letter import (
    DeadLetterEntry,
    DeadLetterQueue,
    PermanentIngestionError,
    RetryingProcessor,
    TransientIngestionError,
    classify_exception,
    compute_backoff_delay,
)


@pytest.mark.unit
class TestClassifyException:
    def test_timeout_error_is_transient(self):
        assert classify_exception(TimeoutError("slow provider")) == "transient"

    def test_connection_error_is_transient(self):
        assert classify_exception(ConnectionError("connection reset")) == "transient"

    def test_custom_transient_error_is_transient(self):
        assert classify_exception(TransientIngestionError("rate limited")) == "transient"

    def test_value_error_is_permanent(self):
        assert classify_exception(ValueError("bad schema")) == "permanent"

    def test_unicode_decode_error_is_permanent(self):
        exc = UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")
        assert classify_exception(exc) == "permanent"

    def test_custom_permanent_error_is_permanent(self):
        assert classify_exception(PermanentIngestionError("corrupt pdf")) == "permanent"

    def test_unknown_exception_type_defaults_to_permanent(self):
        assert classify_exception(RuntimeError("something unexpected")) == "permanent"
        assert classify_exception(KeyError("doc_id")) == "permanent"


@pytest.mark.unit
class TestComputeBackoffDelay:
    def test_grows_exponentially_without_jitter(self):
        delays = [
            compute_backoff_delay(attempt=n, base_delay=1.0, max_delay=1000.0, jitter=False)
            for n in range(1, 5)
        ]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_capped_at_max_delay_without_jitter(self):
        delay = compute_backoff_delay(attempt=10, base_delay=1.0, max_delay=16.0, jitter=False)
        assert delay == 16.0

    def test_jitter_disabled_returns_exact_capped_value(self):
        delay = compute_backoff_delay(attempt=3, base_delay=1.0, max_delay=100.0, jitter=False)
        assert delay == 4.0

    def test_jitter_with_rng_zero_returns_half_of_capped(self):
        delay = compute_backoff_delay(
            attempt=3, base_delay=1.0, max_delay=100.0, jitter=True, rng=lambda: 0.0
        )
        assert delay == pytest.approx(4.0 * 0.5)

    def test_jitter_with_rng_near_one_approaches_full_capped(self):
        delay = compute_backoff_delay(
            attempt=3, base_delay=1.0, max_delay=100.0, jitter=True, rng=lambda: 0.999999
        )
        assert delay == pytest.approx(4.0 * 0.9999995, rel=1e-4)

    def test_jitter_result_always_within_equal_jitter_bounds(self):
        for rng_value in (0.0, 0.25, 0.5, 0.75, 0.999):
            delay = compute_backoff_delay(
                attempt=4, base_delay=1.0, max_delay=100.0, jitter=True, rng=lambda v=rng_value: v
            )
            capped = 8.0
            assert capped * 0.5 <= delay <= capped

    def test_jitter_respects_max_delay_cap(self):
        delay = compute_backoff_delay(
            attempt=10, base_delay=1.0, max_delay=10.0, jitter=True, rng=lambda: 0.999999
        )
        assert delay <= 10.0


@pytest.mark.unit
class TestDeadLetterQueue:
    def make_entry(self, doc_id="doc-1"):
        return DeadLetterEntry(
            doc_id=doc_id,
            event={"doc_id": doc_id, "content": "x"},
            error_message="boom",
            category="permanent",
            attempt_count=1,
            last_attempt_at=123.0,
        )

    def test_add_increases_size(self):
        dlq = DeadLetterQueue()
        dlq.add(self.make_entry())
        assert dlq.size() == 1

    def test_list_entries_returns_copy_not_reference(self):
        dlq = DeadLetterQueue()
        dlq.add(self.make_entry())
        entries = dlq.list_entries()
        entries.append(self.make_entry(doc_id="injected"))
        assert dlq.size() == 1  # мутация внешней копии не затронула DLQ

    def test_requeue_removes_and_returns_entry(self):
        dlq = DeadLetterQueue()
        dlq.add(self.make_entry(doc_id="doc-a"))
        entry = dlq.requeue("doc-a")
        assert entry is not None
        assert entry.doc_id == "doc-a"
        assert dlq.size() == 0

    def test_requeue_missing_doc_id_returns_none(self):
        dlq = DeadLetterQueue()
        dlq.add(self.make_entry(doc_id="doc-a"))
        assert dlq.requeue("doc-does-not-exist") is None
        assert dlq.size() == 1


@pytest.mark.unit
class TestRetryingProcessor:
    def _make_processor(self, process_fn, max_attempts=5, rng=None):
        sleeps = []
        return (
            RetryingProcessor(
                process_fn=process_fn,
                max_attempts=max_attempts,
                base_delay=1.0,
                max_delay=100.0,
                sleep_fn=lambda d: sleeps.append(d),
                clock=lambda: 42.0,
                rng=rng if rng is not None else (lambda: 0.0),
            ),
            sleeps,
        )

    def test_success_on_first_attempt(self):
        def ok(event):
            return None

        processor, sleeps = self._make_processor(ok)
        result = processor.process_with_retry("doc-1", {"doc_id": "doc-1"})

        assert result == "success"
        assert processor.retry_count("doc-1") == 1
        assert sleeps == []
        assert processor.stats()["succeeded_count"] == 1
        assert processor.stats()["dead_letter_count"] == 0

    def test_transient_failure_then_success_retries_with_backoff(self):
        calls = {"n": 0}

        def flaky(event):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("provider timeout")
            return None

        processor, sleeps = self._make_processor(flaky)
        result = processor.process_with_retry("doc-1", {"doc_id": "doc-1"})

        assert result == "success"
        assert calls["n"] == 3
        assert processor.retry_count("doc-1") == 3
        assert len(sleeps) == 2  # backoff перед попыткой 2 и перед попыткой 3
        assert processor.dead_letter_queue.size() == 0

    def test_permanent_failure_goes_to_dlq_without_any_sleep(self):
        def always_broken(event):
            raise PermanentIngestionError("corrupt document")

        processor, sleeps = self._make_processor(always_broken)
        result = processor.process_with_retry("doc-1", {"doc_id": "doc-1"})

        assert result == "dead_letter"
        assert sleeps == []  # ни одной лишней попытки для гарантированного сбоя
        assert processor.dead_letter_queue.size() == 1
        entry = processor.dead_letter_queue.list_entries()[0]
        assert entry.doc_id == "doc-1"
        assert entry.category == "permanent"
        assert entry.attempt_count == 1
        assert "corrupt document" in entry.error_message

    def test_transient_failure_exhausts_max_attempts_then_dlq(self):
        def always_times_out(event):
            raise TimeoutError("provider unreachable")

        processor, sleeps = self._make_processor(always_times_out, max_attempts=4)
        result = processor.process_with_retry("doc-1", {"doc_id": "doc-1"})

        assert result == "dead_letter"
        assert processor.retry_count("doc-1") == 4
        assert len(sleeps) == 3  # backoff перед попытками 2, 3, 4 -- не перед 5-й (её нет)
        entry = processor.dead_letter_queue.list_entries()[0]
        assert entry.category == "transient"
        assert entry.attempt_count == 4

    def test_last_attempt_at_is_tracked(self):
        def ok(event):
            return None

        processor, _ = self._make_processor(ok)
        assert processor.last_attempt_at("doc-1") is None
        processor.process_with_retry("doc-1", {"doc_id": "doc-1"})
        assert processor.last_attempt_at("doc-1") == 42.0

    def test_stats_tracks_multiple_documents(self):
        def sometimes_broken(event):
            if event["doc_id"] == "doc-bad":
                raise PermanentIngestionError("bad doc")
            return None

        processor, _ = self._make_processor(sometimes_broken)
        processor.process_with_retry("doc-ok", {"doc_id": "doc-ok"})
        processor.process_with_retry("doc-bad", {"doc_id": "doc-bad"})

        stats = processor.stats()
        assert stats["succeeded_count"] == 1
        assert stats["dead_letter_count"] == 1
        assert stats["tracked_documents"] == 2


@pytest.mark.integration
class TestRetryDeadLetterIntegration:
    """
    Интеграционные тесты: в проде dead-letter queue должна жить в durable
    хранилище (Redis list, таблица БД, отдельная Kafka-тема), общем для
    всех воркеров, а не в памяти одного процесса, как DeadLetterQueue в
    этом модуле. Тесты ниже требуют такого внешнего хранилища и по
    умолчанию пропускаются.
    """

    def test_dead_letter_queue_survives_process_restart(self):
        pytest.skip(
            "Требует durable-хранилища DLQ (Redis/БД), переживающего "
            "перезапуск процесса -- запускать вручную"
        )

    def test_real_embedding_provider_timeout_classified_as_transient(self):
        pytest.skip("Требует реального embedding-провайдера -- запускать вручную")
