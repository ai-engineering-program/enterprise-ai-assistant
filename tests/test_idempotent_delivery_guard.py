"""
Tests for IdempotentDeliveryGuard / IdempotencyKeyStore / compute_idempotency_key.

Covers:
- Idempotency key derivation (event_id vs. content-hash fallback)
- Atomic claim-before-work semantics (no TOCTOU window)
- TTL expiry allowing legitimate re-claim after a bounded window
- Concurrent redelivery of the same event applied exactly once
- Composition with (not replacement of) EventDrivenIngestionHandler
"""

import threading

import pytest

from app.ingestion.event_handler import EventDrivenIngestionHandler
from app.ingestion.idempotent_delivery_guard import (
    IdempotencyKeyStore,
    IdempotentDeliveryGuard,
    compute_idempotency_key,
)


def make_event(event_id=None, doc_id="doc-a", content="v1", event_type="updated"):
    event = {"event_type": event_type, "doc_id": doc_id, "content": content}
    if event_id is not None:
        event["event_id"] = event_id
    return event


@pytest.mark.unit
class TestComputeIdempotencyKey:
    def test_uses_event_id_when_present(self):
        key = compute_idempotency_key(make_event(event_id="evt-1"))
        assert key == "evt:evt-1"

    def test_falls_back_to_content_hash_when_event_id_missing(self):
        key = compute_idempotency_key(make_event(doc_id="doc-x", content="hello"))
        assert key.startswith("content:")
        assert len(key) == len("content:") + 64  # sha256 hex digest length

    def test_content_hash_is_deterministic(self):
        k1 = compute_idempotency_key(make_event(doc_id="doc-x", content="hello"))
        k2 = compute_idempotency_key(make_event(doc_id="doc-x", content="hello"))
        assert k1 == k2

    def test_content_hash_differs_for_different_content(self):
        k1 = compute_idempotency_key(make_event(doc_id="doc-x", content="hello"))
        k2 = compute_idempotency_key(make_event(doc_id="doc-x", content="world"))
        assert k1 != k2

    def test_content_hash_differs_for_different_doc_id(self):
        k1 = compute_idempotency_key(make_event(doc_id="doc-x", content="same"))
        k2 = compute_idempotency_key(make_event(doc_id="doc-y", content="same"))
        assert k1 != k2


@pytest.mark.unit
class TestIdempotencyKeyStore:
    def test_first_claim_succeeds(self):
        store = IdempotencyKeyStore(ttl_seconds=60.0)
        assert store.try_claim("evt:1") is True

    def test_second_claim_of_same_key_fails(self):
        store = IdempotencyKeyStore(ttl_seconds=60.0)
        store.try_claim("evt:1")
        assert store.try_claim("evt:1") is False

    def test_different_keys_do_not_interfere(self):
        store = IdempotencyKeyStore(ttl_seconds=60.0)
        assert store.try_claim("evt:1") is True
        assert store.try_claim("evt:2") is True

    def test_expired_claim_can_be_reclaimed(self):
        fake_time = [0.0]
        store = IdempotencyKeyStore(ttl_seconds=10.0, clock=lambda: fake_time[0])

        assert store.try_claim("evt:1") is True

        fake_time[0] = 5.0
        assert store.try_claim("evt:1") is False  # TTL ещё не истёк

        fake_time[0] = 11.0
        assert store.try_claim("evt:1") is True  # TTL истёк, повтор claim разрешён

    def test_size_reflects_claimed_keys(self):
        store = IdempotencyKeyStore(ttl_seconds=60.0)
        store.try_claim("evt:1")
        store.try_claim("evt:2")
        store.try_claim("evt:1")  # duplicate claim attempt, does not add a new key
        assert store.size() == 2

    def test_concurrent_claims_of_same_key_only_one_wins(self):
        store = IdempotencyKeyStore(ttl_seconds=60.0)
        results = []
        barrier = threading.Barrier(4)

        def attempt():
            barrier.wait()
            results.append(store.try_claim("evt:race"))

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 3


@pytest.mark.unit
class TestIdempotentDeliveryGuard:
    def test_first_delivery_is_applied(self):
        guard = IdempotentDeliveryGuard()
        result = guard.handle_delivery(
            make_event(event_id="evt-1", event_type="created")
        )
        assert result == "applied"

    def test_applied_delivery_updates_index(self):
        guard = IdempotentDeliveryGuard()
        guard.handle_delivery(
            make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        )
        assert guard._handler.get_index_state()["doc-a"] == "v1"

    def test_redelivered_event_id_is_duplicate(self):
        guard = IdempotentDeliveryGuard()
        event = make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        first = guard.handle_delivery(event)
        second = guard.handle_delivery(event)
        assert first == "applied"
        assert second == "duplicate"

    def test_duplicate_does_not_touch_handler(self):
        guard = IdempotentDeliveryGuard()
        event = make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        guard.handle_delivery(event)
        guard.handle_delivery(event)
        assert guard._handler.get_processed_count() == 1

    def test_redelivery_without_event_id_deduped_by_content(self):
        guard = IdempotentDeliveryGuard()
        event = make_event(doc_id="doc-b", content="same body", event_type="created")
        first = guard.handle_delivery(dict(event))
        second = guard.handle_delivery(dict(event))  # новая доставка, тот же контент
        assert first == "applied"
        assert second == "duplicate"

    def test_accepts_injected_handler(self):
        handler = EventDrivenIngestionHandler()
        guard = IdempotentDeliveryGuard(handler=handler)
        guard.handle_delivery(
            make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        )
        assert handler.get_index_state()["doc-a"] == "v1"

    def test_does_not_subclass_or_replace_handler_logic(self):
        guard = IdempotentDeliveryGuard()
        assert isinstance(guard._handler, EventDrivenIngestionHandler)

    def test_stats_track_applied_and_duplicate(self):
        guard = IdempotentDeliveryGuard()
        guard.handle_delivery(
            make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        )
        guard.handle_delivery(
            make_event(event_id="evt-1", doc_id="doc-a", content="v1", event_type="created")
        )
        guard.handle_delivery(
            make_event(event_id="evt-2", doc_id="doc-b", content="v1", event_type="created")
        )
        stats = guard.stats()
        assert stats["applied_count"] == 2
        assert stats["duplicate_count"] == 1
        assert stats["tracked_keys"] == 2

    def test_concurrent_redelivery_applies_exactly_once(self):
        """
        Regression test for the "НеваСофт" race: two workers receiving the
        same redelivered event at (almost) the same instant must not both
        pass the idempotency check before either has claimed the key.
        """
        guard = IdempotentDeliveryGuard()
        event = make_event(event_id="evt-race", doc_id="doc-a", content="v1", event_type="created")
        results = []
        barrier = threading.Barrier(4)

        def attempt():
            barrier.wait()
            results.append(guard.handle_delivery(dict(event)))

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count("applied") == 1
        assert results.count("duplicate") == 3
        assert guard._handler.get_processed_count() == 1


@pytest.mark.integration
class TestIdempotentDeliveryGuardIntegration:
    """
    Интеграционные тесты: в проде IdempotencyKeyStore должен быть заменён
    на клиент Redis (SETNX + PX) или аналогичное durable-хранилище, общее
    для всех реплик consumer pool. Тесты ниже проверяют именно такой
    сценарий — поэтому требуют запущенного Redis и пропускаются по
    умолчанию.
    """

    def test_with_real_redis_backed_store(self):
        pytest.skip("Требует запущенного Redis — запускать вручную")

    def test_survives_process_restart_with_shared_store(self):
        pytest.skip(
            "Требует запущенного Redis, разделяемого между процессами — "
            "запускать вручную"
        )
