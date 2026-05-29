"""
Tests for InMemoryEventBus.

These tests verify the publish/subscribe contract used by all event-driven
consumers in enterprise-ai-assistant. Run with: pytest tests/test_event_bus.py

Lesson 6.4 — Полная event-driven AI-архитектура
"""

import pytest
from app.event_bus.bus import InMemoryEventBus


@pytest.fixture
def bus():
    return InMemoryEventBus()


class TestSubscribePublish:
    def test_published_event_reaches_handler(self, bus):
        received = []
        bus.subscribe("doc.created", lambda e: received.append(e["id"]))
        count = bus.publish({"type": "doc.created", "id": "d1"})
        assert count == 1
        assert received == ["d1"]

    def test_multiple_subscribers_all_called(self, bus):
        log = []
        bus.subscribe("doc.indexed", lambda e: log.append("A"))
        bus.subscribe("doc.indexed", lambda e: log.append("B"))
        count = bus.publish({"type": "doc.indexed"})
        assert count == 2
        assert set(log) == {"A", "B"}

    def test_publish_without_subscribers_returns_zero(self, bus):
        count = bus.publish({"type": "unknown.event"})
        assert count == 0

    def test_different_event_types_do_not_cross_fire(self, bus):
        received = []
        bus.subscribe("doc.parsed", lambda e: received.append("parsed"))
        bus.subscribe("doc.embedded", lambda e: received.append("embedded"))
        bus.publish({"type": "doc.parsed"})
        assert received == ["parsed"]


class TestUnsubscribe:
    def test_unsubscribe_returns_true_for_existing_subscription(self, bus):
        sub_id = bus.subscribe("doc.embedded", lambda e: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_returns_false_for_unknown_id(self, bus):
        assert bus.unsubscribe("nonexistent-id") is False

    def test_double_unsubscribe_returns_false_on_second_call(self, bus):
        sub_id = bus.subscribe("doc.embedded", lambda e: None)
        bus.unsubscribe(sub_id)
        assert bus.unsubscribe(sub_id) is False

    def test_handler_not_called_after_unsubscribe(self, bus):
        called = []
        sub_id = bus.subscribe("doc.parsed", lambda e: called.append(1))
        bus.publish({"type": "doc.parsed"})
        bus.unsubscribe(sub_id)
        bus.publish({"type": "doc.parsed"})
        assert len(called) == 1


class TestHistory:
    def test_all_events_stored_in_order(self, bus):
        bus.publish({"type": "doc.created", "id": "d1"})
        bus.publish({"type": "doc.embedded", "id": "d1"})
        bus.publish({"type": "doc.created", "id": "d2"})
        assert len(bus.get_history()) == 3

    def test_events_stored_without_subscribers(self, bus):
        bus.publish({"type": "orphan.event"})
        assert len(bus.get_history()) == 1

    def test_filter_by_type(self, bus):
        bus.publish({"type": "doc.created", "id": "d1"})
        bus.publish({"type": "doc.embedded", "id": "d1"})
        bus.publish({"type": "doc.created", "id": "d2"})
        created = bus.get_history("doc.created")
        assert len(created) == 2
        assert all(e["type"] == "doc.created" for e in created)

    def test_get_history_returns_copy(self, bus):
        bus.publish({"type": "doc.created"})
        h1 = bus.get_history()
        h2 = bus.get_history()
        assert h1 is not h2

    def test_empty_filter_returns_empty_list(self, bus):
        bus.publish({"type": "doc.created"})
        result = bus.get_history("doc.indexed")
        assert result == []


class TestSubscriberCount:
    def test_zero_for_unknown_type(self, bus):
        assert bus.get_subscriber_count("doc.parsed") == 0

    def test_increments_on_subscribe(self, bus):
        bus.subscribe("doc.parsed", lambda e: None)
        bus.subscribe("doc.parsed", lambda e: None)
        assert bus.get_subscriber_count("doc.parsed") == 2

    def test_decrements_on_unsubscribe(self, bus):
        sub_id = bus.subscribe("doc.parsed", lambda e: None)
        bus.subscribe("doc.parsed", lambda e: None)
        bus.unsubscribe(sub_id)
        assert bus.get_subscriber_count("doc.parsed") == 1

    def test_unique_subscription_ids(self, bus):
        ids = [bus.subscribe("doc.created", lambda e: None) for _ in range(5)]
        assert len(set(ids)) == 5
