"""
InMemoryEventBus — in-process event bus for testing and local development.

Provides a lightweight publish/subscribe mechanism that mirrors the interface
of production event buses (Kafka, Redis Streams) without requiring external
infrastructure. Used extensively in the test suite to verify consumer logic
in isolation.

Lesson 6.4 — Полная event-driven AI-архитектура
Course: Архитектура AI-систем для production
"""

import uuid
from typing import Callable


class InMemoryEventBus:
    """
    Synchronous in-memory event bus with subscription management and history.

    Semantics:
    - Publish/subscribe by event type (string).
    - Multiple handlers per event type, called in subscription order (FIFO).
    - Every published event is appended to an immutable history log.
    - Handlers receive the original event dict (not a copy); consumers should
      not mutate the event.

    Usage in tests::

        bus = InMemoryEventBus()
        received = []
        sub_id = bus.subscribe("document.indexed", lambda e: received.append(e))

        bus.publish({"type": "document.indexed", "doc_id": "d1"})
        assert len(received) == 1

        bus.unsubscribe(sub_id)
    """

    def __init__(self) -> None:
        # {event_type: {subscription_id: handler}}
        self._subscriptions: dict[str, dict[str, Callable]] = {}
        # Ordered list of all published events
        self._history: list[dict] = []

    def subscribe(self, event_type: str, handler: Callable) -> str:
        """
        Register a handler for events of the given type.

        Args:
            event_type: Event type string, e.g. "document.parsed".
            handler:    Callable accepting a single dict argument (the event).

        Returns:
            subscription_id — a UUID string used to unsubscribe later.
        """
        subscription_id = str(uuid.uuid4())
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = {}
        self._subscriptions[event_type][subscription_id] = handler
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription by its ID.

        Args:
            subscription_id: The ID returned by subscribe().

        Returns:
            True if the subscription existed and was removed; False otherwise.
        """
        for handlers in self._subscriptions.values():
            if subscription_id in handlers:
                del handlers[subscription_id]
                return True
        return False

    def publish(self, event: dict) -> int:
        """
        Publish an event to all registered handlers.

        The event must contain a "type" key. Handlers are invoked synchronously
        in the order they were registered. The event is always appended to
        history regardless of whether there are active subscribers.

        Args:
            event: dict with at minimum {"type": "<event_type>", ...}.

        Returns:
            Number of handlers that were invoked.
        """
        event_type: str = event["type"]
        self._history.append(event)
        handlers = list(self._subscriptions.get(event_type, {}).values())
        for handler in handlers:
            handler(event)
        return len(handlers)

    def get_history(self, event_type: str = None) -> list[dict]:
        """
        Return published events in publication order.

        Args:
            event_type: If given, return only events of this type.

        Returns:
            A new list (copy of the relevant slice of history).
        """
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.get("type") == event_type]

    def get_subscriber_count(self, event_type: str) -> int:
        """
        Return the number of active subscribers for the given event type.

        Args:
            event_type: Event type string.

        Returns:
            Integer >= 0.
        """
        return len(self._subscriptions.get(event_type, {}))
