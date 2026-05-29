"""
Event-driven ingestion handler for enterprise-ai-assistant.

Processes document change events (created/updated/deleted) and maintains
the vector index in near real-time. Implements idempotency via event_id
deduplication to handle at-least-once delivery from webhooks and message
brokers.

Lesson 6.1 — Event-driven ingestion для AI
Course: Архитектура AI-систем для production
"""

from typing import Optional


class EventDrivenIngestionHandler:
    """
    Processes document change events with idempotency guarantees.

    Supports event types: created, updated, deleted.
    Duplicate events (same event_id) are safely ignored.

    In production this class would:
    - Call an embedding model to generate vectors for created/updated docs
    - Write/delete vectors in a vector store (Qdrant, Pinecone, Weaviate)
    - Persist processed_ids to Redis with TTL for cross-process deduplication
    """

    def __init__(self):
        # Set of processed event_id values for deduplication
        self._processed_ids: set = set()
        # In-memory index: {doc_id: content}
        # In production: replaced by vector store client calls
        self._index: dict = {}
        self._processed_count: int = 0

    def handle_event(self, event: dict) -> bool:
        """
        Handle a document change event.

        Args:
            event: dict with keys:
                event_id   — unique event identifier (str)
                event_type — "created" | "updated" | "deleted"
                doc_id     — document identifier (str)
                content    — document text content (str)

        Returns:
            True if the event was processed, False if it was a duplicate.
        """
        event_id = event["event_id"]
        event_type = event["event_type"]
        doc_id = event["doc_id"]

        # Idempotency check: skip already-processed events
        if event_id in self._processed_ids:
            return False

        # Apply the change to the index
        if event_type in ("created", "updated"):
            # Upsert: handles out-of-order delivery gracefully
            self._index[doc_id] = event.get("content", "")
        elif event_type == "deleted":
            # Safe delete: no error if doc_id is already absent
            self._index.pop(doc_id, None)

        # Mark event as processed only after successful index update
        self._processed_ids.add(event_id)
        self._processed_count += 1
        return True

    def get_index_state(self) -> dict:
        """
        Return a snapshot of the current index.

        Returns:
            dict mapping doc_id to content for all indexed documents.
        """
        return self._index.copy()

    def get_processed_count(self) -> int:
        """
        Return the count of uniquely processed events.

        Duplicate events (same event_id) are not counted.
        """
        return self._processed_count
