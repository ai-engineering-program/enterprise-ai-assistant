"""
Tests for EventDrivenIngestionHandler.

Covers:
- created / updated / deleted event processing
- Idempotency (duplicate event_id ignored)
- Out-of-order delivery (updated before created → upsert)
- Safe delete of non-existent doc
- processed_count accuracy
"""

import pytest
from app.ingestion.event_handler import EventDrivenIngestionHandler


@pytest.fixture
def handler():
    return EventDrivenIngestionHandler()


def make_event(event_id, event_type, doc_id, content=""):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "doc_id": doc_id,
        "content": content,
    }


class TestCreatedEvent:
    def test_adds_document_to_index(self, handler):
        result = handler.handle_event(
            make_event("evt-001", "created", "doc-policy", "Travel policy v1")
        )
        assert result is True
        assert handler.get_index_state()["doc-policy"] == "Travel policy v1"

    def test_increments_processed_count(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-a", "A"))
        assert handler.get_processed_count() == 1


class TestUpdatedEvent:
    def test_replaces_content(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-a", "v1"))
        handler.handle_event(make_event("evt-002", "updated", "doc-a", "v2"))
        assert handler.get_index_state()["doc-a"] == "v2"

    def test_upserts_when_doc_absent(self, handler):
        """Out-of-order: updated arrives before created."""
        handler.handle_event(make_event("evt-oo1", "updated", "doc-new", "content"))
        assert "doc-new" in handler.get_index_state()


class TestDeletedEvent:
    def test_removes_document(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-x", "data"))
        handler.handle_event(make_event("evt-002", "deleted", "doc-x", ""))
        assert "doc-x" not in handler.get_index_state()

    def test_safe_delete_nonexistent(self, handler):
        result = handler.handle_event(
            make_event("evt-ghost", "deleted", "doc-nonexistent", "")
        )
        assert result is True  # event processed, no error raised


class TestIdempotency:
    def test_duplicate_returns_false(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-a", "original"))
        result = handler.handle_event(
            make_event("evt-001", "updated", "doc-a", "should not apply")
        )
        assert result is False

    def test_duplicate_does_not_change_index(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-a", "original"))
        handler.handle_event(make_event("evt-001", "updated", "doc-a", "overwrite"))
        assert handler.get_index_state()["doc-a"] == "original"

    def test_duplicate_does_not_increment_count(self, handler):
        handler.handle_event(make_event("evt-001", "created", "doc-a", "v1"))
        handler.handle_event(make_event("evt-001", "created", "doc-a", "v1"))
        assert handler.get_processed_count() == 1


class TestMultipleDocuments:
    def test_independent_documents(self, handler):
        handler.handle_event(make_event("e1", "created", "doc-1", "A"))
        handler.handle_event(make_event("e2", "created", "doc-2", "B"))
        handler.handle_event(make_event("e3", "created", "doc-3", "C"))
        handler.handle_event(make_event("e4", "deleted", "doc-2", ""))
        state = handler.get_index_state()
        assert state.get("doc-1") == "A"
        assert "doc-2" not in state
        assert state.get("doc-3") == "C"
        assert handler.get_processed_count() == 4
