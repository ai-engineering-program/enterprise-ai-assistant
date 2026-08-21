import pytest

from app.ingestion.event_handler import EventDrivenIngestionHandler
from app.ingestion.stream_event_processor import StreamEvent, StreamEventProcessor


def make_event(event_id, doc_id, sequence, event_type="updated", content="v"):
    return StreamEvent(
        event_id=event_id,
        event_type=event_type,
        doc_id=doc_id,
        content=content,
        sequence=sequence,
    )


@pytest.mark.unit
class TestAppliedEvents:
    def test_first_event_is_applied(self):
        processor = StreamEventProcessor()
        result = processor.process(
            make_event("evt-1", "doc-a", sequence=1, event_type="created", content="v1")
        )
        assert result == "applied"

    def test_applied_event_updates_index(self):
        processor = StreamEventProcessor()
        processor.process(
            make_event("evt-1", "doc-a", sequence=1, event_type="created", content="v1")
        )
        state = processor._handler.get_index_state()
        assert state["doc-a"] == "v1"

    def test_increasing_sequence_is_applied(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-1", "doc-a", sequence=1, content="v1"))
        result = processor.process(make_event("evt-2", "doc-a", sequence=2, content="v2"))
        assert result == "applied"
        assert processor._handler.get_index_state()["doc-a"] == "v2"

    def test_independent_documents_tracked_separately(self):
        processor = StreamEventProcessor()
        r1 = processor.process(make_event("evt-1", "doc-a", sequence=5, content="a"))
        r2 = processor.process(make_event("evt-2", "doc-b", sequence=1, content="b"))
        assert r1 == "applied"
        assert r2 == "applied"


@pytest.mark.unit
class TestStaleEvents:
    def test_lower_sequence_is_stale(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-2", "doc-a", sequence=5, content="new"))
        result = processor.process(make_event("evt-1", "doc-a", sequence=3, content="old"))
        assert result == "stale"

    def test_stale_event_does_not_overwrite_index(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-2", "doc-a", sequence=5, content="new"))
        processor.process(make_event("evt-1", "doc-a", sequence=3, content="old"))
        assert processor._handler.get_index_state()["doc-a"] == "new"

    def test_stale_count_increments(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-2", "doc-a", sequence=5, content="new"))
        processor.process(make_event("evt-1", "doc-a", sequence=3, content="old"))
        processor.process(make_event("evt-3", "doc-a", sequence=1, content="older"))
        assert processor.stats()["stale_count"] == 2

    def test_stale_check_is_per_document(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-1", "doc-a", sequence=10, content="a"))
        # doc-b never seen before -> sequence=1 must NOT be considered stale
        result = processor.process(make_event("evt-2", "doc-b", sequence=1, content="b"))
        assert result == "applied"


@pytest.mark.unit
class TestDuplicateEvents:
    def test_same_event_id_redelivered_is_duplicate(self):
        processor = StreamEventProcessor()
        event = make_event("evt-1", "doc-a", sequence=5, event_type="created", content="v1")
        first = processor.process(event)
        second = processor.process(event)
        assert first == "applied"
        assert second == "duplicate"

    def test_duplicate_does_not_change_content(self):
        processor = StreamEventProcessor()
        event = make_event("evt-1", "doc-a", sequence=5, event_type="created", content="v1")
        processor.process(event)
        # Same event_id redelivered, even with a (hypothetically) different
        # payload -- handler dedup by event_id must win, content stays v1.
        replay = StreamEvent(
            event_id="evt-1",
            event_type="created",
            doc_id="doc-a",
            content="tampered",
            sequence=5,
        )
        processor.process(replay)
        assert processor._handler.get_index_state()["doc-a"] == "v1"

    def test_duplicate_does_not_increment_stale_count(self):
        processor = StreamEventProcessor()
        event = make_event("evt-1", "doc-a", sequence=5, content="v1")
        processor.process(event)
        processor.process(event)
        assert processor.stats()["stale_count"] == 0


@pytest.mark.unit
class TestDeletedEvents:
    def test_deleted_event_removes_document(self):
        processor = StreamEventProcessor()
        processor.process(
            make_event("evt-1", "doc-a", sequence=1, event_type="created", content="v1")
        )
        result = processor.process(
            make_event("evt-2", "doc-a", sequence=2, event_type="deleted", content="")
        )
        assert result == "applied"
        assert "doc-a" not in processor._handler.get_index_state()

    def test_stale_delete_does_not_remove_newer_content(self):
        processor = StreamEventProcessor()
        processor.process(
            make_event("evt-2", "doc-a", sequence=5, event_type="updated", content="new")
        )
        # An old delete, delayed on retry, must not remove newer content.
        result = processor.process(
            make_event("evt-1", "doc-a", sequence=2, event_type="deleted", content="")
        )
        assert result == "stale"
        assert processor._handler.get_index_state()["doc-a"] == "new"


@pytest.mark.unit
class TestStats:
    def test_initial_stats(self):
        processor = StreamEventProcessor()
        stats = processor.stats()
        assert stats["applied_count"] == 0
        assert stats["stale_count"] == 0
        assert stats["tracked_documents"] == 0

    def test_stats_after_mixed_events(self):
        processor = StreamEventProcessor()
        processor.process(make_event("evt-1", "doc-a", sequence=1, content="a"))
        processor.process(make_event("evt-2", "doc-b", sequence=1, content="b"))
        processor.process(make_event("evt-3", "doc-a", sequence=0, content="stale"))
        stats = processor.stats()
        assert stats["applied_count"] == 2
        assert stats["stale_count"] == 1
        assert stats["tracked_documents"] == 2


@pytest.mark.unit
class TestUsesExistingHandler:
    def test_accepts_injected_handler(self):
        handler = EventDrivenIngestionHandler()
        processor = StreamEventProcessor(handler=handler)
        processor.process(make_event("evt-1", "doc-a", sequence=1, content="v1"))
        # The injected handler instance must be the one actually mutated.
        assert handler.get_index_state()["doc-a"] == "v1"

    def test_does_not_subclass_or_replace_handler_logic(self):
        # StreamEventProcessor must compose EventDrivenIngestionHandler,
        # not reimplement its dedup/upsert logic from scratch.
        processor = StreamEventProcessor()
        assert isinstance(processor._handler, EventDrivenIngestionHandler)


@pytest.mark.integration
class TestStreamEventProcessorIntegration:
    """
    Интеграционные тесты: требуют реального брокера сообщений (Redis/Kafka)
    и запущенного Qdrant для проверки полного пути webhook -> очередь ->
    StreamEventProcessor -> векторный индекс.
    """

    def test_with_real_broker_and_qdrant(self):
        pytest.skip(
            "Требует запущенных Redis/Kafka и Qdrant — запускать вручную"
        )
