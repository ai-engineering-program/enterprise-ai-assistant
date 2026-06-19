import asyncio
import pytest
from datetime import datetime, timedelta

from app.ingestion.streaming_indexer import DocumentEvent, StreamingIndexer


def _make_event(
    doc_id: str = "doc_001",
    content: str = "Новость о событии в Москве",
    event_type: str = "created",
    seconds_ago: float = 5.0,
) -> DocumentEvent:
    return DocumentEvent(
        event_id=f"evt_{doc_id}_{event_type}",
        doc_id=doc_id,
        content=content,
        source="tass",
        published_at=datetime.utcnow() - timedelta(seconds=seconds_ago),
        event_type=event_type,
    )


@pytest.mark.unit
class TestComputeHash:
    """Тесты метода compute_hash — без внешних зависимостей."""

    def test_returns_string(self):
        indexer = StreamingIndexer("news_index")
        result = indexer.compute_hash("hello world")
        assert isinstance(result, str)

    def test_deterministic(self):
        indexer = StreamingIndexer("news_index")
        h1 = indexer.compute_hash("одинаковый текст")
        h2 = indexer.compute_hash("одинаковый текст")
        assert h1 == h2

    def test_different_content_different_hash(self):
        indexer = StreamingIndexer("news_index")
        h1 = indexer.compute_hash("текст А")
        h2 = indexer.compute_hash("текст Б")
        assert h1 != h2

    def test_hash_length_sha256(self):
        indexer = StreamingIndexer("news_index")
        h = indexer.compute_hash("любой текст")
        # SHA-256 hex digest всегда 64 символа
        assert len(h) == 64


@pytest.mark.unit
class TestIsDuplicate:
    """Тесты метода is_duplicate."""

    def test_unknown_doc_not_duplicate(self):
        indexer = StreamingIndexer("news_index")
        assert indexer.is_duplicate("doc_new", "какой-то текст") is False

    def test_same_content_is_duplicate(self):
        indexer = StreamingIndexer("news_index")
        content = "Новость о пожаре в Екатеринбурге"
        # Имитируем предыдущую индексацию: вручную прописываем хэш
        indexer._hashes["doc_001"] = indexer.compute_hash(content)
        assert indexer.is_duplicate("doc_001", content) is True

    def test_changed_content_not_duplicate(self):
        indexer = StreamingIndexer("news_index")
        indexer._hashes["doc_001"] = indexer.compute_hash("старый текст")
        assert indexer.is_duplicate("doc_001", "новый текст") is False


@pytest.mark.unit
class TestProcessEvent:
    """Тесты метода process_event."""

    def test_new_document_returns_true(self):
        indexer = StreamingIndexer("news_index")
        event = _make_event()
        result = asyncio.get_event_loop().run_until_complete(
            indexer.process_event(event)
        )
        assert result is True

    def test_upsert_count_increments(self):
        indexer = StreamingIndexer("news_index")
        event = _make_event()
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        assert indexer.stats()["upsert_count"] == 1

    def test_duplicate_returns_false(self):
        indexer = StreamingIndexer("news_index")
        event = _make_event(content="Неизменный текст новости")
        # Первая обработка
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        # Повторная с тем же содержимым
        result = asyncio.get_event_loop().run_until_complete(
            indexer.process_event(event)
        )
        assert result is False

    def test_skip_count_increments_on_duplicate(self):
        indexer = StreamingIndexer("news_index")
        event = _make_event(content="Статичный текст")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        assert indexer.stats()["skip_count"] == 1

    def test_updated_content_triggers_new_upsert(self):
        indexer = StreamingIndexer("news_index")
        event_v1 = _make_event(content="Версия 1: черновик")
        event_v2 = _make_event(content="Версия 2: исправленная")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event_v1))
        result = asyncio.get_event_loop().run_until_complete(
            indexer.process_event(event_v2)
        )
        assert result is True
        assert indexer.stats()["upsert_count"] == 2

    def test_hash_updated_after_content_change(self):
        indexer = StreamingIndexer("news_index")
        event_v1 = _make_event(doc_id="doc_x", content="Текст первой версии")
        event_v2 = _make_event(doc_id="doc_x", content="Текст второй версии")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event_v1))
        hash_v1 = indexer._hashes.get("doc_x")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event_v2))
        hash_v2 = indexer._hashes.get("doc_x")
        assert hash_v1 != hash_v2

    def test_deleted_event_removes_hash(self):
        indexer = StreamingIndexer("news_index")
        # Сначала индексируем
        event_create = _make_event(doc_id="doc_del", event_type="created")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event_create))
        assert "doc_del" in indexer._hashes
        # Затем удаляем
        event_delete = _make_event(doc_id="doc_del", event_type="deleted")
        result = asyncio.get_event_loop().run_until_complete(
            indexer.process_event(event_delete)
        )
        assert result is True
        assert "doc_del" not in indexer._hashes


@pytest.mark.unit
class TestStats:
    """Тесты метода stats."""

    def test_initial_stats(self):
        indexer = StreamingIndexer("news_index")
        s = indexer.stats()
        assert s["upsert_count"] == 0
        assert s["skip_count"] == 0
        assert s["avg_lag_seconds"] is None

    def test_avg_lag_positive_after_upsert(self):
        indexer = StreamingIndexer("news_index")
        event = _make_event(seconds_ago=10.0)
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        lag = indexer.stats()["avg_lag_seconds"]
        assert lag is not None
        assert lag >= 0.0


@pytest.mark.unit
class TestCustomEmbedFn:
    """Тесты с пользовательской функцией векторизации."""

    def test_custom_embed_fn_is_called(self):
        calls = []

        async def fake_embed(text: str) -> list[float]:
            calls.append(text)
            return [0.1, 0.2, 0.3, 0.4]

        indexer = StreamingIndexer("news_index", embed_fn=fake_embed)
        event = _make_event(content="Текст для векторизации")
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        assert len(calls) == 1
        assert calls[0] == "Текст для векторизации"

    def test_no_embed_fn_does_not_raise(self):
        indexer = StreamingIndexer("news_index", embed_fn=None)
        event = _make_event()
        # Не должно бросить исключение — используется вектор-заглушка
        asyncio.get_event_loop().run_until_complete(indexer.process_event(event))
        assert indexer.stats()["upsert_count"] == 1


@pytest.mark.integration
class TestStreamingIndexerIntegration:
    """
    Интеграционные тесты: требуют запущенный Qdrant.
    Запустить:  docker compose up -d qdrant
    Затем:      pytest tests/test_streaming_indexer.py -v -m integration
    """

    def test_with_real_qdrant(self):
        pytest.skip(
            "Требует запущенный Qdrant — запустите docker compose up -d qdrant"
        )
