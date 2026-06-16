"""Tests for app/rag/dense_retriever.py — DenseRetriever implementation (C02 L4.1)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.rag.dense_retriever import DenseRetriever, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever_with_mock(
    collection_name: str = "test_col",
) -> tuple[DenseRetriever, MagicMock, MagicMock]:
    """Return (retriever, mock_client, mock_model) with patched dependencies."""
    with (
        patch("app.rag.dense_retriever.QdrantClient") as MockClient,
        patch("app.rag.dense_retriever.SentenceTransformer") as MockModel,
    ):
        mock_client = MagicMock()
        mock_model = MagicMock()
        MockClient.return_value = mock_client
        MockModel.return_value = mock_model

        # Stub encode() to return a fixed 4-dim vector
        import numpy as np
        mock_model.encode.return_value = np.array([0.5, 0.5, 0.5, 0.5])
        mock_model.get_sentence_embedding_dimension.return_value = 4

        retriever = DenseRetriever(collection_name=collection_name)

    # Re-attach mocks after __init__ (they're stored on the instance)
    retriever._client = mock_client
    retriever._model = mock_model
    return retriever, mock_client, mock_model


def _make_qdrant_hit(text: str, source: str, score: float) -> MagicMock:
    """Create a fake Qdrant ScoredPoint."""
    import numpy as np
    hit = MagicMock()
    hit.payload = {"text": text, "source": source}
    hit.score = score
    return hit


# ---------------------------------------------------------------------------
# Unit tests — no external services required
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedQuery:
    def test_returns_list_of_floats(self):
        import numpy as np
        retriever, _, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        result = retriever.embed_query("тестовый запрос")

        assert isinstance(result, list), "embed_query должен возвращать list"
        assert all(isinstance(x, float) for x in result), "Все элементы должны быть float"

    def test_query_prefix_applied(self):
        """Запрос должен быть передан в encode с префиксом 'query: '."""
        import numpy as np
        retriever, _, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])

        retriever.embed_query("поиск по документам")

        call_args = mock_model.encode.call_args
        text_arg = call_args[0][0] if call_args[0] else call_args[1].get("sentences", "")
        assert text_arg.startswith("query: "), (
            f"Запрос должен начинаться с 'query: ', получили: '{text_arg}'"
        )

    def test_normalize_embeddings_true(self):
        """encode должен вызываться с normalize_embeddings=True."""
        import numpy as np
        retriever, _, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([1.0, 0.0])

        retriever.embed_query("нормализация")

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True, (
            "normalize_embeddings должен быть True для корректного косинусного сходства"
        )


@pytest.mark.unit
class TestRetrieve:
    def test_returns_list_of_search_results(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        mock_client.search.return_value = [
            _make_qdrant_hit("Текст 1", "doc1.pdf", 0.92),
            _make_qdrant_hit("Текст 2", "doc2.pdf", 0.81),
        ]

        results = retriever.retrieve("запрос")

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results), (
            "Каждый элемент должен быть SearchResult"
        )

    def test_results_sorted_by_score_descending(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        mock_client.search.return_value = [
            _make_qdrant_hit("Текст A", "a.pdf", 0.91),
            _make_qdrant_hit("Текст B", "b.pdf", 0.75),
            _make_qdrant_hit("Текст C", "c.pdf", 0.60),
        ]

        results = retriever.retrieve("запрос", top_k=3)
        scores = [r.score for r in results]

        assert scores == sorted(scores, reverse=True), (
            f"Результаты должны быть отсортированы по убыванию score: {scores}"
        )

    def test_top_k_passed_to_qdrant(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        mock_client.search.return_value = []

        retriever.retrieve("запрос", top_k=7)

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs.get("limit") == 7, (
            f"Qdrant должен получить limit=7, получили: {call_kwargs.get('limit')}"
        )

    def test_uses_default_top_k_when_not_specified(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        retriever.top_k = 5
        mock_client.search.return_value = []

        retriever.retrieve("запрос")

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs.get("limit") == 5

    def test_result_fields_populated(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        mock_client.search.return_value = [
            _make_qdrant_hit("Политика отпусков", "hr_policy.pdf", 0.88)
        ]

        results = retriever.retrieve("отпуск")

        assert len(results) == 1
        r = results[0]
        assert r.text == "Политика отпусков"
        assert r.source == "hr_policy.pdf"
        assert abs(r.score - 0.88) < 1e-6

    def test_empty_qdrant_response(self):
        retriever, mock_client, _ = _make_retriever_with_mock()
        mock_client.search.return_value = []

        results = retriever.retrieve("запрос без результатов")

        assert results == []


@pytest.mark.unit
class TestBuildContext:
    def _retriever(self) -> DenseRetriever:
        r, _, _ = _make_retriever_with_mock()
        return r

    def test_empty_results_returns_fallback(self):
        context = self._retriever().build_context([])
        assert "не найдено" in context.lower() or len(context) > 0, (
            "Пустой список должен возвращать сообщение об отсутствии документов"
        )

    def test_document_numbering(self):
        results = [
            SearchResult(text="Первый документ", source="a.pdf", score=0.9),
            SearchResult(text="Второй документ", source="b.pdf", score=0.8),
        ]
        context = self._retriever().build_context(results)
        assert "[Документ 1]" in context
        assert "[Документ 2]" in context

    def test_source_included(self):
        results = [SearchResult(text="Текст", source="hr_policy.pdf", score=0.9)]
        context = self._retriever().build_context(results)
        assert "hr_policy.pdf" in context

    def test_text_included(self):
        results = [SearchResult(text="Текст политики отпусков", source="s.pdf", score=0.9)]
        context = self._retriever().build_context(results)
        assert "Текст политики отпусков" in context

    def test_documents_separated_by_blank_line(self):
        results = [
            SearchResult(text="Первый", source="a.pdf", score=0.9),
            SearchResult(text="Второй", source="b.pdf", score=0.8),
        ]
        context = self._retriever().build_context(results)
        assert "\n\n" in context, "Документы должны разделяться двойным переносом строки"


@pytest.mark.unit
class TestIndexDocuments:
    def test_passage_prefix_applied(self):
        """При индексировании текст должен передаваться с префиксом 'passage: '."""
        import numpy as np
        retriever, mock_client, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_client.get_collections.return_value.collections = []

        documents = [{"text": "Политика отпусков", "source": "hr.pdf"}]
        retriever.index_documents(documents)

        # Collect all encode calls
        calls = mock_model.encode.call_args_list
        texts = [c[0][0] for c in calls if c[0]]
        passage_calls = [t for t in texts if t.startswith("passage: ")]
        assert len(passage_calls) >= 1, (
            f"encode должен вызываться с 'passage: ...' для документов. "
            f"Фактические вызовы: {texts}"
        )

    def test_returns_count_of_indexed_documents(self):
        import numpy as np
        retriever, mock_client, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_client.get_collections.return_value.collections = []

        documents = [
            {"text": "Документ A", "source": "a.pdf"},
            {"text": "Документ B", "source": "b.pdf"},
            {"text": "Документ C", "source": "c.pdf"},
        ]
        count = retriever.index_documents(documents)

        assert count == 3, f"Ожидали 3 проиндексированных документа, получили {count}"

    def test_original_text_stored_in_payload(self):
        """В Qdrant должен сохраняться оригинальный текст (без префикса)."""
        import numpy as np
        retriever, mock_client, mock_model = _make_retriever_with_mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_client.get_collections.return_value.collections = []

        documents = [{"text": "Политика командировок", "source": "travel.pdf"}]
        retriever.index_documents(documents)

        upsert_call = mock_client.upsert.call_args
        points = upsert_call[1].get("points") or upsert_call[0][1]
        assert len(points) == 1
        payload = points[0].payload
        assert payload.get("text") == "Политика командировок", (
            "payload['text'] должен содержать оригинальный текст без префикса 'passage: '"
        )
        assert payload.get("source") == "travel.pdf"


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant on localhost:6333
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDenseRetrieverIntegration:
    """Require Qdrant running locally. Run with: pytest -m integration"""

    COLLECTION = "test_dense_retriever_c02_l41"

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create and clean up the test collection."""
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333")
        try:
            client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        yield
        try:
            client.delete_collection(self.COLLECTION)
        except Exception:
            pass

    def test_retrieve_returns_relevant_document(self):
        retriever = DenseRetriever(collection_name=self.COLLECTION)
        documents = [
            {"text": "Ежегодный отпуск составляет 28 календарных дней.", "source": "hr_01"},
            {"text": "Командировочные расходы возмещаются в течение 10 рабочих дней.", "source": "hr_02"},
            {"text": "Удалённая работа допускается до 3 дней в неделю.", "source": "hr_03"},
            {"text": "Медицинская страховка ДМС с первого дня работы.", "source": "hr_04"},
            {"text": "Процесс код-ревью требует двух одобрений.", "source": "tech_01"},
        ]
        retriever.index_documents(documents)

        results = retriever.retrieve("сколько дней отпуска", top_k=3)

        assert len(results) > 0
        top_sources = [r.source for r in results]
        assert "hr_01" in top_sources, (
            f"Ожидали 'hr_01' в top-3, получили: {top_sources}"
        )

    def test_recall_at_5_above_threshold(self):
        """Recall@5 на 10 запросах должен быть >= 0.60."""
        documents = [
            {"text": "Ежегодный отпуск сотрудников составляет 28 дней.", "source": "hr_vacation"},
            {"text": "Командировочные расходы возмещаются по авансовому отчёту.", "source": "hr_travel"},
            {"text": "Секреты и API-ключи хранятся в Vault.", "source": "sec_vault"},
            {"text": "CI/CD пайплайн запускается при мерже в main.", "source": "tech_cicd"},
            {"text": "Медицинская страховка ДМС предоставляется всем сотрудникам.", "source": "hr_dms"},
            {"text": "Оценка эффективности проводится в марте и сентябре.", "source": "hr_review"},
            {"text": "Дополнительный выходной день оформляется по согласованию с руководителем.", "source": "hr_dayoff"},
            {"text": "Задачи ведутся в Jira, доступ выдаёт тимлид.", "source": "hr_jira"},
            {"text": "Нагрузочное тестирование обязательно перед крупным релизом.", "source": "tech_load"},
            {"text": "Реферальный бонус составляет 50 000 рублей.", "source": "hr_referral"},
        ]
        retriever = DenseRetriever(collection_name=self.COLLECTION + "_recall")
        retriever.index_documents(documents)

        queries = [
            ("как оформить отгул", "hr_dayoff"),
            ("сколько дней отпуска", "hr_vacation"),
            ("как возместить командировку", "hr_travel"),
            ("когда оценка сотрудников", "hr_review"),
            ("где хранятся API-ключи", "sec_vault"),
            ("как работает CI/CD", "tech_cicd"),
            ("где ведутся задачи", "hr_jira"),
            ("страховка для сотрудников", "hr_dms"),
            ("бонус за реферала", "hr_referral"),
            ("нагрузочное тестирование перед релизом", "tech_load"),
        ]

        hits = 0
        for query, expected in queries:
            results = retriever.retrieve(query, top_k=5)
            if expected in [r.source for r in results]:
                hits += 1

        recall_at_5 = hits / len(queries)
        assert recall_at_5 >= 0.60, (
            f"Recall@5 = {recall_at_5:.2f} ниже порога 0.60. "
            f"Проверьте реализацию embed_query и index_documents."
        )
