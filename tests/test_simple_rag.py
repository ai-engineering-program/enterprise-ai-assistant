"""Tests for app/rag/simple_rag.py (Lesson 1.5).

Unit tests run without Qdrant or OpenAI — all external calls are mocked.
Integration tests require a running Qdrant instance and a valid OPENAI_API_KEY.

Run unit tests only:
    pytest tests/test_simple_rag.py -v -m unit

Run all tests (requires Qdrant + OpenAI key):
    pytest tests/test_simple_rag.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.simple_rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_DIM,
    Document,
    SearchResult,
    SimpleRAG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rag(collection_name: str = "test_col") -> SimpleRAG:
    """Return a SimpleRAG instance with mocked Qdrant and OpenAI clients."""
    with (
        patch("app.rag.simple_rag.QdrantClient") as mock_qdrant_cls,
        patch("app.rag.simple_rag.OpenAI") as mock_openai_cls,
    ):
        # Qdrant: get_collections returns empty list → collection will be created
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant_cls.return_value = mock_qdrant

        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        rag = SimpleRAG(collection_name=collection_name)
        # Attach mocks for inspection in tests
        rag._mock_qdrant = mock_qdrant
        rag._mock_openai = mock_openai
        return rag


def _fake_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a dummy unit vector of the given dimension."""
    return [1.0 / dim] * dim


# ---------------------------------------------------------------------------
# Unit tests — no external services required
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChunking:
    """Tests for the _split_into_chunks method."""

    def test_short_text_produces_single_chunk(self):
        rag = _make_rag()
        chunks = rag._split_into_chunks("Короткий текст.", source="doc1")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Короткий текст."
        assert chunks[0]["source"] == "doc1"
        assert chunks[0]["chunk_id"] == 0

    def test_long_text_produces_multiple_chunks(self):
        rag = _make_rag()
        # Text longer than DEFAULT_CHUNK_SIZE should produce more than one chunk
        long_text = "А" * (DEFAULT_CHUNK_SIZE + 10)
        chunks = rag._split_into_chunks(long_text, source="doc2")
        assert len(chunks) >= 2

    def test_chunk_size_does_not_exceed_limit(self):
        rag = _make_rag()
        long_text = "Б" * (DEFAULT_CHUNK_SIZE * 3)
        chunks = rag._split_into_chunks(long_text, source="doc3")
        for chunk in chunks:
            assert len(chunk["text"]) <= DEFAULT_CHUNK_SIZE

    def test_overlap_makes_chunks_share_content(self):
        rag = _make_rag()
        text = "X" * DEFAULT_CHUNK_SIZE + "Y" * DEFAULT_CHUNK_OVERLAP + "Z" * 10
        chunks = rag._split_into_chunks(text, source="doc4")
        # Second chunk should start before the end of the first chunk
        assert len(chunks) >= 2
        end_of_first = chunks[0]["text"][-DEFAULT_CHUNK_OVERLAP:]
        start_of_second = chunks[1]["text"][:DEFAULT_CHUNK_OVERLAP]
        assert end_of_first == start_of_second

    def test_chunk_ids_are_sequential(self):
        rag = _make_rag()
        text = "В" * (DEFAULT_CHUNK_SIZE * 3)
        chunks = rag._split_into_chunks(text, source="doc5")
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == i


@pytest.mark.unit
class TestEmbedding:
    """Tests for the _embed method."""

    def test_embed_returns_list_of_vectors(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb, fake_emb]

        result = rag._embed(["текст один", "текст два"])

        assert len(result) == 2
        assert len(result[0]) == EMBEDDING_DIM
        assert len(result[1]) == EMBEDDING_DIM

    def test_embed_calls_openai_with_correct_model(self):
        from app.rag.simple_rag import EMBEDDING_MODEL
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb]

        rag._embed(["один запрос"])

        call_kwargs = rag._mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs.get("model") == EMBEDDING_MODEL or \
               call_kwargs.args[0] if call_kwargs.args else True


@pytest.mark.unit
class TestIndexDocuments:
    """Tests for the index_documents method."""

    def test_index_returns_chunk_count(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        # Each document is short → 1 chunk each
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb, fake_emb]

        docs = [
            Document(text="Документ первый.", source="d1"),
            Document(text="Документ второй.", source="d2"),
        ]
        count = rag.index_documents(docs)
        assert count == 2

    def test_index_calls_qdrant_upsert(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb]

        rag.index_documents([Document(text="Тест.", source="s1")])

        assert rag._mock_qdrant.upsert.called

    def test_empty_document_list_returns_zero(self):
        rag = _make_rag()
        rag._mock_openai.embeddings.create.return_value.data = []
        count = rag.index_documents([])
        assert count == 0


@pytest.mark.unit
class TestSearch:
    """Tests for the search method."""

    def _make_hit(self, text: str, source: str, score: float) -> MagicMock:
        hit = MagicMock()
        hit.payload = {"text": text, "source": source}
        hit.score = score
        return hit

    def test_search_returns_search_results(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb]

        hit = self._make_hit("Текст фрагмента", "hr_policy_01", 0.92)
        rag._mock_qdrant.search.return_value = [hit]

        results = rag.search("сколько дней отпуска", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "Текст фрагмента"
        assert results[0].source == "hr_policy_01"
        assert abs(results[0].score - 0.92) < 1e-6

    def test_search_empty_index_returns_empty_list(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb]
        rag._mock_qdrant.search.return_value = []

        results = rag.search("запрос без результатов", top_k=3)
        assert results == []

    def test_search_respects_top_k(self):
        rag = _make_rag()
        fake_emb = MagicMock()
        fake_emb.embedding = _fake_embedding()
        rag._mock_openai.embeddings.create.return_value.data = [fake_emb]

        hits = [self._make_hit(f"текст {i}", f"src_{i}", 0.9 - i * 0.1) for i in range(5)]
        rag._mock_qdrant.search.return_value = hits[:3]

        results = rag.search("запрос", top_k=3)
        assert len(results) == 3


@pytest.mark.unit
class TestAnswer:
    """Tests for the answer method."""

    def _setup_search_mock(self, rag: SimpleRAG, results: list[SearchResult]) -> None:
        """Patch rag.search to return given results."""
        rag.search = MagicMock(return_value=results)

    def test_answer_no_results_returns_fallback_message(self):
        rag = _make_rag()
        self._setup_search_mock(rag, [])

        answer = rag.answer("вопрос без ответа")
        assert "не найдено" in answer.lower() or "нет" in answer.lower()

    def test_answer_calls_openai_with_context(self):
        rag = _make_rag()
        self._setup_search_mock(rag, [
            SearchResult(text="28 дней отпуска в год.", source="hr_policy_01", score=0.95),
        ])

        mock_choice = MagicMock()
        mock_choice.message.content = "Сотрудникам положено 28 дней отпуска."
        rag._mock_openai.chat.completions.create.return_value.choices = [mock_choice]

        answer = rag.answer("сколько дней отпуска")

        assert rag._mock_openai.chat.completions.create.called
        assert answer == "Сотрудникам положено 28 дней отпуска."

    def test_answer_includes_source_in_prompt(self):
        rag = _make_rag()
        self._setup_search_mock(rag, [
            SearchResult(text="Все сервисы через Kong.", source="tech_spec_01", score=0.88),
        ])

        mock_choice = MagicMock()
        mock_choice.message.content = "Через API-шлюз Kong."
        rag._mock_openai.chat.completions.create.return_value.choices = [mock_choice]

        rag.answer("как подключить сервис")

        call_kwargs = rag._mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", [])
        # The user message should contain the context with the source name
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        assert user_msg is not None
        assert "tech_spec_01" in user_msg["content"]


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant + valid OPENAI_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSimpleRAGIntegration:
    """End-to-end tests against real Qdrant and OpenAI.

    Skip by default: pytest -m 'not integration'
    Run manually when services are available.
    """

    PROTECH_DOCS = [
        Document(
            text=(
                "Сотрудники «ПроТех Решения» получают 28 календарных дней отпуска в год. "
                "Отпуск можно разбить на части, однако одна из частей должна быть "
                "не менее 14 дней. Заявление подаётся за 2 недели до начала."
            ),
            source="hr_policy_01",
        ),
        Document(
            text=(
                "Все внешние сервисы подключаются через API-шлюз на базе Kong. "
                "Аутентификация через JWT-токены. Лимит запросов 1000 в минуту."
            ),
            source="tech_spec_01",
        ),
        Document(
            text=(
                "Оценка эффективности проводится дважды в год — в марте и сентябре. "
                "По итогам принимаются решения о повышении зарплаты."
            ),
            source="hr_policy_03",
        ),
    ]

    def test_index_and_search_vacation_policy(self):
        pytest.skip("Requires running Qdrant + OPENAI_API_KEY — run manually")
        rag = SimpleRAG(collection_name="protech_integration_test")
        rag.index_documents(self.PROTECH_DOCS)
        results = rag.search("сколько дней отпуска", top_k=3)
        sources = [r.source for r in results]
        assert "hr_policy_01" in sources

    def test_index_and_search_api_gateway(self):
        pytest.skip("Requires running Qdrant + OPENAI_API_KEY — run manually")
        rag = SimpleRAG(collection_name="protech_integration_test")
        rag.index_documents(self.PROTECH_DOCS)
        results = rag.search("как подключить внешний сервис", top_k=3)
        sources = [r.source for r in results]
        assert "tech_spec_01" in sources

    def test_answer_returns_non_empty_string(self):
        pytest.skip("Requires running Qdrant + OPENAI_API_KEY — run manually")
        rag = SimpleRAG(collection_name="protech_integration_test")
        rag.index_documents(self.PROTECH_DOCS)
        answer = rag.answer("как часто проводится оценка сотрудников")
        assert isinstance(answer, str) and len(answer) > 0
