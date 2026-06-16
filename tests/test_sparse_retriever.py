"""Tests for app/rag/sparse_retriever.py — BM25Retriever implementation (C02 L4.2)."""
from __future__ import annotations

import pytest

from app.rag.sparse_retriever import BM25Retriever, SearchResult
from tests.fixtures.corpus_mixed import CORPUS, EXACT_QUERIES, SEMANTIC_QUERIES


# ---------------------------------------------------------------------------
# Unit tests — no external services required
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTokenize:
    def test_lowercases_text(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("BACKEND AUTH Rate")
        assert all(t == t.lower() for t in tokens), (
            "Все токены должны быть в нижнем регистре"
        )

    def test_removes_punctuation_except_hyphen(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("задача, выполнена! статус: готово.")
        for t in tokens:
            assert "," not in t and "!" not in t and "." not in t, (
                f"Знаки препинания не должны присутствовать в токенах, получили: '{t}'"
            )

    def test_preserves_hyphenated_identifiers(self):
        """Составные идентификаторы с дефисом должны остаться единым токеном."""
        retriever = BM25Retriever()
        tokens = retriever.tokenize("BACKEND-2891 AUTH-417 НК-РФ-ст-217")
        assert "backend-2891" in tokens, (
            "Идентификатор BACKEND-2891 должен быть единым токеном 'backend-2891'"
        )
        assert "auth-417" in tokens, (
            "Идентификатор AUTH-417 должен быть единым токеном 'auth-417'"
        )

    def test_returns_list_of_strings(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("простой текст")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_string_returns_empty_list(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("")
        assert tokens == []


@pytest.mark.unit
class TestIndexDocuments:
    def test_returns_document_count(self):
        retriever = BM25Retriever()
        docs = [
            {"text": "BACKEND-2891: rate limiting", "source": "yt_1"},
            {"text": "AUTH-417: двухфакторная аутентификация", "source": "yt_2"},
            {"text": "INFRA-1103: обновление Kubernetes", "source": "yt_3"},
        ]
        count = retriever.index_documents(docs)
        assert count == 3, f"Ожидали 3 документа, получили {count}"

    def test_index_stores_documents(self):
        retriever = BM25Retriever()
        docs = [
            {"text": "Тестовый документ один", "source": "src_1"},
            {"text": "Тестовый документ два", "source": "src_2"},
        ]
        retriever.index_documents(docs)
        assert len(retriever._documents) == 2

    def test_index_empty_corpus(self):
        retriever = BM25Retriever()
        count = retriever.index_documents([])
        assert count == 0


@pytest.mark.unit
class TestRetrieve:
    def test_raises_before_indexing(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="[Ии]ндекс"):
            retriever.retrieve("BACKEND-2891")

    def test_returns_list_of_search_results(self):
        retriever = BM25Retriever()
        retriever.index_documents([
            {"text": "BACKEND-2891 rate limiting API Gateway", "source": "yt_1"},
            {"text": "AUTH-417 двухфакторная аутентификация", "source": "yt_2"},
        ])
        results = retriever.retrieve("BACKEND-2891")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_exact_match_found_first(self):
        """BM25 должен поставить точное совпадение на первое место."""
        retriever = BM25Retriever()
        retriever.index_documents([
            {"text": "бэкенд общие вопросы архитектура", "source": "general"},
            {"text": "BACKEND-2891 rate limiting API Gateway", "source": "exact"},
            {"text": "бэкенд сервисы авторизация безопасность", "source": "auth"},
        ])
        results = retriever.retrieve("BACKEND-2891", top_k=3)
        assert len(results) > 0, "Должен быть хотя бы один результат"
        assert results[0].source == "exact", (
            f"Документ с точным совпадением должен быть первым, "
            f"получили: {results[0].source}"
        )

    def test_results_sorted_by_score_descending(self):
        retriever = BM25Retriever()
        retriever.index_documents([
            {"text": "AUTH-417 AUTH-417 AUTH-417 форма", "source": "high"},
            {"text": "AUTH-417 разное", "source": "medium"},
            {"text": "совсем другой документ", "source": "low"},
        ])
        results = retriever.retrieve("AUTH-417", top_k=3)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True), (
                f"Результаты должны быть отсортированы по убыванию score: {scores}"
            )

    def test_zero_score_results_excluded(self):
        """Документы с нулевой оценкой не должны включаться в результаты."""
        retriever = BM25Retriever()
        retriever.index_documents([
            {"text": "BACKEND-2891 rate limiting", "source": "relevant"},
            {"text": "совершенно несвязанный документ про погоду", "source": "irrelevant"},
        ])
        results = retriever.retrieve("BACKEND-2891", top_k=5)
        for r in results:
            assert r.score > 0.0, (
                f"Документ с нулевой оценкой не должен попасть в результаты: {r}"
            )

    def test_top_k_respected(self):
        retriever = BM25Retriever()
        docs = [
            {"text": f"документ {i} содержит слово тест", "source": f"doc_{i}"}
            for i in range(10)
        ]
        retriever.index_documents(docs)
        results = retriever.retrieve("тест", top_k=3)
        assert len(results) <= 3, (
            f"Не должно быть больше top_k=3 результатов, получили {len(results)}"
        )

    def test_result_fields_populated(self):
        retriever = BM25Retriever()
        retriever.index_documents([
            {"text": "статья 217 НК РФ доходы не подлежащие налогообложению", "source": "nk_217"},
        ])
        results = retriever.retrieve("статья 217 НК РФ", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.source == "nk_217"
        assert "статья" in r.text.lower() or "217" in r.text
        assert r.score > 0.0


@pytest.mark.unit
class TestBuildContext:
    def _retriever_with_results(self) -> list[SearchResult]:
        return [
            SearchResult(text="BACKEND-2891: rate limiting", source="yt_1", score=3.5),
            SearchResult(text="AUTH-417: 2FA форма входа", source="yt_2", score=2.1),
        ]

    def test_empty_results_returns_fallback(self):
        retriever = BM25Retriever()
        context = retriever.build_context([])
        assert "не найдено" in context.lower(), (
            "Пустой список должен вернуть сообщение об отсутствии документов"
        )

    def test_document_numbering(self):
        retriever = BM25Retriever()
        context = retriever.build_context(self._retriever_with_results())
        assert "[Документ 1]" in context
        assert "[Документ 2]" in context

    def test_source_included(self):
        retriever = BM25Retriever()
        context = retriever.build_context(self._retriever_with_results())
        assert "yt_1" in context
        assert "yt_2" in context

    def test_text_included(self):
        retriever = BM25Retriever()
        context = retriever.build_context(self._retriever_with_results())
        assert "BACKEND-2891" in context
        assert "AUTH-417" in context

    def test_documents_separated_by_blank_line(self):
        retriever = BM25Retriever()
        context = retriever.build_context(self._retriever_with_results())
        assert "\n\n" in context, "Документы должны разделяться двойным переносом строки"

    def test_format_compatible_with_dense_retriever(self):
        """Формат должен совпадать с DenseRetriever.build_context()."""
        retriever = BM25Retriever()
        results = [SearchResult(text="Текст", source="src.pdf", score=1.0)]
        context = retriever.build_context(results)
        assert "[Документ 1]" in context
        assert "src.pdf" in context
        assert "Текст" in context


@pytest.mark.unit
class TestBM25Parameters:
    def test_custom_k1_accepted(self):
        retriever = BM25Retriever(k1=2.0)
        assert retriever.k1 == 2.0

    def test_custom_b_accepted(self):
        retriever = BM25Retriever(b=0.5)
        assert retriever.b == 0.5

    def test_default_parameters(self):
        retriever = BM25Retriever()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75


# ---------------------------------------------------------------------------
# Recall benchmark tests — no external services, but test quality threshold
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecallBenchmark:
    """Проверяет, что BM25 достигает порогового recall на точных запросах."""

    def test_recall_on_exact_queries_above_threshold(self):
        """BM25 должен найти >= 70% точных запросов (коды, идентификаторы) в top-5."""
        retriever = BM25Retriever()
        retriever.index_documents(CORPUS)

        hits = 0
        misses = []
        for query, expected_source in EXACT_QUERIES:
            results = retriever.retrieve(query, top_k=5)
            found_sources = [r.source for r in results]
            if expected_source in found_sources:
                hits += 1
            else:
                misses.append((query, expected_source, found_sources[:2]))

        recall = hits / len(EXACT_QUERIES)
        assert recall >= 0.70, (
            f"Recall@5 на точных запросах = {recall:.2f} ниже порога 0.70.\n"
            f"Не найдено:\n"
            + "\n".join(
                f"  Запрос: '{q}' | Ожидали: '{exp}' | Получили: {got}"
                for q, exp, got in misses
            )
        )

    def test_bm25_outperforms_on_exact_vs_semantic(self):
        """BM25 должен показать recall на точных запросах выше, чем на семантических.

        Это демонстрирует фундаментальное свойство разреженного поиска:
        он хорош на точных совпадениях, но слаб на перефразировках.
        """
        retriever = BM25Retriever()
        retriever.index_documents(CORPUS)

        exact_hits = sum(
            1
            for q, exp in EXACT_QUERIES
            if exp in [r.source for r in retriever.retrieve(q, top_k=5)]
        )
        semantic_hits = sum(
            1
            for q, exp in SEMANTIC_QUERIES
            if exp in [r.source for r in retriever.retrieve(q, top_k=5)]
        )

        exact_recall = exact_hits / len(EXACT_QUERIES)
        semantic_recall = semantic_hits / len(SEMANTIC_QUERIES)

        assert exact_recall > semantic_recall, (
            f"BM25 должен иметь recall выше на точных запросах ({exact_recall:.2f}) "
            f"чем на семантических ({semantic_recall:.2f}). "
            f"Если это не так — возможно, токенизация разбивает идентификаторы."
        )
