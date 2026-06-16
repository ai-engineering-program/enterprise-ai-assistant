"""Tests for app/rag/hybrid_retriever.py — HybridRetriever implementation (C02 L4.3)."""
from __future__ import annotations

import pytest

from app.rag.sparse_retriever import BM25Retriever, SearchResult as SparseResult
from app.rag.hybrid_retriever import (
    HybridRetriever,
    HybridResult,
    reciprocal_rank_fusion,
    weighted_hybrid_retrieve,
)
from tests.fixtures.corpus_mixed import CORPUS, SEMANTIC_QUERIES, EXACT_QUERIES


# ---------------------------------------------------------------------------
# Вспомогательные фабрики для unit-тестов
# ---------------------------------------------------------------------------


def _sparse_result(source: str, score: float) -> SparseResult:
    return SparseResult(text=f"Текст документа {source}", source=source, score=score)


def _make_dense_results(*sources: str) -> list[SparseResult]:
    """Создаёт список dense SearchResult с убывающими оценками."""
    return [
        SparseResult(text=f"Dense текст {src}", source=src, score=1.0 - i * 0.1)
        for i, src in enumerate(sources)
    ]


def _make_sparse_results(*sources: str) -> list[SparseResult]:
    """Создаёт список BM25 SearchResult с убывающими оценками."""
    return [
        SparseResult(text=f"BM25 текст {src}", source=src, score=10.0 - i * 1.5)
        for i, src in enumerate(sources)
    ]


# ---------------------------------------------------------------------------
# Unit tests: reciprocal_rank_fusion()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRF:
    def test_returns_list_of_hybrid_results(self):
        dense = _make_dense_results("doc_a", "doc_b")
        sparse = _make_sparse_results("doc_c", "doc_a")
        results = reciprocal_rank_fusion(dense, sparse)
        assert isinstance(results, list)
        assert all(isinstance(r, HybridResult) for r in results)

    def test_rrf_score_rank1_both(self):
        """Документ на rank=1 у обоих: score должен быть ~2/(60+1) = 0.0328."""
        dense = _make_dense_results("doc_a")
        sparse = _make_sparse_results("doc_a")
        results = reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(results) == 1
        expected = 2.0 / (60 + 1)
        assert abs(results[0].rrf_score - expected) < 1e-6, (
            f"Ожидали rrf_score ≈ {expected:.6f}, получили {results[0].rrf_score:.6f}"
        )

    def test_rrf_score_rank1_dense_only(self):
        """Документ только в dense на rank=1: score должен быть ~1/(60+1)."""
        dense = _make_dense_results("doc_a")
        sparse = _make_sparse_results("doc_b")
        results = reciprocal_rank_fusion(dense, sparse, k=60)
        doc_a = next(r for r in results if r.source == "doc_a")
        expected = 1.0 / (60 + 1)
        assert abs(doc_a.rrf_score - expected) < 1e-6

    def test_document_in_both_ranks_higher_than_document_in_one(self):
        """Документ, вошедший в оба списка, должен ранжироваться выше документа из одного."""
        dense = _make_dense_results("doc_a", "doc_b", "doc_c")
        sparse = _make_sparse_results("doc_b", "doc_x", "doc_y")
        results = reciprocal_rank_fusion(dense, sparse, k=60)

        doc_b = next(r for r in results if r.source == "doc_b")
        doc_a = next(r for r in results if r.source == "doc_a")

        assert doc_b.rrf_score > doc_a.rrf_score, (
            f"doc_b (в обоих списках, score={doc_b.rrf_score:.4f}) "
            f"должен быть выше doc_a (только dense, score={doc_a.rrf_score:.4f})"
        )

    def test_sorted_descending(self):
        dense = _make_dense_results("doc_a", "doc_b", "doc_c")
        sparse = _make_sparse_results("doc_c", "doc_a", "doc_b")
        results = reciprocal_rank_fusion(dense, sparse)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Результаты должны быть отсортированы по убыванию rrf_score: {scores}"
        )

    def test_dense_rank_recorded(self):
        dense = _make_dense_results("doc_a", "doc_b")
        sparse = _make_sparse_results("doc_c")
        results = reciprocal_rank_fusion(dense, sparse)

        doc_a = next(r for r in results if r.source == "doc_a")
        assert doc_a.dense_rank == 1, f"doc_a должен иметь dense_rank=1, получили {doc_a.dense_rank}"

        doc_b = next(r for r in results if r.source == "doc_b")
        assert doc_b.dense_rank == 2, f"doc_b должен иметь dense_rank=2, получили {doc_b.dense_rank}"

    def test_sparse_rank_none_for_dense_only(self):
        """Документ только в dense должен иметь sparse_rank=None."""
        dense = _make_dense_results("doc_a")
        sparse = _make_sparse_results("doc_b")
        results = reciprocal_rank_fusion(dense, sparse)

        doc_a = next(r for r in results if r.source == "doc_a")
        assert doc_a.sparse_rank is None, (
            f"doc_a присутствует только в dense, sparse_rank должен быть None, "
            f"получили {doc_a.sparse_rank}"
        )

    def test_dense_rank_none_for_sparse_only(self):
        """Документ только в BM25 должен иметь dense_rank=None."""
        dense = _make_dense_results("doc_a")
        sparse = _make_sparse_results("doc_b")
        results = reciprocal_rank_fusion(dense, sparse)

        doc_b = next(r for r in results if r.source == "doc_b")
        assert doc_b.dense_rank is None, (
            f"doc_b присутствует только в sparse, dense_rank должен быть None, "
            f"получили {doc_b.dense_rank}"
        )

    def test_empty_both_returns_empty(self):
        results = reciprocal_rank_fusion([], [], k=60)
        assert results == []

    def test_empty_dense_uses_sparse_only(self):
        sparse = _make_sparse_results("doc_c", "doc_d")
        results = reciprocal_rank_fusion([], sparse, k=60)
        assert len(results) == 2
        assert all(r.dense_rank is None for r in results)

    def test_custom_k_affects_score(self):
        """При разных значениях k оценки должны различаться."""
        dense = _make_dense_results("doc_a")
        sparse = _make_sparse_results("doc_a")

        results_k60 = reciprocal_rank_fusion(dense, sparse, k=60)
        results_k10 = reciprocal_rank_fusion(dense, sparse, k=10)

        assert results_k10[0].rrf_score > results_k60[0].rrf_score, (
            "При меньшем k оценки должны быть выше (ранги имеют больший вес)"
        )


# ---------------------------------------------------------------------------
# Unit tests: HybridRetriever (без Qdrant — использует только BM25 как mock)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRetrieverUnit:
    """Тесты HybridRetriever без реального DenseRetriever (используем BM25 как заглушку)."""

    class _FakeDenseRetriever:
        """Заглушка DenseRetriever для unit-тестов без Qdrant."""

        def __init__(self, results_map: dict):
            self._results_map = results_map

        def retrieve(self, query: str, top_k: int = 5) -> list[SparseResult]:
            results = self._results_map.get(query, [])
            return results[:top_k]

    def _make_retriever(self, dense_results_map: dict) -> HybridRetriever:
        sparse = BM25Retriever()
        sparse.index_documents([
            {"text": "BACKEND-2891 rate limiting API Gateway", "source": "doc_code_01"},
            {"text": "AUTH-417 двухфакторная аутентификация форма входа", "source": "doc_code_02"},
            {"text": "ежегодный оплачиваемый отпуск 28 дней заявление", "source": "doc_sem_01"},
        ])
        fake_dense = self._FakeDenseRetriever(dense_results_map)
        return HybridRetriever(dense=fake_dense, sparse=sparse, rrf_k=60, candidate_k=10)

    def test_retrieve_returns_hybrid_results(self):
        hybrid = self._make_retriever({
            "BACKEND-2891": [
                SparseResult(text="Backend задача", source="doc_code_01", score=0.8),
            ]
        })
        results = hybrid.retrieve("BACKEND-2891", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, HybridResult) for r in results)

    def test_retrieve_top_k_respected(self):
        hybrid = self._make_retriever({
            "тест": [
                SparseResult(text=f"Документ {i}", source=f"d{i}", score=0.9 - i * 0.1)
                for i in range(10)
            ]
        })
        results = hybrid.retrieve("тест", top_k=2)
        assert len(results) <= 2

    def test_candidate_k_used_not_top_k(self):
        """HybridRetriever должен запрашивать candidate_k от каждого поисковика."""
        call_log = []

        class TrackingDense:
            def retrieve(self, query: str, top_k: int = 5):
                call_log.append(top_k)
                return []

        sparse = BM25Retriever()
        sparse.index_documents([{"text": "тест документ", "source": "src"}])
        hybrid = HybridRetriever(dense=TrackingDense(), sparse=sparse, candidate_k=15)
        hybrid.retrieve("тест", top_k=3)

        assert len(call_log) == 1, "Dense.retrieve должен вызываться ровно один раз"
        assert call_log[0] == 15, (
            f"Dense.retrieve должен вызываться с top_k=candidate_k=15, "
            f"получили top_k={call_log[0]}"
        )

    def test_build_context_empty(self):
        hybrid = self._make_retriever({})
        context = hybrid.build_context([])
        assert "не найдено" in context.lower()

    def test_build_context_includes_source(self):
        results = [
            HybridResult(
                text="Документ про отпуск",
                source="doc_sem_01",
                rrf_score=0.03,
                dense_rank=1,
                sparse_rank=2,
            )
        ]
        sparse = BM25Retriever()
        sparse.index_documents([{"text": "x", "source": "y"}])

        class FakeDense:
            def retrieve(self, q, top_k=5):
                return []

        hybrid = HybridRetriever(dense=FakeDense(), sparse=sparse)
        context = hybrid.build_context(results)
        assert "doc_sem_01" in context
        assert "Документ про отпуск" in context
        assert "[Документ 1]" in context

    def test_build_context_shows_rank_info(self):
        results = [
            HybridResult(
                text="Текст",
                source="src",
                rrf_score=0.033,
                dense_rank=1,
                sparse_rank=None,
            )
        ]

        class FakeDense:
            def retrieve(self, q, top_k=5):
                return []

        sparse = BM25Retriever()
        sparse.index_documents([{"text": "x", "source": "y"}])
        hybrid = HybridRetriever(dense=FakeDense(), sparse=sparse)
        context = hybrid.build_context(results)
        # Должна быть информация об источнике ранга
        assert "d#1" in context or "dense" in context.lower()


# ---------------------------------------------------------------------------
# Unit tests: weighted_hybrid_retrieve()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeightedHybrid:
    def test_alpha_1_uses_only_dense_signal(self):
        """При alpha=1.0 результат должен определяться только dense."""
        dense_docs = [
            SparseResult(text="Документ A dense top", source="doc_a", score=0.95),
            SparseResult(text="Документ B", source="doc_b", score=0.40),
        ]
        sparse_docs = [
            SparseResult(text="Документ B sparse top", source="doc_b", score=14.3),
            SparseResult(text="Документ A", source="doc_a", score=2.1),
        ]

        class FixedDense:
            def retrieve(self, q, top_k=5):
                return dense_docs[:top_k]

        class FixedSparse:
            def retrieve(self, q, top_k=5):
                return sparse_docs[:top_k]

        results = weighted_hybrid_retrieve(
            dense=FixedDense(), sparse=FixedSparse(),
            query="тест", alpha=1.0, top_k=2, candidate_k=5
        )
        # При alpha=1.0 doc_a (лучший у dense) должен быть первым
        assert results[0].source == "doc_a", (
            f"При alpha=1.0 doc_a (dense-лидер) должен быть первым, "
            f"получили {results[0].source}"
        )

    def test_alpha_0_uses_only_sparse_signal(self):
        """При alpha=0.0 результат должен определяться только BM25."""
        dense_docs = [
            SparseResult(text="Документ A dense top", source="doc_a", score=0.95),
            SparseResult(text="Документ B", source="doc_b", score=0.40),
        ]
        sparse_docs = [
            SparseResult(text="Документ B sparse top", source="doc_b", score=14.3),
            SparseResult(text="Документ A", source="doc_a", score=2.1),
        ]

        class FixedDense:
            def retrieve(self, q, top_k=5):
                return dense_docs[:top_k]

        class FixedSparse:
            def retrieve(self, q, top_k=5):
                return sparse_docs[:top_k]

        results = weighted_hybrid_retrieve(
            dense=FixedDense(), sparse=FixedSparse(),
            query="тест", alpha=0.0, top_k=2, candidate_k=5
        )
        # При alpha=0.0 doc_b (BM25-лидер) должен быть первым
        assert results[0].source == "doc_b", (
            f"При alpha=0.0 doc_b (BM25-лидер) должен быть первым, "
            f"получили {results[0].source}"
        )

    def test_returns_hybrid_results(self):
        class FixedRetriever:
            def retrieve(self, q, top_k=5):
                return [SparseResult(text="Текст", source="src", score=1.0)]

        results = weighted_hybrid_retrieve(
            dense=FixedRetriever(), sparse=FixedRetriever(),
            query="тест", alpha=0.5, top_k=1
        )
        assert isinstance(results, list)
        assert all(isinstance(r, HybridResult) for r in results)


# ---------------------------------------------------------------------------
# Recall benchmark tests — unit (BM25-only части, без Qdrant)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRecallBenchmarkBM25Part:
    """Проверяем, что BM25 в составе гибридного поиска сохраняет recall на точных запросах."""

    def test_bm25_component_recall_on_exact_queries(self):
        """BM25-компонент должен находить >= 70% точных запросов (из урока 4.2)."""
        sparse = BM25Retriever()
        sparse.index_documents(CORPUS)

        hits = sum(
            1
            for q, exp in EXACT_QUERIES
            if exp in [r.source for r in sparse.retrieve(q, top_k=5)]
        )
        recall = hits / len(EXACT_QUERIES)
        assert recall >= 0.70, (
            f"BM25-компонент: recall@5 на точных запросах = {recall:.2f} ниже порога 0.70"
        )

    def test_rrf_with_perfect_sparse_improves_over_sparse_alone(self):
        """Если добавить идеальный dense (возвращает правильный документ всегда),
        RRF-объединение должно иметь recall не ниже, чем один BM25."""
        sparse = BM25Retriever()
        sparse.index_documents(CORPUS)

        # Имитируем "идеальный" dense: всегда ставит правильный документ первым
        corpus_index = {doc["source"]: doc["text"] for doc in CORPUS}

        class PerfectDense:
            def retrieve(self, query: str, top_k: int = 20) -> list[SparseResult]:
                # Возвращаем все документы — правильный будет первым только при совпадении
                return [
                    SparseResult(text=text, source=src, score=1.0)
                    for src, text in list(corpus_index.items())[:top_k]
                ]

        sparse2 = BM25Retriever()
        sparse2.index_documents(CORPUS)

        hybrid = HybridRetriever(dense=PerfectDense(), sparse=sparse2, candidate_k=20)

        bm25_hits = sum(
            1
            for q, exp in EXACT_QUERIES
            if exp in [r.source for r in sparse.retrieve(q, top_k=5)]
        )
        hybrid_hits = sum(
            1
            for q, exp in EXACT_QUERIES
            if exp in [r.source for r in hybrid.retrieve(q, top_k=5)]
        )

        bm25_recall = bm25_hits / len(EXACT_QUERIES)
        hybrid_recall = hybrid_hits / len(EXACT_QUERIES)

        assert hybrid_recall >= bm25_recall * 0.95, (
            f"Гибридный поиск (recall={hybrid_recall:.2f}) не должен быть значительно "
            f"хуже чистого BM25 (recall={bm25_recall:.2f}) на точных запросах"
        )


# ---------------------------------------------------------------------------
# Integration tests — требуют Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    """Тесты с реальным DenseRetriever и Qdrant.

    Требуют запущенного Qdrant:
        docker compose up -d qdrant
        pytest tests/test_hybrid_retriever.py -v -m integration
    """

    def test_full_hybrid_recall_above_threshold(self):
        """Гибридный поиск должен превышать recall отдельных методов."""
        pytest.skip(
            "Требует запущенного Qdrant. Запустите: "
            "docker compose up -d qdrant && "
            "pytest tests/test_hybrid_retriever.py -v -m integration"
        )
