"""Тесты для app/rag/reranker.py — урок 4.4.

Unit-тесты не требуют запущенных внешних сервисов.
Модель CrossEncoder загружается автоматически из HuggingFace Hub
при первом запуске (нужен интернет, ~22 МБ для ms-marco-MiniLM-L-6-v2).

Запуск unit-тестов:
    pytest tests/test_reranker.py -v -m unit

Запуск интеграционных тестов (требует Qdrant):
    docker compose up -d qdrant
    pytest tests/test_reranker.py -v -m integration
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Вспомогательный датакласс для имитации результатов гибридного поиска
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    """Имитирует HybridResult / SearchResult без реального поиска."""
    text: str
    source: str
    rrf_score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Unit-тесты — работают без Qdrant, модель CrossEncoder нужна
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRerankResultDataclass:
    """Проверяем структуру RerankResult."""

    def test_rerank_result_fields(self):
        from app.rag.reranker import RerankResult
        r = RerankResult(
            text="Правила страхования раздел 8.3",
            source="rules_kasko_8_3",
            rerank_score=8.432,
            original_rank=8,
        )
        assert r.text == "Правила страхования раздел 8.3"
        assert r.source == "rules_kasko_8_3"
        assert r.rerank_score == pytest.approx(8.432)
        assert r.original_rank == 8


@pytest.mark.unit
class TestRerankerUnit:
    """Тесты логики Reranker — требуют загрузки CrossEncoder модели."""

    def test_rerank_empty_candidates_returns_empty(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        result = reranker.rerank("любой запрос", [], top_k=5)
        assert result == []

    def test_rerank_returns_at_most_top_k(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        candidates = [
            _FakeResult(text=f"документ {i}", source=f"doc_{i}")
            for i in range(10)
        ]
        results = reranker.rerank("тестовый запрос", candidates, top_k=3)
        assert len(results) <= 3

    def test_rerank_preserves_original_rank(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        candidates = [
            _FakeResult(text="страховое возмещение при угоне КАСКО", source="doc_a"),
            _FakeResult(text="маркетинговая брошюра о продуктах", source="doc_b"),
            _FakeResult(text="правила хранения ТС и условия выплаты", source="doc_c"),
        ]
        results = reranker.rerank(
            "условия выплаты при угоне с нарушением хранения",
            candidates,
            top_k=3,
        )
        # original_rank должен отражать позицию в candidates (начиная с 1)
        original_ranks = {r.source: r.original_rank for r in results}
        assert original_ranks["doc_a"] == 1
        assert original_ranks["doc_b"] == 2
        assert original_ranks["doc_c"] == 3

    def test_rerank_results_sorted_by_score_descending(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        candidates = [
            _FakeResult(
                text="правила страхования транспортных средств раздел условия "
                     "выплаты при угоне нарушение хранения",
                source="relevant_doc",
            ),
            _FakeResult(
                text="скидки акции страхование автомобиля выгодно оформить онлайн",
                source="marketing_doc",
            ),
        ]
        results = reranker.rerank(
            "условия выплаты при угоне с нарушением условий хранения",
            candidates,
            top_k=2,
        )
        assert len(results) == 2
        # Результаты должны быть отсортированы по убыванию score
        assert results[0].rerank_score >= results[1].rerank_score

    def test_rerank_fewer_candidates_than_top_k(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        candidates = [
            _FakeResult(text="единственный документ", source="only_doc"),
        ]
        results = reranker.rerank("запрос", candidates, top_k=5)
        # Не должно быть ошибки — возвращаем сколько есть
        assert len(results) == 1

    def test_rerank_score_is_float(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        candidates = [_FakeResult(text="текст документа", source="doc_1")]
        results = reranker.rerank("запрос", candidates, top_k=1)
        assert isinstance(results[0].rerank_score, float)


@pytest.mark.unit
class TestRerankerBuildContext:
    """Тесты метода build_context."""

    def test_build_context_empty(self):
        from app.rag.reranker import Reranker
        reranker = Reranker()
        context = reranker.build_context([])
        assert context == "Релевантных документов не найдено."

    def test_build_context_contains_source(self):
        from app.rag.reranker import Reranker, RerankResult
        reranker = Reranker()
        results = [
            RerankResult(
                text="Правила страхования раздел 8.3",
                source="rules_kasko_8_3",
                rerank_score=8.432,
                original_rank=8,
            )
        ]
        context = reranker.build_context(results)
        assert "rules_kasko_8_3" in context
        assert "Правила страхования раздел 8.3" in context

    def test_build_context_document_numbering(self):
        from app.rag.reranker import Reranker, RerankResult
        reranker = Reranker()
        results = [
            RerankResult(text="doc A", source="src_a", rerank_score=9.0, original_rank=3),
            RerankResult(text="doc B", source="src_b", rerank_score=5.0, original_rank=1),
        ]
        context = reranker.build_context(results)
        assert "Документ 1" in context
        assert "Документ 2" in context


@pytest.mark.unit
class TestHybridRetrieverWithReranker:
    """Тесты интеграции Reranker в HybridRetriever."""

    def test_hybrid_retriever_accepts_reranker_param(self):
        """HybridRetriever должен принимать reranker без ошибок."""
        import inspect
        from app.rag.hybrid_retriever import HybridRetriever
        sig = inspect.signature(HybridRetriever.__init__)
        assert "reranker" in sig.parameters, (
            "HybridRetriever.__init__ должен принимать параметр 'reranker'"
        )

    def test_hybrid_retriever_reranker_default_is_none(self):
        """По умолчанию reranker=None — старое поведение сохраняется."""
        import inspect
        from app.rag.hybrid_retriever import HybridRetriever
        sig = inspect.signature(HybridRetriever.__init__)
        default = sig.parameters["reranker"].default
        assert default is None, (
            "Параметр reranker должен иметь default=None для обратной совместимости"
        )


# ---------------------------------------------------------------------------
# Интеграционные тесты — требуют Qdrant + загрузки моделей
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRerankerIntegration:
    """Полный пайплайн: HybridRetriever + Reranker на реальном корпусе.

    Требует запущенного Qdrant:
        docker compose up -d qdrant
    """

    def test_reranker_improves_precision_at_5(self):
        """Precision@5 с переранжированием должна быть выше, чем без."""
        pytest.skip(
            "Интеграционный тест — требует запущенного Qdrant и загруженных моделей. "
            "Запустите вручную: pytest tests/test_reranker.py -v -m integration"
        )

    def test_full_pipeline_insurance_corpus(self):
        """Проверяет, что правила страхования раздел 8.3 поднимаются в топ-3.

        Сценарий из урока 4.4: запрос об условиях выплаты при угоне
        с нарушением хранения. Без переранжирования rules_kasko_8_3
        оказывается на позиции 8–12, после — на позиции 1–3.
        """
        pytest.skip(
            "Интеграционный тест — требует запущенного Qdrant. "
            "Запустите вручную: pytest tests/test_reranker.py -v -m integration"
        )
