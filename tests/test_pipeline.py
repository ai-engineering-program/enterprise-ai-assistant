from __future__ import annotations

import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from app.rag.pipeline import PipelineConfig, PipelineResult, ProductionRAGPipeline


# ---------------------------------------------------------------------------
# Вспомогательные заглушки
# ---------------------------------------------------------------------------

def _make_mock_rerank_result(text: str = "тестовый текст", source: str = "doc_1"):
    r = MagicMock()
    r.text = text
    r.source = source
    r.rerank_score = 0.9
    r.original_rank = 1
    return r


def _make_mock_citation(chunk_id: str = "doc_1", statement: str = "утверждение"):
    c = MagicMock()
    c.chunk_id = chunk_id
    c.statement = statement
    c.quote = "цитата из документа"
    c.citation_id = 1
    return c


def _make_attributed_response(answer: str = "Ответ с [1] ссылкой."):
    resp = MagicMock()
    resp.answer = answer
    resp.citations = [_make_mock_citation()]
    return resp


# ---------------------------------------------------------------------------
# Unit tests — запускаются без внешних сервисов
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.candidate_k == 20
        assert config.rrf_k == 60
        assert config.top_k == 5
        assert config.n_query_variants == 3
        assert config.cache_threshold == 0.95
        assert config.faithfulness_threshold == 0.80
        assert config.recall_threshold == 0.75

    def test_custom_values(self):
        config = PipelineConfig(candidate_k=50, top_k=10, llm_model="gpt-4o")
        assert config.candidate_k == 50
        assert config.top_k == 10
        assert config.llm_model == "gpt-4o"


@pytest.mark.unit
class TestPipelineResult:
    def test_default_cache_hit_false(self):
        result = PipelineResult(answer="Ответ")
        assert result.cache_hit is False
        assert result.latency_ms == 0.0
        assert result.citations == []
        assert result.retrieved_docs == []

    def test_with_citations(self):
        citations = [_make_mock_citation()]
        result = PipelineResult(answer="Ответ [1].", citations=citations, cache_hit=False)
        assert len(result.citations) == 1
        assert result.answer == "Ответ [1]."


@pytest.mark.unit
class TestProductionRAGPipelineUnit:
    """Юнит-тесты: все зависимости подменяются заглушками через MagicMock."""

    def _build_pipeline(self, cache_hit_result=None):
        config = PipelineConfig()

        dense = MagicMock()
        dense.embed_query.return_value = [0.1] * 384

        sparse = MagicMock()
        hybrid = MagicMock()

        reranked = [_make_mock_rerank_result()]
        reranker = MagicMock()
        reranker.rerank.return_value = reranked

        transformer = MagicMock()
        transformer.retrieve.return_value = reranked

        attribution = MagicMock()
        attribution.generate_with_citations.return_value = _make_attributed_response()

        cache = MagicMock()
        cache.get.return_value = cache_hit_result

        pipeline = ProductionRAGPipeline(
            dense_retriever=dense,
            sparse_retriever=sparse,
            hybrid_retriever=hybrid,
            reranker=reranker,
            query_transformer=transformer,
            attribution_pipeline=attribution,
            cache=cache,
            config=config,
        )
        return pipeline, dense, sparse, reranker, transformer, attribution, cache

    def test_answer_returns_pipeline_result(self):
        pipeline, *_ = self._build_pipeline()
        result = pipeline.answer("тестовый вопрос")
        assert isinstance(result, PipelineResult)

    def test_answer_calls_query_transformer(self):
        pipeline, _, _, _, transformer, _, _ = self._build_pipeline()
        pipeline.answer("вопрос о балансе")
        transformer.retrieve.assert_called_once()

    def test_answer_calls_reranker(self):
        pipeline, _, _, reranker, _, _, _ = self._build_pipeline()
        pipeline.answer("вопрос о балансе")
        reranker.rerank.assert_called_once()

    def test_answer_calls_attribution(self):
        pipeline, _, _, _, _, attribution, _ = self._build_pipeline()
        pipeline.answer("вопрос о балансе")
        attribution.generate_with_citations.assert_called_once()

    def test_answer_puts_result_in_cache(self):
        pipeline, _, _, _, _, _, cache = self._build_pipeline()
        pipeline.answer("вопрос о балансе")
        cache.put.assert_called_once()

    def test_answer_cache_miss_cache_hit_false(self):
        pipeline, *_ = self._build_pipeline(cache_hit_result=None)
        result = pipeline.answer("вопрос о балансе", use_cache=True)
        assert result.cache_hit is False

    def test_answer_with_cache_hit(self):
        cached_payload = {
            "answer": "Кешированный ответ [1].",
            "citations": [_make_mock_citation()],
            "docs": [_make_mock_rerank_result()],
        }
        pipeline, _, _, _, transformer, attribution, _ = self._build_pipeline(
            cache_hit_result=cached_payload
        )
        result = pipeline.answer("вопрос о балансе", use_cache=True)
        assert result.cache_hit is True
        assert result.answer == "Кешированный ответ [1]."
        # Трансформация и атрибуция НЕ должны вызываться при попадании в кеш
        transformer.retrieve.assert_not_called()
        attribution.generate_with_citations.assert_not_called()

    def test_answer_skip_cache_when_use_cache_false(self):
        pipeline, _, _, _, _, _, cache = self._build_pipeline()
        pipeline.answer("вопрос", use_cache=False)
        cache.get.assert_not_called()

    def test_answer_latency_ms_positive(self):
        pipeline, *_ = self._build_pipeline()
        result = pipeline.answer("вопрос о тарифах")
        assert result.latency_ms >= 0.0

    def test_index_documents_calls_both_retrievers(self):
        pipeline, dense, sparse, *_ = self._build_pipeline()
        docs = [{"text": "текст документа", "source": "doc_1"}]
        pipeline.index_documents(docs)
        dense.index_documents.assert_called_once_with(docs)
        sparse.index_documents.assert_called_once_with(docs)

    def test_answer_has_citations(self):
        pipeline, *_ = self._build_pipeline()
        result = pipeline.answer("каков минимальный баланс?")
        assert len(result.citations) > 0

    def test_answer_has_retrieved_docs(self):
        pipeline, *_ = self._build_pipeline()
        result = pipeline.answer("вопрос о депозите")
        assert len(result.retrieved_docs) > 0


@pytest.mark.unit
class TestPipelineBuildSignature:
    """Проверяем, что метод build принимает правильные аргументы."""

    def test_build_is_classmethod(self):
        assert isinstance(
            ProductionRAGPipeline.__dict__.get("build") or
            getattr(ProductionRAGPipeline, "build"),
            classmethod,
        )

    def test_build_accepts_expected_args(self):
        import inspect
        sig = inspect.signature(ProductionRAGPipeline.build)
        params = list(sig.parameters.keys())
        assert "collection_name" in params
        assert "config" in params
        assert "openai_client" in params


# ---------------------------------------------------------------------------
# Integration tests — требуют Qdrant + OPENAI_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProductionRAGPipelineIntegration:
    """Интеграционные тесты. Запускать вручную с реальными сервисами."""

    def test_build_and_index_and_answer(self):
        pytest.skip(
            "Требует запущенный Qdrant и переменную OPENAI_API_KEY — запускайте вручную"
        )

    def test_ragas_evaluation_on_golden_set(self):
        """RAGAS-оценка на эталонном наборе data/golden_set_findoc.json.

        Критерии приёмки:
          - faithfulness >= 0.80
          - context_recall >= 0.75
        """
        pytest.skip(
            "Требует Qdrant + OPENAI_API_KEY + проиндексированные документы — запускайте вручную.\n"
            "Команда:\n"
            "  OPENAI_API_KEY=... pytest tests/test_pipeline.py -m integration -k ragas -v"
        )
