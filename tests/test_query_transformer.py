"""Тесты для app/rag/query_transformer.py — урок 4.5.

Unit-тесты не требуют запущенных сервисов и ключа OpenAI.
Интеграционные тесты требуют OPENAI_API_KEY в переменных окружения.

Запуск:
    pytest tests/test_query_transformer.py -v -m unit
    OPENAI_API_KEY=sk-... pytest tests/test_query_transformer.py -v -m integration
"""
from __future__ import annotations

import pytest

from app.rag.query_transformer import (
    MockQueryGenerator,
    MultiQueryRetriever,
    SearchResult,
    SubqueryDecomposer,
    TransformConfig,
    TransformMode,
    QueryTransformer,
    deduplicate_results,
)


# ---------------------------------------------------------------------------
# Вспомогательная заглушка retriever для unit-тестов
# ---------------------------------------------------------------------------

class FakeRetriever:
    """Детерминированный retriever без внешних зависимостей.

    Возвращает документы из in-memory корпуса по подстрочному совпадению.
    """

    def __init__(self, corpus: list[dict]) -> None:
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_lower = query.lower()
        results = []
        for i, doc in enumerate(self._corpus):
            text_lower = doc["text"].lower()
            # простой скоринг: сколько слов из запроса встречается в документе
            words = query_lower.split()
            score = sum(1 for w in words if w in text_lower) / max(len(words), 1)
            if score > 0:
                results.append(
                    SearchResult(
                        text=doc["text"],
                        source=doc["source"],
                        score=score,
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def devops_corpus():
    from tests.fixtures.corpus_devops import CORPUS
    return CORPUS


@pytest.fixture
def fake_retriever(devops_corpus):
    return FakeRetriever(devops_corpus)


@pytest.fixture
def mock_generator():
    return MockQueryGenerator()


# ---------------------------------------------------------------------------
# Тесты deduplicate_results
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeduplicateResults:

    def test_empty_input(self):
        result = deduplicate_results([])
        assert result == []

    def test_single_list_no_duplicates(self):
        docs = [
            SearchResult(text="A", source="doc_a", score=0.9),
            SearchResult(text="B", source="doc_b", score=0.7),
        ]
        result = deduplicate_results([docs])
        assert len(result) == 2
        assert result[0].source == "doc_a"  # отсортировано по убыванию

    def test_duplicate_keeps_highest_score(self):
        list1 = [SearchResult(text="A", source="doc_a", score=0.5)]
        list2 = [SearchResult(text="A", source="doc_a", score=0.9)]
        result = deduplicate_results([list1, list2])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_sorted_by_score_descending(self):
        docs = [
            [SearchResult(text="C", source="doc_c", score=0.3)],
            [SearchResult(text="A", source="doc_a", score=0.9)],
            [SearchResult(text="B", source="doc_b", score=0.6)],
        ]
        result = deduplicate_results(docs)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_three_lists_with_overlap(self):
        shared = SearchResult(text="X", source="shared", score=0.8)
        list1 = [shared, SearchResult(text="A", source="doc_a", score=0.5)]
        list2 = [shared, SearchResult(text="B", source="doc_b", score=0.7)]
        list3 = [SearchResult(text="C", source="doc_c", score=0.6)]
        result = deduplicate_results([list1, list2, list3])
        sources = [r.source for r in result]
        assert sources.count("shared") == 1
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Тесты MockQueryGenerator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMockQueryGenerator:

    def test_generate_includes_original(self, mock_generator):
        query = "тест запроса"
        variants = mock_generator.generate(query, n=3)
        assert query in variants

    def test_generate_returns_n_plus_one(self, mock_generator):
        variants = mock_generator.generate("запрос", n=3)
        assert len(variants) == 4  # original + 3

    def test_generate_respects_n(self, mock_generator):
        variants = mock_generator.generate("запрос", n=1)
        assert len(variants) == 2  # original + 1

    def test_decompose_splits_by_conjunction(self, mock_generator):
        query = "задержка и поды падают"
        parts = mock_generator.decompose(query)
        assert len(parts) >= 2
        assert all(p.strip() for p in parts)

    def test_decompose_simple_query_returns_list(self, mock_generator):
        parts = mock_generator.decompose("простой запрос без союзов")
        assert isinstance(parts, list)
        assert len(parts) >= 1


# ---------------------------------------------------------------------------
# Тесты MultiQueryRetriever
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiQueryRetriever:

    def test_returns_list_of_search_results(self, fake_retriever, mock_generator):
        mq = MultiQueryRetriever(
            retriever=fake_retriever,
            query_generator=mock_generator,
            n_variants=2,
        )
        results = mq.retrieve("деплой задержка", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_respects_top_k(self, fake_retriever, mock_generator, devops_corpus):
        mq = MultiQueryRetriever(
            retriever=fake_retriever,
            query_generator=mock_generator,
            n_variants=3,
        )
        results = mq.retrieve("kubernetes поды", top_k=3)
        assert len(results) <= 3

    def test_no_duplicate_sources(self, fake_retriever, mock_generator):
        mq = MultiQueryRetriever(
            retriever=fake_retriever,
            query_generator=mock_generator,
            n_variants=3,
        )
        results = mq.retrieve("деплой задержка", top_k=10)
        sources = [r.source for r in results]
        assert len(sources) == len(set(sources)), "Дубликаты источников в результатах"

    def test_results_sorted_by_score(self, fake_retriever, mock_generator):
        mq = MultiQueryRetriever(
            retriever=fake_retriever,
            query_generator=mock_generator,
        )
        results = mq.retrieve("инцидент производительность", top_k=5)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_returns_empty(self, mock_generator):
        empty_retriever = FakeRetriever([])
        mq = MultiQueryRetriever(
            retriever=empty_retriever,
            query_generator=mock_generator,
        )
        results = mq.retrieve("любой запрос", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Тесты SubqueryDecomposer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSubqueryDecomposer:

    def test_returns_list_of_search_results(self, fake_retriever, mock_generator):
        decomposer = SubqueryDecomposer(
            retriever=fake_retriever,
            query_generator=mock_generator,
        )
        results = decomposer.retrieve("задержка и поды падают", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_respects_top_k(self, fake_retriever, mock_generator):
        decomposer = SubqueryDecomposer(
            retriever=fake_retriever,
            query_generator=mock_generator,
        )
        results = decomposer.retrieve("задержка и поды падают", top_k=3)
        assert len(results) <= 3

    def test_no_duplicate_sources(self, fake_retriever, mock_generator):
        decomposer = SubqueryDecomposer(
            retriever=fake_retriever,
            query_generator=mock_generator,
        )
        results = decomposer.retrieve("задержка и поды падают", top_k=10)
        sources = [r.source for r in results]
        assert len(sources) == len(set(sources))

    def test_compound_query_finds_more_than_simple(
        self, fake_retriever, mock_generator, devops_corpus
    ):
        """Декомпозиция составного запроса находит документы из разных разделов."""
        from tests.fixtures.corpus_devops import COMPOUND_QUERIES_WITH_GOLD

        decomposer = SubqueryDecomposer(
            retriever=fake_retriever,
            query_generator=mock_generator,
            top_k_per_subquery=5,
        )
        query, gold_sources = COMPOUND_QUERIES_WITH_GOLD[0]
        results = decomposer.retrieve(query, top_k=10)
        retrieved_sources = {r.source for r in results}
        # хотя бы один из эталонных документов должен быть найден
        assert any(s in retrieved_sources for s in gold_sources), (
            f"Ни один из эталонных документов не найден. "
            f"Ожидалось: {gold_sources}, найдено: {retrieved_sources}"
        )


# ---------------------------------------------------------------------------
# Тесты QueryTransformer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryTransformer:

    def test_none_mode_passes_through(self, fake_retriever, mock_generator):
        cfg = TransformConfig(mode=TransformMode.NONE)
        transformer = QueryTransformer(
            retriever=fake_retriever,
            config=cfg,
            query_generator=mock_generator,
        )
        results = transformer.retrieve("kubernetes деплой", top_k=5)
        assert isinstance(results, list)

    def test_multi_query_mode(self, fake_retriever, mock_generator):
        cfg = TransformConfig(mode=TransformMode.MULTI_QUERY, n_variants=2)
        transformer = QueryTransformer(
            retriever=fake_retriever,
            config=cfg,
            query_generator=mock_generator,
        )
        results = transformer.retrieve("задержка после деплоя", top_k=5)
        assert isinstance(results, list)
        sources = [r.source for r in results]
        assert len(sources) == len(set(sources))

    def test_decompose_mode(self, fake_retriever, mock_generator):
        cfg = TransformConfig(mode=TransformMode.DECOMPOSE)
        transformer = QueryTransformer(
            retriever=fake_retriever,
            config=cfg,
            query_generator=mock_generator,
        )
        results = transformer.retrieve("задержка и поды падают", top_k=5)
        assert isinstance(results, list)

    def test_none_mode_requires_no_generator(self, fake_retriever):
        cfg = TransformConfig(mode=TransformMode.NONE)
        # query_generator=None должен работать при mode=NONE
        transformer = QueryTransformer(
            retriever=fake_retriever,
            config=cfg,
            query_generator=None,
        )
        results = transformer.retrieve("запрос", top_k=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Тест recall: MultiQueryRetriever >= baseline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecallImprovement:

    def test_multi_query_recall_not_worse_than_baseline(
        self, fake_retriever, mock_generator
    ):
        """MultiQueryRetriever не должен давать recall хуже одиночного поиска."""
        from tests.fixtures.corpus_devops import COMPOUND_QUERIES_WITH_GOLD

        mq = MultiQueryRetriever(
            retriever=fake_retriever,
            query_generator=mock_generator,
            n_variants=3,
        )

        def recall_at_5(retriever, queries):
            hits = 0
            total = 0
            for query, gold_sources in queries:
                results = retriever.retrieve(query, top_k=5)
                retrieved = {r.source for r in results}
                hits += sum(1 for s in gold_sources if s in retrieved)
                total += len(gold_sources)
            return hits / total if total else 0.0

        r_baseline = recall_at_5(fake_retriever, COMPOUND_QUERIES_WITH_GOLD)
        r_multi = recall_at_5(mq, COMPOUND_QUERIES_WITH_GOLD)

        assert r_multi >= r_baseline, (
            f"MultiQueryRetriever recall ({r_multi:.2f}) хуже baseline ({r_baseline:.2f}). "
            f"Проверьте реализацию deduplicate_results и метода retrieve."
        )


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют OPENAI_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOpenAIQueryGeneratorIntegration:

    def test_generate_returns_variants(self):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY не установлен")

        from app.rag.query_transformer import OpenAIQueryGenerator
        generator = OpenAIQueryGenerator()
        variants = generator.generate("как сбросить пароль в linux", n=3)
        assert len(variants) >= 2
        assert isinstance(variants[0], str)

    def test_decompose_splits_compound_query(self):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY не установлен")

        from app.rag.query_transformer import OpenAIQueryGenerator
        generator = OpenAIQueryGenerator()
        query = "что делать если после деплоя выросла задержка и поды падают в kubernetes"
        parts = generator.decompose(query)
        assert len(parts) >= 2, (
            f"Ожидалось >=2 подзапросов, получено {len(parts)}: {parts}"
        )
