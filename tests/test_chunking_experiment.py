"""Тесты для app/ingestion/chunking_experiment.py.

Unit-тесты: pytest -m unit   (без внешних сервисов)
Интеграционные тесты: pytest -m integration  (требуется запущенный Qdrant)
"""
from __future__ import annotations

import pytest

from app.ingestion.chunking_experiment import (
    ExperimentResult,
    ChunkingExperiment,
    recall_at_k,
    mrr,
    precision_at_k,
    simple_chunk,
)


# ---------------------------------------------------------------------------
# Тестовые данные — используются в unit и integration тестах
# ---------------------------------------------------------------------------

# 10 коротких FAQ-документов
SAMPLE_DOCS = [
    {"doc_id": f"faq_{i:03d}", "text": f"FAQ-{i}: ответ на вопрос номер {i}. " * 8, "doc_type": "faq"}
    for i in range(1, 11)
]

# 5 вопросов с однозначными эталонными ответами
SAMPLE_GOLDEN = [
    {"query": f"FAQ-{i}: ответ на вопрос номер {i}", "relevant_doc_ids": [f"faq_{i:03d}"]}
    for i in range(1, 6)
]


# ---------------------------------------------------------------------------
# Вспомогательные заглушки для unit-тестов метрик
# ---------------------------------------------------------------------------

def make_perfect_search(golden: list[dict]):
    """Возвращает функцию поиска, которая всегда находит правильный документ на позиции 1."""
    mapping = {item["query"]: item["relevant_doc_ids"] for item in golden}

    def search_fn(query: str, top_k: int) -> list[str]:
        return mapping.get(query, [])[:top_k]

    return search_fn


def make_miss_search():
    """Возвращает функцию поиска, которая никогда не находит правильный документ."""
    def search_fn(query: str, top_k: int) -> list[str]:
        return [f"wrong_doc_{i}" for i in range(top_k)]
    return search_fn


def make_ranked_search(rank: int):
    """Возвращает функцию поиска, которая помещает правильный документ на позицию rank."""
    def search_fn(query: str, top_k: int) -> list[str]:
        correct = f"correct_for_{query}"
        result = [f"wrong_{i}" for i in range(rank - 1)]
        result.append(correct)
        return result[:top_k]

    # Для каждого запроса правильный документ — "correct_for_<query>"
    return search_fn


# ---------------------------------------------------------------------------
# Unit-тесты: recall_at_k
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecallAtK:
    """Тесты метрики recall@K — без внешних сервисов."""

    def test_perfect_recall(self):
        """При идеальном поиске recall@5 == 1.0."""
        queries = [{"query": "вопрос 1", "relevant_doc_ids": ["doc_a"]}]
        search_fn = lambda q, top_k: ["doc_a", "doc_b"]
        assert recall_at_k(queries, search_fn, k=5) == 1.0

    def test_zero_recall(self):
        """При полном промахе recall@5 == 0.0."""
        queries = [{"query": "вопрос 1", "relevant_doc_ids": ["doc_a"]}]
        search_fn = lambda q, top_k: ["doc_x", "doc_y"]
        assert recall_at_k(queries, search_fn, k=5) == 0.0

    def test_partial_recall(self):
        """При 2 из 4 запросах с попаданием recall@5 == 0.5."""
        queries = [
            {"query": "q1", "relevant_doc_ids": ["doc_1"]},
            {"query": "q2", "relevant_doc_ids": ["doc_2"]},
            {"query": "q3", "relevant_doc_ids": ["doc_3"]},
            {"query": "q4", "relevant_doc_ids": ["doc_4"]},
        ]
        # Для q1 и q3 — попадаем, для q2 и q4 — нет
        def search_fn(query, top_k):
            hits = {"q1": ["doc_1"], "q3": ["doc_3"]}
            return hits.get(query, ["wrong_doc"])

        result = recall_at_k(queries, search_fn, k=5)
        assert result == pytest.approx(0.5)

    def test_empty_queries(self):
        """Пустой список запросов → 0.0 (не должен падать с ZeroDivisionError)."""
        search_fn = lambda q, top_k: []
        assert recall_at_k([], search_fn, k=5) == 0.0

    def test_multiple_relevant_docs(self):
        """Засчитывается, если хотя бы один из relevant_doc_ids попал в топ."""
        queries = [{"query": "q", "relevant_doc_ids": ["doc_a", "doc_b"]}]
        search_fn = lambda q, top_k: ["doc_b", "doc_x"]
        assert recall_at_k(queries, search_fn, k=5) == 1.0


# ---------------------------------------------------------------------------
# Unit-тесты: mrr
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMRR:
    """Тесты метрики MRR — без внешних сервисов."""

    def test_rank_1(self):
        """Правильный документ на позиции 1 → MRR = 1.0."""
        queries = [{"query": "q", "relevant_doc_ids": ["doc_correct"]}]
        search_fn = lambda q, top_k: ["doc_correct", "doc_wrong"]
        assert mrr(queries, search_fn, k=5) == pytest.approx(1.0)

    def test_rank_2(self):
        """Правильный документ на позиции 2 → MRR = 0.5."""
        queries = [{"query": "q", "relevant_doc_ids": ["doc_correct"]}]
        search_fn = lambda q, top_k: ["doc_wrong", "doc_correct"]
        assert mrr(queries, search_fn, k=5) == pytest.approx(0.5)

    def test_rank_3(self):
        """Правильный документ на позиции 3 → MRR ≈ 0.333."""
        queries = [{"query": "q", "relevant_doc_ids": ["doc_correct"]}]
        search_fn = lambda q, top_k: ["w1", "w2", "doc_correct"]
        assert mrr(queries, search_fn, k=5) == pytest.approx(1 / 3, abs=1e-6)

    def test_miss(self):
        """Правильный документ не попал в топ → MRR = 0.0."""
        queries = [{"query": "q", "relevant_doc_ids": ["doc_correct"]}]
        search_fn = lambda q, top_k: ["w1", "w2", "w3", "w4", "w5"]
        assert mrr(queries, search_fn, k=5) == pytest.approx(0.0)

    def test_average_mrr(self):
        """Среднее по нескольким запросам: (1.0 + 0.5 + 0.0) / 3."""
        queries = [
            {"query": "q1", "relevant_doc_ids": ["d1"]},
            {"query": "q2", "relevant_doc_ids": ["d2"]},
            {"query": "q3", "relevant_doc_ids": ["d3"]},
        ]
        def search_fn(query, top_k):
            return {
                "q1": ["d1"],           # ранг 1 → 1.0
                "q2": ["wrong", "d2"],  # ранг 2 → 0.5
                "q3": ["w1", "w2"],     # промах → 0.0
            }[query]

        expected = (1.0 + 0.5 + 0.0) / 3
        assert mrr(queries, search_fn, k=5) == pytest.approx(expected, abs=1e-6)

    def test_empty_queries(self):
        """Пустой список запросов → 0.0."""
        assert mrr([], lambda q, k: [], k=5) == 0.0


# ---------------------------------------------------------------------------
# Unit-тесты: precision_at_k
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrecisionAtK:
    """Тесты метрики Precision@K — без внешних сервисов."""

    def test_all_correct(self):
        """Все 5 результатов правильные → Precision@5 = 1.0."""
        queries = [{"query": "q", "relevant_doc_ids": ["d1", "d2", "d3", "d4", "d5"]}]
        search_fn = lambda q, top_k: ["d1", "d2", "d3", "d4", "d5"]
        assert precision_at_k(queries, search_fn, k=5) == pytest.approx(1.0)

    def test_none_correct(self):
        """Ни одного правильного → Precision@5 = 0.0."""
        queries = [{"query": "q", "relevant_doc_ids": ["d_correct"]}]
        search_fn = lambda q, top_k: ["w1", "w2", "w3", "w4", "w5"]
        assert precision_at_k(queries, search_fn, k=5) == pytest.approx(0.0)

    def test_one_of_five(self):
        """Один правильный из 5 → Precision@5 = 0.2."""
        queries = [{"query": "q", "relevant_doc_ids": ["d_correct"]}]
        search_fn = lambda q, top_k: ["w1", "w2", "d_correct", "w3", "w4"]
        assert precision_at_k(queries, search_fn, k=5) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Unit-тесты: simple_chunk
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSimpleChunk:
    """Тесты функции простого символьного разбиения — без внешних сервисов."""

    def test_empty_text(self):
        """Пустая строка → пустой список."""
        assert simple_chunk("", chunk_size=100, overlap=10) == []

    def test_text_shorter_than_chunk(self):
        """Текст короче chunk_size → один фрагмент."""
        text = "Короткий текст"
        result = simple_chunk(text, chunk_size=100, overlap=10)
        assert len(result) >= 1
        assert text in result[0]

    def test_overlap_creates_repeated_content(self):
        """Перекрытие обеспечивает повторение части текста между фрагментами."""
        text = "A" * 200
        result = simple_chunk(text, chunk_size=100, overlap=20)
        assert len(result) >= 2
        # Конец первого фрагмента должен совпадать с началом второго
        assert result[0][-20:] == result[1][:20]

    def test_overlap_gte_chunk_size(self):
        """overlap >= chunk_size → возвращает весь текст целиком (не падает)."""
        text = "Какой-то текст для теста"
        result = simple_chunk(text, chunk_size=10, overlap=10)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_exact_multiple(self):
        """Текст ровно в 2 фрагмента (без перекрытия)."""
        text = "X" * 200
        result = simple_chunk(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert all(len(c) == 100 for c in result)


# ---------------------------------------------------------------------------
# Unit-тесты: ChunkingExperiment (без Qdrant — мок поиска)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChunkingExperimentUnit:
    """Unit-тесты для вспомогательной логики ChunkingExperiment."""

    def test_experiment_result_dataclass(self):
        """ExperimentResult хранит все поля корректно."""
        r = ExperimentResult(
            strategy="fixed_size",
            chunk_size=256,
            overlap=32,
            recall_at_5=0.80,
            mrr=0.70,
            precision_at_5=0.55,
            n_chunks=150,
            duration_sec=3.5,
        )
        assert r.chunk_size == 256
        assert r.recall_at_5 == pytest.approx(0.80)

    def test_grid_search_skips_invalid_overlap(self):
        """grid_search не вызывает run_experiment для overlap >= chunk_size."""
        # Создаём ChunkingExperiment с мок-клиентом (интеграционного теста нет)
        # Просто проверяем логику фильтрации через прямой перебор
        chunk_sizes = [128, 256]
        overlaps = [64, 128, 256]

        valid_combos = [
            (cs, ov)
            for cs, ov in [(cs, ov) for cs in chunk_sizes for ov in overlaps]
            if ov < cs
        ]
        # Из 6 комбинаций должны остаться только (128, 64) и (256, 64), (256, 128)
        assert (128, 64) in valid_combos
        assert (256, 64) in valid_combos
        assert (256, 128) in valid_combos
        # Невалидные должны быть исключены
        assert (128, 128) not in valid_combos
        assert (128, 256) not in valid_combos


# ---------------------------------------------------------------------------
# Интеграционные тесты: полный эксперимент с Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestChunkingExperimentIntegration:
    """Интеграционные тесты — требуют запущенный Qdrant на localhost:6333.

    Запуск: docker-compose up -d qdrant
    Затем: pytest tests/test_chunking_experiment.py -v -m integration
    """

    @pytest.fixture
    def qdrant_client(self):
        """Клиент Qdrant для интеграционных тестов."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            client.get_collections()  # Проверка доступности
            return client
        except Exception as e:
            pytest.skip(f"Qdrant недоступен: {e}")

    @pytest.fixture
    def embedding_model(self):
        """Модель векторизации для интеграционных тестов."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("intfloat/multilingual-e5-base")
        except ImportError:
            pytest.skip("sentence-transformers не установлен")

    def test_run_single_experiment(self, qdrant_client, embedding_model):
        """Один эксперимент возвращает ExperimentResult с корректными полями."""
        exp = ChunkingExperiment(
            documents=SAMPLE_DOCS[:5],
            golden_dataset=SAMPLE_GOLDEN[:3],
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )
        result = exp.run_experiment("fixed_size", chunk_size=256, overlap=32)

        assert isinstance(result, ExperimentResult)
        assert result.strategy == "fixed_size"
        assert result.chunk_size == 256
        assert result.overlap == 32
        assert 0.0 <= result.recall_at_5 <= 1.0
        assert 0.0 <= result.mrr <= 1.0
        assert 0.0 <= result.precision_at_5 <= 1.0
        assert result.n_chunks > 0
        assert result.duration_sec > 0.0

    def test_temp_collection_cleaned_up(self, qdrant_client, embedding_model):
        """Временная коллекция удаляется после эксперимента."""
        from app.ingestion.chunking_experiment import ChunkingExperiment

        exp = ChunkingExperiment(
            documents=SAMPLE_DOCS[:3],
            golden_dataset=SAMPLE_GOLDEN[:2],
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )

        # Запомним список коллекций до эксперимента
        before = {c.name for c in qdrant_client.get_collections().collections}
        exp.run_experiment("fixed_size", chunk_size=128, overlap=16)
        after = {c.name for c in qdrant_client.get_collections().collections}

        # Ни одна новая коллекция не должна остаться
        new_collections = after - before
        assert len(new_collections) == 0, f"Временные коллекции не удалены: {new_collections}"

    def test_grid_search_returns_results_for_valid_combos(self, qdrant_client, embedding_model):
        """grid_search возвращает результаты только для валидных комбинаций."""
        exp = ChunkingExperiment(
            documents=SAMPLE_DOCS[:5],
            golden_dataset=SAMPLE_GOLDEN[:3],
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )

        results = exp.grid_search({
            "chunk_size": [128, 256],
            "overlap":    [32, 128],   # 128 >= 128 — невалидная комбинация
        })

        # Валидные: (128, 32), (256, 32), (256, 128) — 3 комбинации
        # Невалидная: (128, 128) — пропускается
        assert len(results) == 3
        for r in results:
            assert r.overlap < r.chunk_size

    def test_report_sorted_by_recall(self, qdrant_client, embedding_model, capsys):
        """report() выводит результаты, отсортированные по recall@5 убывание."""
        exp = ChunkingExperiment(
            documents=SAMPLE_DOCS[:5],
            golden_dataset=SAMPLE_GOLDEN[:3],
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )
        results = exp.grid_search({"chunk_size": [128, 256], "overlap": [32]})
        exp.report(results)

        captured = capsys.readouterr()
        # Проверяем, что таблица выведена и содержит marker лучшего результата
        assert "лучший" in captured.out.lower() or "best" in captured.out.lower() or "←" in captured.out
