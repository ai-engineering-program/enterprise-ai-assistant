"""
Tests for app.rag.model_selector.ModelBenchmark

Unit tests run without any external services or API keys.
Integration tests require OPENAI_API_KEY environment variable.
"""
import numpy as np
import pytest

from app.rag.model_selector import BenchmarkResult, ModelBenchmark

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "Настройка протокола авторизации OAuth2 в Kubernetes-кластере.",
    "Регламент проведения квартальной аттестации сотрудников.",
    "Процедура эскалации инцидентов в службу технической поддержки.",
    "Инструкция по настройке CI/CD пайплайна в GitLab.",
    "Политика безопасного хранения персональных данных клиентов.",
]

QUERIES = [
    "Как настроить OAuth2 авторизацию?",
    "Как проходит аттестация сотрудников?",
    "Что делать при критическом инциденте?",
    "Как запустить CI/CD в GitLab?",
    "Какие правила хранения персональных данных?",
]

# ground_truth[i] — индекс правильного документа для QUERIES[i]
GROUND_TRUTH = [0, 1, 2, 3, 4]


def make_benchmark() -> ModelBenchmark:
    return ModelBenchmark(
        test_documents=DOCUMENTS,
        test_queries=QUERIES,
        ground_truth=GROUND_TRUTH,
    )


# ---------------------------------------------------------------------------
# Unit tests — no external services required
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModelBenchmarkUnit:
    """Tests that run without any external services or API keys."""

    def test_instantiation_stores_data(self):
        """ModelBenchmark stores documents, queries, and ground truth."""
        bm = make_benchmark()
        assert hasattr(bm, "_documents") or hasattr(bm, "_queries"), (
            "ModelBenchmark must store test_documents and test_queries"
        )

    def test_benchmark_result_is_dataclass(self):
        """BenchmarkResult is a proper dataclass with expected fields."""
        result = BenchmarkResult(
            model_name="test-model",
            recall_at_5=0.8,
            avg_latency_sec=0.05,
            model_type="local",
        )
        assert result.model_name == "test-model"
        assert result.recall_at_5 == 0.8
        assert result.avg_latency_sec == 0.05
        assert result.model_type == "local"

    def test_evaluate_with_mock_vectors(self, monkeypatch):
        """evaluate() returns correct recall@5 when vectors are provided via monkeypatch."""
        bm = make_benchmark()

        # Создаём идентичные векторы для документов и запросов — recall должен быть 1.0
        identity_vectors = np.eye(len(DOCUMENTS), dtype=np.float32)

        def mock_local(texts, model_name):
            return identity_vectors[: len(texts)]

        monkeypatch.setattr(bm, "_build_vectors_local", mock_local)

        result = bm.evaluate({"name": "mock-model", "type": "local"})

        assert isinstance(result, BenchmarkResult)
        assert result.recall_at_5 == pytest.approx(1.0, abs=0.01), (
            f"Expected recall@5=1.0 for identical vectors, got {result.recall_at_5}"
        )
        assert result.avg_latency_sec >= 0.0
        assert result.model_name == "mock-model"

    def test_evaluate_bad_vectors_gives_low_recall(self, monkeypatch):
        """evaluate() returns recall~0 when vectors are randomly reversed."""
        bm = make_benchmark()

        rng = np.random.default_rng(seed=42)

        def mock_random(texts, model_name):
            # Запросы кодируем случайными векторами — нет соответствия документам
            return rng.standard_normal((len(texts), 32)).astype(np.float32)

        monkeypatch.setattr(bm, "_build_vectors_local", mock_random)

        result = bm.evaluate({"name": "random-model", "type": "local"})

        assert isinstance(result, BenchmarkResult)
        assert result.recall_at_5 < 0.8, (
            "Random vectors should not achieve recall@5 >= 0.8"
        )

    def test_compare_returns_sorted_results(self, monkeypatch):
        """compare() returns results sorted by recall_at_5 descending."""
        bm = make_benchmark()

        call_count = {"n": 0}

        def mock_evaluate(model_config):
            # Возвращаем убывающий recall: 0.9, 0.7, 0.5 для трёх моделей
            recall_values = [0.9, 0.7, 0.5]
            idx = call_count["n"]
            call_count["n"] += 1
            return BenchmarkResult(
                model_name=model_config["name"],
                recall_at_5=recall_values[idx],
                avg_latency_sec=0.1,
                model_type=model_config.get("type", "local"),
            )

        monkeypatch.setattr(bm, "evaluate", mock_evaluate)

        configs = [
            {"name": "model-A", "type": "local"},
            {"name": "model-B", "type": "local"},
            {"name": "model-C", "type": "local"},
        ]

        results = bm.compare(configs)

        assert len(results) == 3
        recalls = [r.recall_at_5 for r in results]
        assert recalls == sorted(recalls, reverse=True), (
            f"Results must be sorted by recall_at_5 descending, got: {recalls}"
        )

    def test_evaluate_handles_api_error_gracefully(self, monkeypatch):
        """evaluate() does not raise when _build_vectors raises an exception."""
        bm = make_benchmark()

        def mock_error(texts, model_name):
            raise ConnectionError("API недоступен")

        monkeypatch.setattr(bm, "_build_vectors_openai", mock_error)

        result = bm.evaluate({"name": "broken-model", "type": "openai"})

        assert isinstance(result, BenchmarkResult), (
            "evaluate() must return BenchmarkResult even on API error"
        )
        assert result.recall_at_5 == 0.0, (
            "recall_at_5 must be 0.0 when API call fails"
        )


# ---------------------------------------------------------------------------
# Integration tests — require OPENAI_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModelBenchmarkIntegration:
    """Tests that require external services. Skip with: pytest -m 'not integration'"""

    def test_openai_evaluate_returns_nonzero_recall(self):
        """evaluate() with OpenAI API returns recall > 0 on the test set."""
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        bm = make_benchmark()
        result = bm.evaluate({"name": "text-embedding-3-small", "type": "openai"})

        assert result.recall_at_5 > 0.0, (
            "OpenAI text-embedding-3-small should find at least some correct documents"
        )
        assert result.avg_latency_sec > 0.0
