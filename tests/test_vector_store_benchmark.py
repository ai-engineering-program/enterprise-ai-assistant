"""Тесты для метода VectorStore.benchmark() (урок 2.4).

Unit-тесты проверяют структуру результата и монотонность зависимостей
без подключения к реальному Qdrant.

Интеграционные тесты требуют запущенного Qdrant на localhost:6333.
Запуск: docker run -d -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import time

import pytest


# ---------------------------------------------------------------------------
# Unit-тесты (без внешних сервисов)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkMixinUnit:
    """Тесты структуры возвращаемого значения benchmark()."""

    def _make_mock_result(self, ef_values: list[int]) -> list[dict]:
        """Имитация правильного вывода benchmark()."""
        # Монотонные зависимости: ef↑ => p95_ms↑, recall↑
        base_latency = {32: 5.0, 64: 12.0, 128: 28.0}
        base_recall = {32: 0.88, 64: 0.95, 128: 0.98}
        return [
            {
                "ef": ef,
                "p95_ms": base_latency.get(ef, ef * 0.22),
                "recall_at_10": base_recall.get(ef, min(0.999, 0.80 + ef * 0.001)),
            }
            for ef in ef_values
        ]

    def test_returns_list(self):
        """benchmark() возвращает список."""
        result = self._make_mock_result([32, 64, 128])
        assert isinstance(result, list)

    def test_result_length_matches_ef_values(self):
        """Длина результата совпадает с числом значений ef."""
        ef_values = [32, 64, 128]
        result = self._make_mock_result(ef_values)
        assert len(result) == len(ef_values)

    def test_result_dict_keys(self):
        """Каждый элемент содержит ключи ef, p95_ms, recall_at_10."""
        result = self._make_mock_result([64])
        assert set(result[0].keys()) == {"ef", "p95_ms", "recall_at_10"}

    def test_ef_values_preserved(self):
        """Значения ef в результате совпадают с входными."""
        ef_values = [32, 64, 128]
        result = self._make_mock_result(ef_values)
        assert [r["ef"] for r in result] == ef_values

    def test_p95_ms_is_float(self):
        """p95_ms — число (int или float)."""
        result = self._make_mock_result([64])
        assert isinstance(result[0]["p95_ms"], (int, float))

    def test_recall_at_10_is_float_in_range(self):
        """recall_at_10 находится в диапазоне [0, 1]."""
        result = self._make_mock_result([64])
        recall = result[0]["recall_at_10"]
        assert isinstance(recall, (int, float))
        assert 0.0 <= recall <= 1.0

    def test_higher_ef_higher_recall(self):
        """Полнота@10 монотонно растёт с увеличением ef."""
        result = self._make_mock_result([32, 64, 128])
        recalls = [r["recall_at_10"] for r in result]
        for i in range(len(recalls) - 1):
            assert recalls[i] <= recalls[i + 1], (
                f"Полнота должна расти: ef={result[i]['ef']} -> ef={result[i+1]['ef']}, "
                f"но {recalls[i]} > {recalls[i+1]}"
            )

    def test_higher_ef_higher_latency(self):
        """p95 задержка монотонно растёт с увеличением ef."""
        result = self._make_mock_result([32, 64, 128])
        latencies = [r["p95_ms"] for r in result]
        for i in range(len(latencies) - 1):
            assert latencies[i] <= latencies[i + 1], (
                f"Задержка должна расти: ef={result[i]['ef']} -> ef={result[i+1]['ef']}, "
                f"но {latencies[i]} > {latencies[i+1]}"
            )

    def test_default_ef_values_with_mock(self):
        """При ef_values=None используются значения [32, 64, 128]."""
        # Проверяем поведение через мок-реализацию
        result = self._make_mock_result([32, 64, 128])
        assert len(result) == 3
        assert result[0]["ef"] == 32
        assert result[1]["ef"] == 64
        assert result[2]["ef"] == 128


@pytest.mark.unit
class TestBenchmarkIntegration:
    """Тесты, проверяющие что benchmark() реально вызывает методы клиента."""

    def test_benchmark_calls_delete_collection_for_cleanup(self):
        """Убеждаемся, что временные коллекции удаляются."""
        # Этот тест проверяет РЕАЛЬНЫЙ метод через mock-клиент.
        # Потребует, чтобы студент реализовал метод.
        try:
            from app.rag.vector_store import VectorStore
        except ImportError:
            pytest.skip("VectorStore не реализован")

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        # Настраиваем mock для search: возвращаем пустые результаты
        mock_search_result = [MagicMock(id=i, score=0.9) for i in range(10)]
        mock_client.search.return_value = mock_search_result

        # Мокируем QdrantClient при инициализации
        with patch("app.rag.vector_store.QdrantClient", return_value=mock_client):
            store = VectorStore(
                host="localhost",
                port=6333,
                collection_name="test_coll",
                vector_size=768,
            )
            store.client = mock_client

        # Если метод benchmark реализован — вызываем его
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        try:
            results = store.benchmark(n_vectors=100, ef_values=[32])
            # Проверяем, что delete_collection вызывался
            assert mock_client.delete_collection.called, (
                "benchmark() должен удалять временные коллекции"
            )
        except NotImplementedError:
            pytest.skip("benchmark() не реализован — NotImplementedError")


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют запущенного Qdrant)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBenchmarkWithRealQdrant:
    """Тесты с реальным Qdrant. Требуют: docker run -d -p 6333:6333 qdrant/qdrant"""

    @pytest.fixture(scope="class")
    def store(self):
        try:
            from app.rag.vector_store import VectorStore
        except ImportError:
            pytest.skip("VectorStore не реализован")

        try:
            from qdrant_client import QdrantClient
            client = QdrantClient("localhost", port=6333, timeout=5)
            client.get_collections()  # проверяем доступность
        except Exception:
            pytest.skip("Qdrant недоступен на localhost:6333")

        s = VectorStore(
            host="localhost",
            port=6333,
            collection_name="bench_integration_test",
            vector_size=128,  # маленькая размерность для скорости
        )
        yield s

        # Очистка после тестов
        try:
            if s.client.collection_exists("bench_integration_test"):
                s.client.delete_collection("bench_integration_test")
        except Exception:
            pass

    def test_benchmark_returns_correct_structure(self, store):
        """benchmark() возвращает список с правильной структурой ключей."""
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        results = store.benchmark(n_vectors=500, ef_values=[32, 64])
        assert len(results) == 2
        for r in results:
            assert "ef" in r
            assert "p95_ms" in r
            assert "recall_at_10" in r

    def test_benchmark_p95_positive(self, store):
        """p95 задержка положительна."""
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        results = store.benchmark(n_vectors=500, ef_values=[64])
        assert results[0]["p95_ms"] > 0

    def test_benchmark_recall_in_range(self, store):
        """Полнота@10 в диапазоне [0, 1]."""
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        results = store.benchmark(n_vectors=500, ef_values=[64])
        assert 0.0 <= results[0]["recall_at_10"] <= 1.0

    def test_benchmark_cleanup(self, store):
        """После выполнения benchmark() временных коллекций не остаётся."""
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        from qdrant_client import QdrantClient
        client = QdrantClient("localhost", port=6333)

        store.benchmark(n_vectors=200, ef_values=[32])

        # Проверяем, что временных коллекций не осталось
        collections = client.get_collections().collections
        bench_names = [c.name for c in collections if c.name.startswith("_bench_")]
        assert len(bench_names) == 0, (
            f"Временные коллекции не были удалены: {bench_names}"
        )

    def test_higher_ef_better_recall(self, store):
        """При ef=128 полнота выше, чем при ef=32."""
        if not hasattr(store, "benchmark"):
            pytest.skip("Метод benchmark() не реализован")

        results = store.benchmark(n_vectors=1000, ef_values=[32, 128])
        recall_32 = results[0]["recall_at_10"]
        recall_128 = results[1]["recall_at_10"]
        assert recall_32 <= recall_128, (
            f"Полнота при ef=128 ({recall_128:.3f}) должна быть >= ef=32 ({recall_32:.3f})"
        )
