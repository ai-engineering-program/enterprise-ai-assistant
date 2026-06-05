"""Тесты для SemanticCache (Урок 2.6).

Unit-тесты запускаются без сервисов:
    pytest tests/test_semantic_cache.py -v -m unit

Интеграционные тесты требуют запущенный Qdrant:
    pytest tests/test_semantic_cache.py -v -m integration
"""
import math
import time

import pytest

from app.rag.semantic_cache import CacheEntry, SemanticCache


def _unit_vector(dim: int, index: int) -> list[float]:
    """Создаёт единичный вектор с 1.0 на позиции index."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


@pytest.mark.unit
class TestSemanticCacheUnit:
    """Тесты, не требующие внешних сервисов."""

    def test_miss_on_empty_cache(self):
        cache = SemanticCache()
        result = cache.get([0.1, 0.2, 0.3])
        assert result is None

    def test_put_and_exact_hit(self):
        cache = SemanticCache()
        vec = _unit_vector(8, 0)
        payload = [{"text": "результат", "score": 0.9}]
        cache.put(vec, payload)

        result = cache.get(vec, threshold=0.95)
        assert result is not None
        assert result[0]["text"] == "результат"

    def test_semantic_hit_close_vectors(self):
        """Два близких вектора должны давать попадание."""
        import numpy as np

        cache = SemanticCache()
        # Вектор A: небольшой угол от вектора B
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.999, 0.045, 0.0]  # косинусное сходство ~0.999

        payload = [{"text": "кроссовки Nike красные", "score": 0.85}]
        cache.put(vec_a, payload)

        result = cache.get(vec_b, threshold=0.95)
        assert result is not None, "Семантически близкий запрос должен попадать в кеш"

    def test_miss_orthogonal_vectors(self):
        """Ортогональные векторы (сходство = 0) не должны давать попадание."""
        cache = SemanticCache()
        vec_a = _unit_vector(4, 0)
        vec_b = _unit_vector(4, 1)

        cache.put(vec_a, [{"text": "A"}])
        result = cache.get(vec_b, threshold=0.95)
        assert result is None

    def test_hits_counter_increments(self):
        cache = SemanticCache()
        vec = _unit_vector(4, 0)
        cache.put(vec, [{"text": "test"}])

        cache.get(vec)
        cache.get(vec)

        assert cache.stats()["hits"] == 2

    def test_misses_counter_increments(self):
        cache = SemanticCache()
        vec_a = _unit_vector(4, 0)
        vec_b = _unit_vector(4, 1)
        cache.put(vec_a, [])

        cache.get(vec_b)
        cache.get(vec_b)

        assert cache.stats()["misses"] == 2

    def test_hit_rate_calculation(self):
        cache = SemanticCache()
        vec = _unit_vector(4, 0)
        cache.put(vec, [{"text": "x"}])

        cache.get(vec)    # попадание
        cache.get(vec)    # попадание
        cache.get(_unit_vector(4, 1))  # промах

        stats = cache.stats()
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_hit_rate_zero_when_no_requests(self):
        cache = SemanticCache()
        assert cache.stats()["hit_rate"] == 0.0

    def test_size_reflects_cache_content(self):
        cache = SemanticCache()
        assert cache.stats()["size"] == 0

        cache.put(_unit_vector(4, 0), [])
        assert cache.stats()["size"] == 1

        cache.put(_unit_vector(4, 1), [])
        assert cache.stats()["size"] == 2

    def test_eviction_when_max_size_exceeded(self):
        cache = SemanticCache(max_size=2)

        vec_a = _unit_vector(4, 0)
        vec_b = _unit_vector(4, 1)
        vec_c = _unit_vector(4, 2)

        cache.put(vec_a, [{"text": "A"}])
        cache.put(vec_b, [{"text": "B"}])
        cache.put(vec_c, [{"text": "C"}])

        # Размер не должен превышать max_size
        assert cache.stats()["size"] <= 2

    def test_eviction_removes_oldest_entry(self):
        cache = SemanticCache(max_size=2)

        vec_a = _unit_vector(4, 0)
        vec_b = _unit_vector(4, 1)
        vec_c = _unit_vector(4, 2)

        cache.put(vec_a, [{"text": "A"}])
        cache.put(vec_b, [{"text": "B"}])
        cache.put(vec_c, [{"text": "C"}])

        # vec_a (самая старая) должна быть вытеснена
        result_a = cache.get(vec_a, threshold=0.99)
        assert result_a is None, "Самая старая запись должна быть вытеснена"

    def test_ttl_expired_entries_not_returned(self):
        cache = SemanticCache(ttl_seconds=0)  # TTL = 0 секунд → мгновенное устаревание
        vec = _unit_vector(4, 0)
        cache.put(vec, [{"text": "устаревший"}])

        # Запись уже устарела (ttl=0 секунд)
        time.sleep(0.01)
        result = cache.get(vec, threshold=0.99)
        assert result is None, "Запись с истёкшим TTL не должна возвращаться"

    def test_stats_returns_correct_keys(self):
        cache = SemanticCache()
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "hit_rate" in stats


@pytest.mark.integration
class TestSemanticCacheIntegration:
    """Тесты с реальным Qdrant. Пропустить без сервиса: pytest -m 'not integration'."""

    def test_cache_reduces_qdrant_calls(self):
        pytest.skip("Требует запущенный Qdrant — запустите вручную")
