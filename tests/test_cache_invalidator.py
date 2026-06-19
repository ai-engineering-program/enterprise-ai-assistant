"""Тесты для CacheInvalidator (Урок 7.2).

Unit-тесты запускаются без сервисов:
    pytest tests/test_cache_invalidator.py -v -m unit

Интеграционные тесты требуют запущенный Redis:
    pytest tests/test_cache_invalidator.py -v -m integration
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from app.rag.cache_invalidator import CacheBackend, CacheIndex, CacheInvalidator


# ---------------------------------------------------------------------------
# Тестовый stub для CacheBackend
# ---------------------------------------------------------------------------

class InMemoryCacheBackend:
    """Простой словарь, имитирующий Redis для unit-тестов."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self.delete_calls: list[tuple[str, ...]] = []

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete(self, *keys: str) -> int:
        self.delete_calls.append(keys)
        deleted = sum(1 for k in keys if self._store.pop(k, None) is not None)
        return deleted

    def exists(self, key: str) -> bool:
        return key in self._store


# ---------------------------------------------------------------------------
# Unit-тесты CacheIndex
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCacheIndex:
    """Тесты индекса без внешних сервисов."""

    def test_register_single_doc(self):
        index = CacheIndex()
        index.register("key_1", ["doc_a"])
        assert "key_1" in index.get_keys_for_doc("doc_a")

    def test_register_multiple_docs(self):
        index = CacheIndex()
        index.register("key_1", ["doc_a", "doc_b"])
        assert "key_1" in index.get_keys_for_doc("doc_a")
        assert "key_1" in index.get_keys_for_doc("doc_b")

    def test_multiple_keys_for_same_doc(self):
        index = CacheIndex()
        index.register("key_1", ["doc_a"])
        index.register("key_2", ["doc_a"])
        keys = index.get_keys_for_doc("doc_a")
        assert "key_1" in keys
        assert "key_2" in keys

    def test_get_keys_returns_empty_for_unknown_doc(self):
        index = CacheIndex()
        result = index.get_keys_for_doc("nonexistent_doc")
        assert result == set()

    def test_remove_key_cleans_index(self):
        index = CacheIndex()
        index.register("key_1", ["doc_a", "doc_b"])
        index.remove_key("key_1")
        assert "key_1" not in index.get_keys_for_doc("doc_a")
        assert "key_1" not in index.get_keys_for_doc("doc_b")

    def test_remove_key_deletes_empty_doc_entries(self):
        index = CacheIndex()
        index.register("key_1", ["doc_a"])
        index.remove_key("key_1")
        # После удаления единственного ключа запись doc_a должна исчезнуть
        assert index.get_keys_for_doc("doc_a") == set()

    def test_remove_nonexistent_key_does_not_raise(self):
        index = CacheIndex()
        # Не должно поднимать исключение
        index.remove_key("key_that_never_existed")

    def test_get_keys_returns_copy(self):
        """Изменение возвращённого множества не должно влиять на индекс."""
        index = CacheIndex()
        index.register("key_1", ["doc_a"])
        keys = index.get_keys_for_doc("doc_a")
        keys.add("injected_key")
        assert "injected_key" not in index.get_keys_for_doc("doc_a")


# ---------------------------------------------------------------------------
# Unit-тесты CacheInvalidator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCacheInvalidator:
    """Тесты инвалидатора без внешних сервисов."""

    def _setup(self) -> tuple[InMemoryCacheBackend, CacheIndex, CacheInvalidator]:
        cache = InMemoryCacheBackend()
        index = CacheIndex()
        inv = CacheInvalidator(cache=cache, index=index)
        return cache, index, inv

    def test_invalidate_returns_count(self):
        cache, index, inv = self._setup()
        cache.set("key_1", "answer_1")
        cache.set("key_2", "answer_2")
        index.register("key_1", ["doc_a"])
        index.register("key_2", ["doc_a"])

        deleted = inv.invalidate("doc_a")
        assert deleted == 2

    def test_invalidate_removes_keys_from_cache(self):
        cache, index, inv = self._setup()
        cache.set("key_1", "answer")
        index.register("key_1", ["doc_a"])

        inv.invalidate("doc_a")
        assert not cache.exists("key_1")

    def test_invalidate_cleans_index(self):
        cache, index, inv = self._setup()
        cache.set("key_1", "answer")
        index.register("key_1", ["doc_a"])

        inv.invalidate("doc_a")
        assert index.get_keys_for_doc("doc_a") == set()

    def test_invalidate_unknown_doc_returns_zero(self):
        _, _, inv = self._setup()
        result = inv.invalidate("nonexistent_doc")
        assert result == 0

    def test_invalidate_unknown_doc_does_not_raise(self):
        _, _, inv = self._setup()
        inv.invalidate("ghost_doc")  # не должно поднимать исключение

    def test_key_shared_between_two_docs(self):
        """Ключ, связанный с двумя документами, удаляется при инвалидации любого."""
        cache, index, inv = self._setup()
        cache.set("key_shared", "answer")
        index.register("key_shared", ["doc_a", "doc_b"])

        deleted = inv.invalidate("doc_a")
        assert deleted == 1
        assert not cache.exists("key_shared")

    def test_invalidate_does_not_touch_unrelated_keys(self):
        cache, index, inv = self._setup()
        cache.set("key_a", "answer_a")
        cache.set("key_b", "answer_b")
        index.register("key_a", ["doc_a"])
        index.register("key_b", ["doc_b"])

        inv.invalidate("doc_a")
        assert cache.exists("key_b"), "Несвязанные ключи не должны удаляться"

    def test_invalidate_batch_single_call_to_delete(self):
        """invalidate_batch должен делать один вызов cache.delete() на весь batch."""
        cache, index, inv = self._setup()
        cache.set("key_1", "a1")
        cache.set("key_2", "a2")
        index.register("key_1", ["doc_a"])
        index.register("key_2", ["doc_b"])

        inv.invalidate_batch(["doc_a", "doc_b"])
        # Именно один вызов delete (не два отдельных)
        assert len(cache.delete_calls) == 1

    def test_invalidate_batch_returns_total_count(self):
        cache, index, inv = self._setup()
        cache.set("key_1", "a1")
        cache.set("key_2", "a2")
        cache.set("key_3", "a3")
        index.register("key_1", ["doc_a"])
        index.register("key_2", ["doc_b"])
        index.register("key_3", ["doc_b"])

        deleted = inv.invalidate_batch(["doc_a", "doc_b"])
        assert deleted == 3

    def test_invalidate_batch_empty_list_returns_zero(self):
        _, _, inv = self._setup()
        assert inv.invalidate_batch([]) == 0


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют Redis)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCacheInvalidatorIntegration:
    """Тесты с реальным Redis. Пропустить без сервиса: pytest -m 'not integration'."""

    def test_with_real_redis(self):
        pytest.skip("Требует запущенный Redis — запустите вручную: docker run -p 6379:6379 redis")
