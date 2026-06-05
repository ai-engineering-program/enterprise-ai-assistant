from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

__all__ = ["CacheEntry", "SemanticCache"]


@dataclass
class CacheEntry:
    """Запись в семантическом кеше: вектор запроса, результат и время создания."""

    embedding: list[float]
    result: list
    created_at: datetime


class SemanticCache:
    """Семантический кеш для результатов векторного поиска.

    Вместо точного совпадения строк использует косинусное сходство
    между векторными представлениями запросов. Если входящий запрос
    семантически близок к уже выполненному (сходство >= threshold),
    возвращается закешированный результат без обращения к Qdrant.

    Стратегия вытеснения: LRU (вытесняется самая давно использованная запись).
    Записи с истёкшим временем жизни (TTL) удаляются при обращении.

    Урок 2.6: реализуйте методы get(), put(), stats().
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        # Используем OrderedDict: порядок вставки = порядок вытеснения LRU
        self._cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, query_embedding: list[float], threshold: float = 0.95) -> Optional[list]:
        """Ищет семантически близкий запрос в кеше.

        TODO (Урок 2.6): реализуйте метод.
        - Переберите записи self._cache.
        - Для каждой записи проверьте TTL: если запись устарела — удалите её.
        - Вычислите косинусное сходство между query_embedding и entry.embedding
          через numpy: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        - Если сходство >= threshold: увеличьте self._hits, переместите запись
          в конец OrderedDict (move_to_end) и верните entry.result.
        - Если совпадение не найдено: увеличьте self._misses и верните None.

        Args:
            query_embedding: вектор входящего запроса.
            threshold: минимальное косинусное сходство для попадания в кеш.

        Returns:
            Список SearchResult из кеша или None при промахе.
        """
        ...

    def put(self, query_embedding: list[float], result: list) -> None:
        """Добавляет запрос и его результат в кеш.

        TODO (Урок 2.6): реализуйте метод.
        - Если len(self._cache) >= self.max_size, вытесните самую старую запись:
          self._cache.popitem(last=False)
        - Создайте CacheEntry и сохраните в self._cache по ключу
          hash(tuple(query_embedding)).

        Args:
            query_embedding: вектор запроса.
            result: список результатов для кеширования.
        """
        ...

    def stats(self) -> dict:
        """Возвращает статистику эффективности кеша.

        TODO (Урок 2.6): реализуйте метод.
        - Верните словарь с ключами:
            hits (int): количество попаданий в кеш
            misses (int): количество промахов
            size (int): текущий размер кеша
            hit_rate (float): доля попаданий от общего числа запросов (0.0 если запросов не было)

        Returns:
            Словарь {'hits': int, 'misses': int, 'size': int, 'hit_rate': float}
        """
        ...
