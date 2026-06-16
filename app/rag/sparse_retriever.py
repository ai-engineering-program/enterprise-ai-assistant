from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi


__all__ = ["BM25Retriever", "SearchResult"]


@dataclass
class SearchResult:
    """Результат поиска — фрагмент документа с оценкой релевантности."""

    text: str
    source: str
    score: float


class BM25Retriever:
    """Разреженный (sparse) поиск по корпусу документов через BM25.

    Не требует Qdrant или GPU — индекс хранится в памяти.
    Подходит для поиска по точным терминам: кодам задач YouTrack,
    артикулам 1С, аббревиатурам (ИНН, ОГРН), номерам нормативных актов.

    Пример использования:
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index_documents([
            {"text": "BACKEND-2891: rate limiting в API Gateway", "source": "yt_2891"},
            {"text": "AUTH-417: добавить 2FA", "source": "yt_417"},
        ])
        results = retriever.retrieve("BACKEND-2891", top_k=3)
        context = retriever.build_context(results)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # TODO: сохранить k1 и b как атрибуты экземпляра
        # TODO: инициализировать self._bm25 = None (BM25Okapi создаётся в index_documents)
        # TODO: инициализировать self._documents = [] (список dict с text и source)
        ...

    # ------------------------------------------------------------------
    # Токенизация
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Токенизирует текст для BM25-индекса.

        Стратегия:
        1. Убрать знаки препинания, кроме дефиса (re.sub r"[^\w\s\-]" → пробел).
        2. Привести к нижнему регистру.
        3. Разбить по пробелам.
        4. Отфильтровать пустые строки.

        Это сохраняет составные идентификаторы как единый токен:
            "BACKEND-2891" → ["backend-2891"]   (не ["backend", "2891"])
            "НК-РФ-ст-217" → ["нк-рф-ст-217"]

        Returns:
            Список токенов в нижнем регистре.

        TODO: реализуйте эту функцию.
        """
        ...

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict]) -> int:
        """Строит BM25-индекс из списка документов.

        Каждый документ — dict с ключами:
            'text' (str)   — текст фрагмента
            'source' (str) — идентификатор источника (например, "yt_2891")

        Шаги:
        1. Сохранить documents в self._documents.
        2. Токенизировать каждый документ через self.tokenize(doc["text"]).
        3. Создать self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b).

        Returns:
            Количество проиндексированных документов.

        TODO: реализуйте эту функцию.
        """
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Выполняет BM25-поиск по проиндексированному корпусу.

        Шаги:
        1. Проверить, что self._bm25 не None; иначе выбросить RuntimeError.
        2. Токенизировать query через self.tokenize(query).
        3. Вычислить оценки: self._bm25.get_scores(query_tokens) → numpy array.
        4. Создать SearchResult для каждого документа с его оценкой.
        5. Отсортировать по убыванию score.
        6. Вернуть top_k результатов с ненулевой оценкой (score > 0.0).

        Args:
            query: текст запроса пользователя.
            top_k: максимальное количество результатов.

        Returns:
            Список SearchResult, отсортированный по убыванию score.
            Документы с нулевой оценкой не включаются.

        TODO: реализуйте эту функцию.
        """
        ...

    def build_context(self, results: list[SearchResult]) -> str:
        """Форматирует результаты поиска в строку контекста для LLM.

        Формат совместим с DenseRetriever.build_context():
            [Документ 1] (source)
            text

            [Документ 2] (source)
            text

        При пустом списке вернуть строку «Релевантных документов не найдено.»

        Args:
            results: список SearchResult (обычно из retrieve()).

        Returns:
            Многострочная строка с отформатированными документами.

        TODO: реализуйте эту функцию.
        """
        ...
