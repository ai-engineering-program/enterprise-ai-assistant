from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


__all__ = ["DenseRetriever", "SearchResult"]


@dataclass
class SearchResult:
    """Результат поиска — фрагмент документа с оценкой релевантности."""

    text: str
    source: str
    score: float


class DenseRetriever:
    """Плотный (dense) поиск по векторному индексу в Qdrant.

    Преобразует запрос в вектор через SentenceTransformer,
    выполняет поиск ближайших соседей через HNSW и возвращает top-K фрагментов.

    Пример использования:
        retriever = DenseRetriever(collection_name="my_docs")
        retriever.index_documents([{"text": "...", "source": "doc.pdf"}])
        results = retriever.retrieve("мой запрос", top_k=5)
        context = retriever.build_context(results)
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-base"

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
        model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.model_name = model_name

        self._client = QdrantClient(url=qdrant_url)
        self._model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Публичный интерфейс — реализуйте эти методы
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Преобразует текст запроса в вектор.

        Для семейства multilingual-e5 запросы нужно префиксировать «query: »,
        документы при индексировании — «passage: ».
        Используйте normalize_embeddings=True для нормализации к единичной длине.

        Returns:
            Нормализованный вектор в виде list[float].

        TODO: реализуйте эту функцию.
        """
        ...

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """Выполняет поиск и возвращает top_k релевантных фрагментов.

        Шаги:
        1. Вызовите embed_query(query) для получения вектора запроса.
        2. Выполните self._client.search() с параметром with_payload=True.
        3. Преобразуйте результаты в список SearchResult.

        Args:
            query: текст запроса пользователя.
            top_k: количество результатов. Если None — использует self.top_k.

        Returns:
            Список SearchResult, отсортированный по убыванию score.

        TODO: реализуйте эту функцию.
        """
        ...

    def build_context(self, results: list[SearchResult]) -> str:
        """Форматирует результаты поиска в строку контекста для LLM.

        Формат каждого документа:
            [Документ N] (source)
            text

        Документы разделяются двойным переносом строки.
        Если results пуст, вернуть строку «Релевантных документов не найдено.»

        Args:
            results: список SearchResult (обычно из retrieve()).

        Returns:
            Многострочная строка с отформатированными документами.

        TODO: реализуйте эту функцию.
        """
        ...

    # ------------------------------------------------------------------
    # Утилиты для индексирования
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Создаёт коллекцию в Qdrant, если она не существует.

        Используйте self._client.get_collections() для проверки существования
        и self._client.create_collection() для создания с Distance.COSINE.
        Размерность вектора: self._model.get_sentence_embedding_dimension().

        TODO: реализуйте эту функцию.
        """
        ...

    def index_documents(self, documents: list[dict]) -> int:
        """Индексирует список документов в Qdrant.

        Каждый документ — словарь с ключами:
            'text' (str) — текст фрагмента
            'source' (str) — идентификатор источника

        При векторизации добавляйте префикс «passage: » к тексту.
        Сохраняйте оригинальный текст (без префикса) в payload.

        Args:
            documents: список документов для индексирования.

        Returns:
            Количество успешно добавленных точек.

        TODO: реализуйте эту функцию.
        """
        ...
