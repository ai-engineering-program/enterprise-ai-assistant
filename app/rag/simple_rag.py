from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
GENERATION_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

__all__ = ["Document", "SearchResult", "SimpleRAG"]


@dataclass
class Document:
    """Входной документ с текстом и источником."""
    text: str
    source: str = ""


@dataclass
class SearchResult:
    """Результат поиска: фрагмент, источник и оценка релевантности."""
    text: str
    source: str
    score: float


class SimpleRAG:
    """Минимальный рабочий RAG-пайплайн (Урок 1.5).

    Намеренно упрощённая версия — точка отсчёта для всего курса.
    Каждый последующий модуль улучшает один из компонентов.

    Методы:
        index_documents(docs)  — шаги 1-4: загрузка, разбивка, векторные представления, хранение
        search(query, top_k)   — шаг 5: поиск по косинусному сходству
        answer(query, top_k)   — шаги 5-6: поиск + генерация ответа
    """

    def __init__(
        self,
        collection_name: str = "documents",
        qdrant_url: str = "http://localhost:6333",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant = QdrantClient(url=qdrant_url)
        self.openai = OpenAI(api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию в Qdrant если она ещё не существует.

        TODO (Урок 1.5): этот метод уже реализован — изучите как создаётся коллекция.
        """
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def _split_into_chunks(self, text: str, source: str) -> list[dict]:
        """Разбивает текст на фрагменты фиксированного размера с перекрытием.

        TODO (Урок 1.5): реализуйте метод.
        - Размер фрагмента: DEFAULT_CHUNK_SIZE символов
        - Перекрытие: DEFAULT_CHUNK_OVERLAP символов
        - Каждый элемент результата — словарь с ключами "text", "source", "chunk_id"
        """
        ...

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Строит векторные представления для списка текстов через OpenAI API.

        TODO (Урок 1.5): реализуйте метод.
        - Используйте self.openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        - Верните список векторов: [item.embedding for item in response.data]
        """
        ...

    def index_documents(self, documents: list[Document]) -> int:
        """Разбивает документы на фрагменты, строит представления и сохраняет в Qdrant.

        TODO (Урок 1.5): реализуйте метод.
        - Для каждого документа вызовите _split_into_chunks
        - Постройте векторные представления для всех фрагментов через _embed
        - Создайте список PointStruct и сохраните в Qdrant через self.qdrant.upsert
        - Верните количество сохранённых фрагментов

        Returns:
            Количество проиндексированных фрагментов.
        """
        ...

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Ищет top_k наиболее релевантных фрагментов для запроса.

        TODO (Урок 1.5): реализуйте метод.
        - Преобразуйте запрос в вектор через _embed([query])[0]
        - Выполните поиск через self.qdrant.search(... limit=top_k, with_payload=True)
        - Верните список SearchResult из полей hit.payload и hit.score

        Returns:
            Список SearchResult, отсортированный по score убыванию.
        """
        ...

    def answer(self, query: str, top_k: int = 5) -> str:
        """Выполняет полный RAG-пайплайн: поиск + генерация ответа.

        TODO (Урок 1.5): реализуйте метод.
        - Найдите релевантные фрагменты через search(query, top_k)
        - Если результатов нет — верните "Не найдено релевантных документов для ответа на вопрос."
        - Сформируйте контекст: пронумерованный список фрагментов с указанием источника
        - Вызовите self.openai.chat.completions.create с промптом:
            system: отвечать только из контекста, сообщить если ответа нет
            user: "Контекст:\n{context}\n\nВопрос: {query}"
        - temperature=0.1 для более фактичных ответов

        Returns:
            Строка с ответом языковой модели.
        """
        ...
