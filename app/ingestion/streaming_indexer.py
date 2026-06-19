import hashlib
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable, Optional


@dataclass
class DocumentEvent:
    """Уведомление о событии изменения документа."""
    event_id: str
    doc_id: str
    content: str
    source: str           # например "tass", "ria", "internal_cms"
    published_at: datetime
    event_type: str       # "created" | "updated" | "deleted"


class StreamingIndexer:
    """
    Потоковый индексатор документов с дедупликацией по хэшу содержимого.

    Получает уведомления о событиях (DocumentEvent) и записывает изменения
    в Qdrant через операцию upsert. Документы с неизменившимся содержимым
    пропускаются без повторной векторизации.

    В production embed_fn — это вызов реальной модели (sentence-transformers,
    OpenAI embeddings API и т.п.). В тестах можно передать заглушку.
    """

    def __init__(
        self,
        collection_name: str,
        embed_fn: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    ):
        self._collection = collection_name
        self._embed_fn = embed_fn
        # doc_id -> SHA-256 хэш последнего проиндексированного содержимого
        self._hashes: dict[str, str] = {}
        # Замеры задержки индексации (от published_at до завершения upsert)
        self._lag_samples: list[float] = []
        self._skip_count: int = 0
        self._upsert_count: int = 0

    def compute_hash(self, content: str) -> str:
        """
        Вычислить SHA-256 хэш от строки content.

        Возвращает hex-строку длиной 64 символа.

        TODO: используйте hashlib.sha256, закодируйте content в UTF-8 перед хэшированием.
        """
        # TODO: реализуйте вычисление SHA-256 хэша
        ...

    def is_duplicate(self, doc_id: str, content: str) -> bool:
        """
        Проверить, является ли документ дубликатом (содержимое не изменилось).

        Возвращает True, если doc_id уже есть в self._hashes
        и сохранённый хэш совпадает с compute_hash(content).

        TODO: сравните compute_hash(content) с self._hashes.get(doc_id).
        """
        # TODO: реализуйте проверку дублирования
        ...

    async def upsert_document(self, event: DocumentEvent) -> None:
        """
        Векторизовать документ и выполнить upsert в Qdrant.

        Шаги:
        1. Если self._embed_fn задан — получить вектор через await self._embed_fn(event.content).
           Иначе использовать вектор-заглушку [0.0, 0.0, 0.0, 0.0].
        2. (В реальном коде здесь был бы вызов qdrant_client.upsert — в скелете пропускаем.)
        3. Сохранить хэш содержимого в self._hashes[event.doc_id].
        4. Увеличить self._upsert_count на 1.
        5. Вычислить задержку как разницу между datetime.utcnow() и event.published_at
           (в секундах), добавить в self._lag_samples.

        TODO: реализуйте все пять шагов.
        """
        # TODO: получить вектор (или использовать заглушку)
        # TODO: сохранить хэш в self._hashes
        # TODO: увеличить self._upsert_count
        # TODO: вычислить lag и добавить в self._lag_samples
        ...

    async def process_event(self, event: DocumentEvent) -> bool:
        """
        Обработать уведомление о событии.

        Логика:
        - Если event_type == "deleted": удалить doc_id из self._hashes,
          вернуть True.
        - Если is_duplicate(doc_id, content): увеличить self._skip_count,
          вернуть False.
        - Иначе: вызвать await upsert_document(event), вернуть True.

        Возвращает True, если документ был добавлен/обновлён/удалён.
        Возвращает False, если документ был пропущен как дубликат.

        TODO: реализуйте всю логику ветвления.
        """
        # TODO: обработать удаление
        # TODO: проверить дублирование
        # TODO: вызвать upsert_document
        ...

    def stats(self) -> dict:
        """
        Вернуть статистику работы индексатора.

        Returns:
            dict с ключами:
                upsert_count     — количество выполненных upsert
                skip_count       — количество пропущенных дубликатов
                avg_lag_seconds  — средняя задержка индексации (None если нет данных)
        """
        avg_lag: Optional[float] = (
            sum(self._lag_samples) / len(self._lag_samples)
            if self._lag_samples
            else None
        )
        return {
            "upsert_count": self._upsert_count,
            "skip_count": self._skip_count,
            "avg_lag_seconds": avg_lag,
        }


__all__ = ["DocumentEvent", "StreamingIndexer"]
