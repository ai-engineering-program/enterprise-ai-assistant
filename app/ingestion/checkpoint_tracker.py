from __future__ import annotations

import hashlib
from dataclasses import dataclass


__all__ = ["IndexedChunk", "IdempotentIngestionPipeline"]


@dataclass
class IndexedChunk:
    """Запись одного проиндексированного чанка (эмулирует точку в Qdrant)."""

    key: str
    doc_id: str
    chunk_index: int
    content: str


class IdempotentIngestionPipeline:
    """
    Пайплайн поглощения документов, устойчивый к перезапуску после сбоя.

    Реализует три обязательных паттерна из урока:
    1. Идемпотентность через детерминированный уникальный ключ чанка
       (doc_id + chunk_index) — повторная запись не создаёт дубль.
    2. Чекпоинтинг с подтверждением записи — позиция чекпоинта продвигается
       только ПОСЛЕ успешной записи чанка в индекс, никогда до.
    3. Дедупликация на этапе индексирования — перед записью можно проверить,
       не проиндексирован ли данный ключ уже.
    """

    def __init__(self) -> None:
        # key -> IndexedChunk. Эмулирует upsert по детерминированному id в Qdrant.
        self._index: dict[str, IndexedChunk] = {}
        # Чекпоинт: позиция последнего ПОДТВЕРЖДЁННОГО чанка. -1 = ничего не подтверждено.
        self._last_confirmed_offset: int = -1

    def make_key(self, doc_id: str, chunk_index: int) -> str:
        """
        Построить детерминированный уникальный ключ чанка.

        TODO: вернуть sha256 hex-дайджест строки f"{doc_id}::{chunk_index}"
        (используйте .encode("utf-8") перед хэшированием).

        Один и тот же doc_id + chunk_index должны ВСЕГДА давать один и тот же
        ключ — это то, что делает повторную обработку безопасной.
        """
        ...

    def is_duplicate(self, key: str) -> bool:
        """
        Проверить, есть ли уже запись с данным ключом в индексе.

        TODO: вернуть True, если key присутствует в self._index, иначе False.
        """
        ...

    def upsert_chunk(self, doc_id: str, chunk_index: int, content: str) -> str:
        """
        Идемпотентная запись чанка в индекс.

        TODO:
        1. Вычислить key = self.make_key(doc_id, chunk_index).
        2. Записать/перезаписать self._index[key] = IndexedChunk(key, doc_id,
           chunk_index, content) — это upsert, а не append, поэтому повторный
           вызов с тем же key не создаёт вторую запись.
        3. Вернуть key.
        """
        ...

    def confirm_offset(self, offset: int) -> None:
        """
        Подтвердить чекпоинт после успешной записи чанка с данным offset.

        TODO: обновить self._last_confirmed_offset = offset, но ТОЛЬКО если
        offset > self._last_confirmed_offset — чекпоинт не должен двигаться
        назад (например, если подтверждение из более старой, "опоздавшей"
        пачки пришло после более новой).

        Вызывайте этот метод только после успешного upsert_chunk — если
        confirm_offset вызван раньше записи в индекс, при сбое между этими
        двумя шагами документ будет тихо потерян при следующем перезапуске.
        """
        ...

    def resume_offset(self) -> int:
        """
        Позиция, с которой безопасно продолжить обработку после перезапуска.

        TODO: вернуть self._last_confirmed_offset + 1.
        """
        ...

    def index_size(self) -> int:
        """Текущее количество уникальных записей в индексе (без дублей)."""
        return len(self._index)
