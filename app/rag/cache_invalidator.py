from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["CacheBackend", "CacheIndex", "CacheInvalidator"]


class CacheBackend(Protocol):
    """Интерфейс кэш-хранилища, используемый при инвалидации.

    Минимальный контракт: метод delete принимает один или несколько ключей
    и возвращает число фактически удалённых записей.
    """

    def delete(self, *keys: str) -> int:
        ...


@dataclass
class CacheIndex:
    """Обратный индекс: документ → множество кэш-ключей.

    Позволяет по doc_id найти все кэш-записи, в формировании
    которых участвовал этот документ, и передать их на удаление.

    Урок 7.2: реализуйте методы register(), get_keys_for_doc(), remove_key().
    """

    _index: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)

    def register(self, cache_key: str, source_doc_ids: list[str]) -> None:
        """Регистрирует связь между кэш-ключом и исходными документами.

        TODO: для каждого doc_id из source_doc_ids добавьте cache_key
        в self._index[doc_id].  Если doc_id ещё нет в индексе —
        создайте новое множество.

        Args:
            cache_key: уникальный ключ записи в кэш-хранилище.
            source_doc_ids: список идентификаторов документов, из которых
                был сформирован ответ, хранящийся под cache_key.
        """
        ...

    def get_keys_for_doc(self, doc_id: str) -> set[str]:
        """Возвращает все кэш-ключи, связанные с данным документом.

        TODO: вернуть копию множества self._index.get(doc_id).
        Если doc_id отсутствует в индексе — вернуть пустое множество.

        Args:
            doc_id: идентификатор документа в базе знаний.

        Returns:
            Множество строк-ключей кэша (может быть пустым).
        """
        ...

    def remove_key(self, cache_key: str) -> None:
        """Удаляет cache_key из всех записей индекса.

        TODO:
        - Пройдите по self._index и удалите cache_key из каждого множества.
        - Удалите записи doc_id, чьи множества стали пустыми.

        Args:
            cache_key: ключ, который нужно убрать из индекса.
        """
        ...


class CacheInvalidator:
    """Инвалидирует записи кэша при обновлении документов в базе знаний.

    Использует CacheIndex для поиска затронутых кэш-ключей и CacheBackend
    для их удаления.  После удаления индекс очищается от устаревших ссылок.

    Пример использования:
        index = CacheIndex()
        invalidator = CacheInvalidator(cache=redis_client, index=index)

        # При сохранении ответа в кэш:
        index.register(cache_key, source_doc_ids=["doc_42", "doc_17"])

        # При обновлении документа:
        deleted = invalidator.invalidate("doc_42")

    Урок 7.2: реализуйте методы invalidate() и invalidate_batch().
    """

    def __init__(self, cache: CacheBackend, index: CacheIndex) -> None:
        self.cache = cache
        self.index = index

    def invalidate(self, doc_id: str) -> int:
        """Инвалидирует все кэш-записи, связанные с документом.

        TODO:
        1. Получить множество ключей через self.index.get_keys_for_doc(doc_id).
        2. Если множество пусто — вернуть 0.
        3. Вызвать self.cache.delete(*keys) и получить число удалённых.
        4. Для каждого удалённого ключа вызвать self.index.remove_key(key).
        5. Вернуть число удалённых записей.

        Args:
            doc_id: идентификатор обновлённого или удалённого документа.

        Returns:
            Число кэш-записей, фактически удалённых из хранилища.
        """
        ...

    def invalidate_batch(self, doc_ids: list[str]) -> int:
        """Инвалидирует кэш для нескольких документов одним вызовом.

        TODO:
        - Собрать union всех ключей для каждого doc_id из doc_ids.
        - Если итоговое множество пусто — вернуть 0.
        - Удалить все ключи одним вызовом self.cache.delete(*all_keys).
        - Очистить индекс для каждого удалённого ключа.
        - Вернуть общее число удалённых записей.

        Важно: один вызов cache.delete() на весь batch, не по одному на doc_id.

        Args:
            doc_ids: список идентификаторов документов для инвалидации.

        Returns:
            Суммарное число удалённых кэш-записей.
        """
        ...
