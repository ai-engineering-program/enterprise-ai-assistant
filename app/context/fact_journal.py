from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


__all__ = [
    "FactVersion",
    "ClockFn",
    "FactJournal",
]


# Функция, возвращающая текущее время. Тот же принцип dependency injection,
# что и ClockFn у SessionMemory (5.1), LongTermMemoryStore (5.2) и
# EpisodicMemoryStore (5.3) — без инъекции часов протестировать "какая
# версия факта побеждает по timestamp" можно было бы только реальным
# ожиданием, что неприемлемо для unit-тестов.
ClockFn = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class FactVersion:
    """
    Одна неизменяемая запись в журнале факта — либо значение, действующее
    начиная с recorded_at, либо (если oblivion=True) отметка о том, что
    начиная с recorded_at у этого fact_key больше нет актуального значения
    ("логическое удаление", см. текст урока 5.4).

    Запись НИКОГДА не редактируется и не удаляется после создания —
    append-only, тот же принцип, что и Fact (5.2) и Episode (5.3). Отличие
    от них: FactVersion не привязана к конкретному user_id одного диалога —
    это универсальная единица журнала для ЛЮБОЙ волатильной характеристики
    любой сущности, проиндексированной в контекстном слое (карточка
    клиента, профиль сотрудника, статус тикета и т.д.).
    """

    fact_key: str
    value: Optional[str]
    recorded_at: datetime = field(default_factory=_utc_now)
    oblivion: bool = False
    source: Optional[str] = None


class FactJournal:
    """
    Модуль M5, урок 5.4: паттерн fact_key + journal — фикс для класса
    ошибок "конфликт временных сущностей" (см. текст урока: инцидент, где
    обе версии факта — прежний и новый персональный менеджер клиента —
    полностью проиндексированы и почти неотличимы по embedding, и
    similarity-поиск не может выбрать актуальную).

    В отличие от LongTermMemoryStore (5.2), который хранит append-with-
    timestamp факты ОДНОГО клиента для памяти диалога, FactJournal — более
    общий примитив контекстного слоя: fact_key может указывать на ЛЮБУЮ
    волатильную характеристику любой сущности из общей базы знаний, а не
    только на факт о конкретном собеседнике.

    Эта реализация — in-process backend (см. урок 5.1, раздел про
    backend'ы), но метод отбора спроектирован так, как он работал бы в
    реальном векторном сторе (см. текст урока, implementation sketch):
    в Qdrant-подобном хранилище append_version() соответствует upsert
    точки с payload {fact_key, recorded_at, oblivion}, а get_current() —
    payload-фильтру по fact_key (scroll/filter, БЕЗ векторного поиска —
    сущность уже известна) с последующей сортировкой результатов на
    стороне клиента по recorded_at. Публичный интерфейс класса не
    меняется независимо от backend'а.
    """

    def __init__(self, clock: ClockFn = _utc_now) -> None:
        self._clock = clock
        # fact_key -> список FactVersion в порядке добавления. Append-only:
        # ни один метод этого класса не удаляет и не изменяет существующие
        # записи, только добавляет новые (в том числе oblivion-маркеры).
        self._versions: dict[str, list[FactVersion]] = {}

    def append_version(
        self,
        fact_key: str,
        value: str,
        source: Optional[str] = None,
    ) -> FactVersion:
        """
        Зафиксировать новое значение факта как отдельную неизменяемую
        запись журнала (НЕ заменяет и не удаляет предыдущие версии).

        TODO:
        1. Создать version = FactVersion(fact_key=fact_key, value=value,
           recorded_at=self._clock(), oblivion=False, source=source).
        2. Добавить version в конец self._versions.setdefault(fact_key, [])
           (append — существующие записи не трогать).
        3. Вернуть version.
        """
        ...

    def mark_oblivion(
        self,
        fact_key: str,
        source: Optional[str] = None,
    ) -> FactVersion:
        """
        Зафиксировать логическое удаление: начиная с текущего момента у
        fact_key больше нет актуального значения (например, клиент ушёл,
        сотрудник уволился, тикет закрыт без замены). Это ТОЖЕ append-only
        запись, а не физическое удаление предыдущих версий — полная
        история остаётся доступна через get_history() (см. текст урока,
        раздел про журнал).

        TODO:
        1. Создать version = FactVersion(fact_key=fact_key, value=None,
           recorded_at=self._clock(), oblivion=True, source=source).
        2. Добавить version в конец self._versions.setdefault(fact_key, [])
           (append, как и в append_version).
        3. Вернуть version.
        """
        ...

    def get_current(self, fact_key: str) -> Optional[FactVersion]:
        """
        Правило выбора актуальной версии (см. текст урока, раздел
        "правило выбора"): среди ВСЕХ записей данного fact_key (включая
        oblivion-маркеры) найти запись с максимальным recorded_at — она и
        определяет текущее состояние факта.

        TODO:
        1. versions = self._versions.get(fact_key, []). Если пусто —
           вернуть None.
        2. latest = запись с максимальным recorded_at; при равенстве
           recorded_at у нескольких записей — та, что стоит ПОСЛЕДНЕЙ в
           списке (добавлена позже по порядку append).
        3. Если latest.oblivion is True — вернуть None (факт логически
           удалён, актуального значения нет).
        4. Иначе вернуть latest (это и есть текущая актуальная версия).

        ВАЖНО: правило работает по МАКСИМУМУ timestamp среди ВСЕХ записей,
        а не по последней НЕ-oblivion записи — если oblivion-маркер новее
        последнего значения, актуального значения нет, даже если где-то
        раньше в истории лежит валидное значение.
        """
        ...

    def get_history(self, fact_key: str) -> list[FactVersion]:
        """
        Вернуть ПОЛНУЮ историю версий факта (включая oblivion-маркеры) в
        хронологическом порядке добавления — используется для аудита и
        отладки конфликтов версий (см. get_fact_history в
        LongTermMemoryStore, 5.2, — тот же принцип, применённый к более
        общему journal pattern).

        TODO: вернуть list(self._versions.get(fact_key, [])) — именно
        КОПИЮ списка (через list(...)), чтобы изменение возвращённого
        списка вызывающим кодом не повредило внутреннее состояние журнала.
        """
        ...
