from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


__all__ = ["SourceType", "SourceProfile", "SourceRegistry"]


class SourceType(Enum):
    """Тип системы-источника контекста."""

    SLACK = "slack"
    CONFLUENCE = "confluence"
    JIRA = "jira"
    OTHER = "other"


@dataclass
class SourceProfile:
    """
    Профиль одного зарегистрированного источника контекста.

    authority_weight — вес авторитетности источника в диапазоне [0.0, 1.0].
    Это НЕ окончательное решение о доверии — это лишь один из сигналов,
    который в модуле M2 будет комбинироваться со свежестью и типом факта.

    volatility — насколько часто меняется контент такого рода источника:
    "low" | "medium" | "high".
    """

    name: str
    source_type: SourceType
    authority_weight: float
    volatility: str


class SourceRegistry:
    """
    Реестр источников контекста — фундамент для будущей иерархии доверия.

    Этот класс сознательно НЕ разрешает конфликты между источниками
    с учётом свежести факта (recency) — это тема модуля M2 "Инжиниринг
    качества источников". Здесь закладывается только словарь: как описать
    источник и сравнить источники по чистой авторитетности.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SourceProfile] = {}

    def register(
        self,
        name: str,
        source_type: SourceType,
        authority_weight: float,
        volatility: str,
    ) -> SourceProfile:
        """
        Зарегистрировать источник контекста.

        TODO:
        1. Если authority_weight не в диапазоне [0.0, 1.0] — поднять
           ValueError с понятным сообщением.
        2. Создать SourceProfile(name, source_type, authority_weight,
           volatility) и сохранить в self._profiles[name].
        3. Вернуть созданный профиль.
        """
        ...

    def get(self, name: str) -> SourceProfile | None:
        """
        Вернуть профиль источника по имени.

        TODO: вернуть self._profiles.get(name).
        """
        ...

    def rank_by_authority(self) -> list[str]:
        """
        Ранжировать все зарегистрированные источники по авторитетности.

        TODO: вернуть список имён источников, отсортированный по
        authority_weight по убыванию. При равенстве весов — сортировать
        по имени в алфавитном порядке, чтобы результат был детерминирован.
        """
        ...

    def resolve_conflict(self, name_a: str, name_b: str) -> str:
        """
        Разрешить конфликт между двумя источниками по чистой авторитетности.

        TODO:
        1. Получить профили name_a и name_b из self._profiles. Если хотя бы
           один не найден — поднять KeyError с именем отсутствующего
           источника.
        2. Если authority_weight отличаются — вернуть имя источника с
           более высоким весом.
        3. Если веса равны — поднять ValueError: одной авторитетности
           недостаточно для разрешения конфликта (осознанное ограничение
           этого урока; учёт свежести появится в модуле M2).
        """
        ...
