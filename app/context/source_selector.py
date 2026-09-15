from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Pattern, Union

from app.context.source_registry import SourceType
from app.context.truth_axis_router import TruthAxis, TruthAxisRouter


__all__ = [
    "DEFAULT_TICKET_PATTERN",
    "DEFAULT_FALLBACK_SOURCES",
    "SourceSelectionResult",
    "SourceSelector",
]


# Сигнал "упоминание номера тикета" — самый специфичный из всех: если
# запрос буквально называет идентификатор тикета (например, "SUPPORT-8842"
# или "ATLAS-231"), сомнений в источнике нет вообще — это Jira, и никакой
# более общий сигнал (ось правды, ключевые слова) не должен его перебивать.
DEFAULT_TICKET_PATTERN: str = r"\b[A-ZА-Я]{2,10}-\d{2,6}\b"

# Маркеры регламентной/спецификационной лексики — сигнал того же рода, что
# и DESIGN_INTENT в TruthAxisRouter (урок 2.4), но выраженный явными
# ключевыми словами, а не осью правды целиком.
_DEFAULT_CONFLUENCE_KEYWORDS: tuple[str, ...] = (
    "регламент",
    "политика",
    "инструкция",
    "спецификация",
    "согласно документу",
)

# Маркеры переписки — запрос явно ссылается на обсуждение, а не на
# формальный документ или тикет.
_DEFAULT_SLACK_KEYWORDS: tuple[str, ...] = (
    "в чате",
    "тред",
    "обсуждали",
    "переписка",
    "в канале",
)

# Источники, которые опрашиваются, когда ни специфичный сигнал, ни ось
# правды не дали уверенного ответа — сознательный, редкий fallback, а не
# поведение по умолчанию для каждого запроса (см. инцидент этого урока).
DEFAULT_FALLBACK_SOURCES: tuple[SourceType, ...] = (
    SourceType.CONFLUENCE,
    SourceType.JIRA,
    SourceType.SLACK,
)


@dataclass
class SourceSelectionResult:
    """
    Итог решения "какие источники вообще опрашивать" для ОДНОГО
    (под)запроса — ДО того, как retrieval выполнит хоть один вызов к
    векторной базе или API коннектора.

    entity_hint — сработавший специфичный сигнал (номер тикета или
    ключевое слово), если он был найден; None, если запрос пришлось
    классифицировать по оси правды или отправить в fallback.

    used_fallback — True, только если ни entity_hint, ни ось правды не
    дали уверенного сигнала, и SourceSelector сознательно вернул широкий
    список источников по умолчанию вместо того, чтобы угадывать один.
    """

    query: str
    selected_sources: list[SourceType]
    entity_hint: Optional[SourceType]
    axis: TruthAxis
    used_fallback: bool
    notes: list[str] = field(default_factory=list)


class SourceSelector:
    """
    Решает, какие типы источников вообще стоит опрашивать для конкретного
    (под)запроса — ДО retrieval, а не как переранжировать приоритет уже
    найденных кандидатов (это делает KnowledgeHierarchyResolver, урок 2.5,
    ПОСЛЕ того, как retrieval уже отработал по каждому выбранному
    источнику). Это два последовательных, независимых рубежа одного
    планировщика retrieval (модуль M3), а не одна и та же задача, решённая
    дважды: селекция источников экономит вызовы и латентность, ранжирование
    повышает точность приоритета уже найденного.

    Три сигнала проверяются строго в порядке убывания специфичности:
    1. entity/keyword hint — самый специфичный сигнал (regex номера
       тикета, ключевые слова регламента/переписки). Конкретный
       идентификатор объекта строго специфичнее общего класса вопроса,
       к которому этот объект относится, поэтому проверяется первым и
       побеждает при конфликте с любым другим сигналом.
    2. ось правды запроса — TruthAxisRouter (урок 2.4), переданный снаружи
       через конструктор (dependency injection, тот же принцип, что и у
       KnowledgeHierarchyResolver).
    3. fallback — сознательно широкий список источников, когда первые два
       сигнала не дали уверенного ответа. Явно помечается used_fallback,
       чтобы отличаться от неявного "broadcast всегда" из инцидента урока.

    SourceSelector НЕ реализует TruthAxisRouter заново — он его
    переиспользует как готовый компонент, тем же способом, каким его уже
    переиспользует KnowledgeHierarchyResolver.
    """

    def __init__(
        self,
        truth_axis_router: Optional[TruthAxisRouter] = None,
        ticket_pattern: Union[str, Pattern[str]] = DEFAULT_TICKET_PATTERN,
        confluence_keywords: Optional[Iterable[str]] = None,
        slack_keywords: Optional[Iterable[str]] = None,
        fallback_sources: Optional[Iterable[SourceType]] = None,
    ) -> None:
        self._truth_axis_router = truth_axis_router
        self._ticket_pattern: Pattern[str] = (
            re.compile(ticket_pattern) if isinstance(ticket_pattern, str) else ticket_pattern
        )
        self._confluence_keywords: tuple[str, ...] = tuple(
            confluence_keywords
            if confluence_keywords is not None
            else _DEFAULT_CONFLUENCE_KEYWORDS
        )
        self._slack_keywords: tuple[str, ...] = tuple(
            slack_keywords if slack_keywords is not None else _DEFAULT_SLACK_KEYWORDS
        )
        self._fallback_sources: tuple[SourceType, ...] = tuple(
            fallback_sources if fallback_sources is not None else DEFAULT_FALLBACK_SOURCES
        )

    def find_entity_hint(self, query: str) -> Optional[SourceType]:
        """
        Найти самый специфичный сигнал — номер тикета или ключевое слово,
        однозначно указывающее на конкретный источник.

        TODO:
        1. Если self._ticket_pattern.search(query) находит совпадение —
           вернуть SourceType.JIRA (номер тикета — самый сильный сигнал,
           проверяется первым, раньше ключевых слов).
        2. lower = query.lower(). Если хотя бы одно ключевое слово из
           self._confluence_keywords встречается в lower как подстрока —
           вернуть SourceType.CONFLUENCE.
        3. Если хотя бы одно ключевое слово из self._slack_keywords
           встречается в lower — вернуть SourceType.SLACK.
        4. Иначе вернуть None — специфичного сигнала нет.
        """
        ...

    def classify_intent(self, query: str) -> TruthAxis:
        """
        Определить ось правды запроса, переиспользуя TruthAxisRouter
        (урок 2.4) как готовый компонент.

        TODO: если self._truth_axis_router is None — вернуть
        TruthAxis.AMBIGUOUS (без переданного router'а нет способа
        определить ось — это честное "сигнала нет", а не ошибка).
        Иначе вернуть self._truth_axis_router.classify_axis(query).
        """
        ...

    def select(self, query: str) -> SourceSelectionResult:
        """
        Главный метод: решить, какие источники опрашивать для query.

        TODO:
        1. entity_hint = self.find_entity_hint(query)
        2. axis = self.classify_intent(query)
        3. Если entity_hint is not None:
           selected = [entity_hint]; used_fallback = False
           notes = ["специфичный сигнал (тикет/ключевое слово) — единственный источник"]
        4. Иначе если axis is TruthAxis.DESIGN_INTENT:
           selected = [SourceType.CONFLUENCE]; used_fallback = False
           notes = ["ось правды: design intent — рекомендован Confluence"]
        5. Иначе если axis is TruthAxis.OPERATIONAL_STATUS:
           selected = [SourceType.JIRA]; used_fallback = False
           notes = ["ось правды: operational status — рекомендована Jira"]
        6. Иначе (axis is TruthAxis.AMBIGUOUS и entity_hint is None):
           selected = list(self._fallback_sources); used_fallback = True
           notes = ["сигналы не дали уверенного ответа — сознательный broad fallback"]
        7. Вернуть SourceSelectionResult(query=query,
           selected_sources=selected, entity_hint=entity_hint, axis=axis,
           used_fallback=used_fallback, notes=notes).
        """
        ...
