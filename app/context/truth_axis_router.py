from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from app.context.source_registry import SourceType


__all__ = [
    "TruthAxis",
    "QueryTruthClassification",
    "TruthAxisRouter",
]


class TruthAxis(Enum):
    """
    Тип правды, о котором спрашивает запрос — НЕ авторитетность источника
    (урок 1.1), НЕ форма текста (урок 2.1) и НЕ жизненный цикл конкретной
    страницы (урок 2.3). Это четвёртая, независимая ось: какого рода
    факт вообще нужен, прежде чем выбирать, какому источнику доверять.
    """

    # "Каким должна быть система" — целевое, спроектированное состояние.
    # Типичный источник: Confluence (спецификации, целевые архитектуры).
    DESIGN_INTENT = "design_intent"

    # "Что реально происходит с системой прямо сейчас". Типичный
    # источник: Jira (статус тикета, история изменений полей).
    OPERATIONAL_STATUS = "operational_status"

    # Запрос содержит маркеры ОБЕИХ осей одновременно, либо не содержит
    # ни одного — самонадеянно выбирать источник в этом случае нельзя.
    AMBIGUOUS = "ambiguous"


# Маркеры дизайн-намерения: вопрос про то, каким состояние должно быть
# по спецификации/архитектуре/регламенту, а не про текущий факт.
_DEFAULT_INTENT_KEYWORDS: tuple[str, ...] = (
    "должно быть",
    "по дизайну",
    "целевая архитектура",
    "спецификация",
    "согласно регламенту",
)

# Маркеры операционного статуса: вопрос про реальное состояние системы
# в конкретный момент времени, обычно "сейчас".
_DEFAULT_STATUS_KEYWORDS: tuple[str, ...] = (
    "сейчас",
    "текущий статус",
    "статус тикета",
    "на данный момент",
    "по состоянию на",
    "выполнено ли",
)


@dataclass
class QueryTruthClassification:
    """Результат классификации одного запроса по оси типа правды."""

    query: str
    axis: TruthAxis
    matched_intent_keywords: list[str]
    matched_status_keywords: list[str]
    recommended_source_type: SourceType | None


class TruthAxisRouter:
    """
    Определяет, какой тип правды спрашивает пользователь — дизайн-
    намерение (Confluence: "каким должно быть") или операционный статус
    (Jira: "что происходит сейчас") — и рекомендует, какому классу
    источников отдать приоритет ДО того, как в игру вступят авторитетность
    (SourceRegistry, урок 1.1), структура (SourceStructureClassifier,
    урок 2.1) и доверие к конкретной странице (PageTrustScorer, урок 2.3).

    Это не замена и не дублирование перечисленных инструментов — это
    более ранний фильтр. Инцидент урока 2.4 произошёл именно потому, что
    страница Confluence прошла бы все три существующие проверки с высоким
    баллом, но отвечала не на тот тип вопроса, который был задан.

    Классификатор намеренно простой — совпадение подстрок по спискам
    ключевых маркеров, без морфологического анализа. Это осознанный
    компромисс курса: детерминированность и объяснимость важнее полноты
    словаря (та же философия, что у TextStructureScorer, урок 2.1, и
    PageTrustScorer, урок 2.3).
    """

    def __init__(
        self,
        intent_keywords: Iterable[str] | None = None,
        status_keywords: Iterable[str] | None = None,
    ) -> None:
        self._intent_keywords: tuple[str, ...] = tuple(
            intent_keywords if intent_keywords is not None else _DEFAULT_INTENT_KEYWORDS
        )
        self._status_keywords: tuple[str, ...] = tuple(
            status_keywords if status_keywords is not None else _DEFAULT_STATUS_KEYWORDS
        )

    def find_matches(self, query: str, keywords: Iterable[str]) -> list[str]:
        """
        Найти, какие из keywords встречаются в query как подстроки
        (без учёта регистра).

        TODO:
        1. Привести query к нижнему регистру (query.lower()).
        2. Пройтись по keywords В ТОМ ЖЕ ПОРЯДКЕ, в котором они переданы,
           и для каждого keyword проверить, встречается ли keyword.lower()
           как подстрока в нормализованном query.
        3. Вернуть список совпавших keyword (в исходном виде, не в lower()),
           сохраняя порядок из keywords. Пустой список, если совпадений нет.
        """
        ...

    def classify(self, query: str) -> QueryTruthClassification:
        """
        Основной метод: классифицировать запрос по оси типа правды и
        сразу порекомендовать класс источника.

        TODO:
        1. intent_matches = self.find_matches(query, self._intent_keywords)
        2. status_matches = self.find_matches(query, self._status_keywords)
        3. Определить axis:
           - Если intent_matches И status_matches оба непустые ->
             TruthAxis.AMBIGUOUS (запрос смешивает обе оси).
           - Иначе если status_matches непустой (а intent_matches
             пустой) -> TruthAxis.OPERATIONAL_STATUS.
           - Иначе если intent_matches непустой (а status_matches
             пустой) -> TruthAxis.DESIGN_INTENT.
           - Иначе (оба списка пустые — ни одного маркера не найдено) ->
             TruthAxis.AMBIGUOUS (недостаточно сигнала для уверенного
             выбора, нельзя молча выбирать наугад).
        4. recommended = self.recommend_source_type(axis)
        5. Вернуть QueryTruthClassification(query=query, axis=axis,
           matched_intent_keywords=intent_matches,
           matched_status_keywords=status_matches,
           recommended_source_type=recommended).
        """
        ...

    def classify_axis(self, query: str) -> TruthAxis:
        """
        Удобный тонкий метод, когда нужна только ось, без полной
        классификации.

        TODO: вернуть self.classify(query).axis. Не дублировать логику
        шага 3 метода classify — только переиспользовать его.
        """
        ...

    def recommend_source_type(self, axis: TruthAxis) -> SourceType | None:
        """
        Порекомендовать класс источника по уже определённой оси.

        TODO:
        1. TruthAxis.DESIGN_INTENT -> SourceType.CONFLUENCE
        2. TruthAxis.OPERATIONAL_STATUS -> SourceType.JIRA
        3. TruthAxis.AMBIGUOUS -> None (не выбирать заранее — см.
           should_cross_check)
        """
        ...

    def should_cross_check(self, classification: QueryTruthClassification) -> bool:
        """
        Нужно ли сверять оба класса источников (или эскалировать вопрос
        человеку) вместо того, чтобы доверять единственному источнику.

        TODO: вернуть True, если classification.axis is
        TruthAxis.AMBIGUOUS, иначе False.
        """
        ...
