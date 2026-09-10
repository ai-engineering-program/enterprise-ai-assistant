from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.context.source_registry import SourceType


__all__ = [
    "StructureClass",
    "RetrievalTreatment",
    "StructuralProfile",
    "SourceStructureClassifier",
]


class StructureClass(Enum):
    """
    Структурный класс контента источника — это НЕ тип системы, а природа
    того, как в источнике устроен текст: как знание, опубликованное для
    будущего читателя (документ), как переписка, побочный продукт
    совместной работы (переписка), или как гибрид (смешанный).
    """

    DOCUMENT = "document"
    SEMI_STRUCTURED = "semi_structured"
    CONVERSATION = "conversation"


class RetrievalTreatment(Enum):
    """Рекомендованная стратегия retrieval для структурного класса."""

    SECTION_CHUNK = "section_chunk"
    FIELD_LEVEL_EXTRACT = "field_level_extract"
    THREAD_AWARE_CHUNK = "thread_aware_chunk"


@dataclass
class StructuralProfile:
    """
    Результат классификации источника: структурный класс, рекомендованная
    стратегия retrieval и ожидаемый период полураспада истины в днях.

    Это НЕ замена SourceRegistry (урок 1.1, авторитетность источника и
    разрешение конфликтов между источниками) и НЕ замена
    MessageQualityScorer (урок 1.2, оценка отдельного сообщения чата).
    Это более ранний и более общий вопрос: какую стратегию обработки
    в принципе стоит применять к источнику этого типа, до того как
    отдельные чанки или сообщения вообще начнут оцениваться.
    """

    source_type: SourceType
    structure_class: StructureClass
    treatment: RetrievalTreatment
    expected_half_life_days: int


class SourceStructureClassifier:
    """
    Первая, черновая версия классификационного фреймворка
    "тип источника -> структурный класс -> стратегия retrieval"
    из урока 1.3. Полноценная иерархия доверия с учётом типа факта
    появится в модуле M2 "Инжиниринг качества источников".
    """

    def classify_structure(self, source_type: SourceType) -> StructureClass:
        """
        Определить структурный класс источника по его типу.

        TODO:
        1. SourceType.CONFLUENCE -> StructureClass.DOCUMENT
        2. SourceType.JIRA -> StructureClass.SEMI_STRUCTURED
        3. SourceType.SLACK -> StructureClass.CONVERSATION
        4. SourceType.OTHER -> поднять ValueError с понятным сообщением:
           неизвестный тип источника нельзя классифицировать вслепую.
        """
        ...

    def recommend_treatment(self, structure_class: StructureClass) -> RetrievalTreatment:
        """
        Рекомендовать стратегию retrieval по структурному классу.

        TODO:
        1. StructureClass.DOCUMENT -> RetrievalTreatment.SECTION_CHUNK
        2. StructureClass.SEMI_STRUCTURED -> RetrievalTreatment.FIELD_LEVEL_EXTRACT
        3. StructureClass.CONVERSATION -> RetrievalTreatment.THREAD_AWARE_CHUNK
        """
        ...

    def expected_half_life_days(self, structure_class: StructureClass) -> int:
        """
        Вернуть ориентировочный период полураспада истины источника
        такого структурного класса, в днях.

        TODO:
        1. StructureClass.DOCUMENT -> 90
        2. StructureClass.SEMI_STRUCTURED -> 14
        3. StructureClass.CONVERSATION -> 1
        """
        ...

    def profile(self, source_type: SourceType) -> StructuralProfile:
        """
        Построить полный структурный профиль источника.

        TODO:
        1. structure_class = self.classify_structure(source_type)
        2. treatment = self.recommend_treatment(structure_class)
        3. half_life = self.expected_half_life_days(structure_class)
        4. Вернуть StructuralProfile(source_type, structure_class,
           treatment, half_life). Не дублировать логику сопоставления
           из шагов 1-3 — только вызвать соответствующие методы.
        """
        ...
