from __future__ import annotations

from dataclasses import dataclass

from app.context.message_quality import ChatMessage, MessageQualityScorer
from app.context.source_registry import SourceRegistry, SourceType
from app.context.structure_classifier import (
    SourceStructureClassifier,
    StructureClass,
)
from enum import Enum


__all__ = [
    "ConflictType",
    "ContextFragment",
    "ConflictFinding",
    "ContextConflictAnalyzer",
]


class ConflictType(Enum):
    """
    Таксономия конфликтов enterprise-контекста из урока 1.4. Обобщает
    инциденты "Печора Телеком" из уроков 1.1 и 1.2 в четыре архитектурные
    категории, а не в список частных случаев.
    """

    # Оба фрагмента одного структурного класса (оба документа, оба
    # чат-лога, оба тикета) — конфликт решается свежестью, а не формой.
    STALE_VS_FRESH = "stale_vs_fresh"

    # Формальный источник (DOCUMENT/SEMI_STRUCTURED) против неформальной
    # переписки (CONVERSATION). Инцидент 1.1: Confluence vs Slack.
    AUTHORITATIVE_VS_INFORMAL = "authoritative_vs_informal"

    # Структурированное поле (Jira due date/статус) против
    # неструктурированного текста (абзац Confluence). Инцидент 1.1:
    # Confluence vs Jira.
    STRUCTURED_VS_UNSTRUCTURED = "structured_vs_unstructured"

    # Конфликт внутри одного и того же источника — источник сам себе
    # противоречит во времени. Инцидент 1.2: Slack-тред про шифрование.
    CONTRADICTORY_WITHIN_SOURCE = "contradictory_within_source"


@dataclass
class ContextFragment:
    """
    Один извлечённый фрагмент контекста — кандидат для промпта генерации.

    source_name должен быть зарегистрирован в SourceRegistry (урок 1.1).
    chat_message заполняется, только если фрагмент пришёл из источника со
    структурным классом CONVERSATION (Slack и аналоги); для документов и
    структурированных полей Jira остаётся None.
    """

    source_name: str
    source_type: SourceType
    text: str
    chat_message: ChatMessage | None = None


@dataclass
class ConflictFinding:
    """Результат анализа одной пары конфликтующих фрагментов."""

    fragment_a: ContextFragment
    fragment_b: ContextFragment
    conflict_type: ConflictType
    recommended_source_name: str | None
    reason: str


class ContextConflictAnalyzer:
    """
    Оркестрация трёх кирпичей модуля M1 в единый инструмент диагностики
    конфликтов enterprise-контекста:

    - SourceRegistry (урок 1.1) — авторитетность источников и чистое
      разрешение конфликта по весу.
    - MessageQualityScorer (урок 1.2) — доверие к отдельному сообщению
      чата как самостоятельному фрагменту.
    - SourceStructureClassifier (урок 1.3) — структурный класс источника.

    Этот класс НЕ вводит новую логику разрешения конфликтов — он
    классифицирует конфликт по таксономии этого урока и делегирует
    решение уже готовым механизмам. Полноценный учёт типа факта и
    свежести появится в модуле M2 "Инжиниринг качества источников".
    """

    def __init__(
        self,
        registry: SourceRegistry,
        quality_scorer: MessageQualityScorer,
        structure_classifier: SourceStructureClassifier,
    ) -> None:
        self.registry = registry
        self.quality_scorer = quality_scorer
        self.structure_classifier = structure_classifier

    def classify_conflict_type(
        self, fragment_a: ContextFragment, fragment_b: ContextFragment
    ) -> ConflictType:
        """
        Определить тип конфликта между двумя фрагментами по таксономии
        этого урока. Порядок проверок важен — применяйте их именно в
        этом порядке (по убыванию специфичности).

        TODO:
        1. Если fragment_a.source_name == fragment_b.source_name —
           вернуть ConflictType.CONTRADICTORY_WITHIN_SOURCE немедленно
           (источник противоречит сам себе, структура тут не важна).
        2. Получить структурные классы через
           self.structure_classifier.classify_structure(fragment.source_type)
           для обоих фрагментов.
        3. Если структурные классы совпадают — вернуть
           ConflictType.STALE_VS_FRESH.
        4. Если ровно один из двух структурных классов —
           StructureClass.CONVERSATION — вернуть
           ConflictType.AUTHORITATIVE_VS_INFORMAL.
        5. Иначе (структурные классы различаются, и ни один не
           CONVERSATION — например, DOCUMENT против SEMI_STRUCTURED) —
           вернуть ConflictType.STRUCTURED_VS_UNSTRUCTURED.
        """
        ...

    def analyze_pair(
        self, fragment_a: ContextFragment, fragment_b: ContextFragment
    ) -> ConflictFinding:
        """
        Проанализировать пару конфликтующих фрагментов и предложить
        решение, переиспользуя уже готовые механизмы модуля M1.

        TODO:
        1. conflict_type = self.classify_conflict_type(fragment_a, fragment_b)

        2. Если conflict_type == ConflictType.CONTRADICTORY_WITHIN_SOURCE:
           - Если у ОБОИХ фрагментов задан chat_message — сравнить
             self.quality_scorer.score(...) для каждого. Рекомендовать
             source_name фрагмента с более высоким score (reason:
             "выше качество как самостоятельного сообщения"). При
             равенстве score — recommended_source_name = None, reason
             про эскалацию человеку.
           - Если хотя бы у одного chat_message не задан —
             recommended_source_name = None, reason: "конфликт внутри
             одного источника без данных о сообщениях неразрешим на
             этом уровне".
           - Вернуть ConflictFinding и завершить (дальше не идти).

        3. Если conflict_type == ConflictType.AUTHORITATIVE_VS_INFORMAL:
           - Определить, у какого из двух фрагментов structure_class
             (через self.structure_classifier.classify_structure) равен
             StructureClass.CONVERSATION и задан chat_message.
           - Если такой фрагмент есть и
             self.quality_scorer.score(chat_message) < 0.5 —
             recommended_source_name = source_name ДРУГОГО (формального)
             фрагмента, reason: "неформальный фрагмент не прошёл порог
             качества сообщения". Вернуть ConflictFinding и завершить.
           - Иначе (сообщение прошло порог, либо chat_message не задан)
             — перейти к шагу 4.

        4. Для ConflictType.STALE_VS_FRESH, ConflictType.
           STRUCTURED_VS_UNSTRUCTURED и как продолжение шага 3 —
           вызвать self.registry.resolve_conflict(fragment_a.source_name,
           fragment_b.source_name):
           - Успех -> recommended_source_name = результат, reason:
             "разрешено по authority_weight в SourceRegistry".
           - ValueError (веса равны) -> recommended_source_name = None,
             reason: "равная авторитетность — требуется эскалация
             человеку (учёт свежести появится в модуле M2)".
           - KeyError (источник не зарегистрирован) ->
             recommended_source_name = None, reason: "источник не
             зарегистрирован в SourceRegistry".

        5. Вернуть ConflictFinding(fragment_a, fragment_b, conflict_type,
           recommended_source_name, reason).
        """
        ...

    def analyze(self, fragments: list[ContextFragment]) -> list[ConflictFinding]:
        """
        Проанализировать список фрагментов, найденных retrieval-слоем для
        одного запроса, и вернуть находки по каждой потенциально
        конфликтующей паре.

        TODO:
        1. Пройтись по всем уникальным парам фрагментов (i < j по индексу
           в списке).
        2. Пропустить пару, если fragment_a.text == fragment_b.text —
           буквально одинаковый текст не является конфликтом (это уже
           должно быть устранено дедупликацией в app/ingestion/, курс 3).
        3. Для остальных пар вызвать self.analyze_pair(fragment_a,
           fragment_b) и добавить результат в список находок.
        4. Вернуть список находок, сохраняя порядок обхода пар.
        """
        ...
