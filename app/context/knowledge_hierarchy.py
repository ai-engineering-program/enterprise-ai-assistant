from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.context.page_trust_scorer import ConfluencePage, PageTrustScorer
from app.context.source_registry import SourceType
from app.context.structure_scorer import TextStructureScorer
from app.context.thread_filter import Thread, ThreadNoiseFilter
from app.context.truth_axis_router import (
    QueryTruthClassification,
    TruthAxis,
    TruthAxisRouter,
)


__all__ = [
    "AxisWeights",
    "DEFAULT_WEIGHTS",
    "DEFAULT_AXIS_MISMATCH_PENALTY",
    "DEFAULT_NEUTRAL_TRUST",
    "KnowledgeCandidate",
    "PriorityBreakdown",
    "HierarchyResolution",
    "KnowledgeHierarchyResolver",
]


@dataclass(frozen=True)
class AxisWeights:
    """
    Веса трёх независимых сигналов при своде в единый priority score для
    ОДНОГО типа источника. Сумма не обязана давать ровно 1.0 — это не
    вероятностное распределение, а относительная важность сигналов именно
    для этого класса источников.
    """

    structure: float = 0.30
    trust: float = 0.40
    axis_alignment: float = 0.30


# Дефолтные веса по типу источника. Это НЕ единственно верные числа — это
# отправная точка, которую предполагается пересматривать по мере того, как
# в компанию добавляются новые типы источников. Добавление нового типа
# источника НЕ требует правки кода резолвера — только одного вызова
# KnowledgeHierarchyResolver.register_weights(new_type, AxisWeights(...)).
DEFAULT_WEIGHTS: dict[SourceType, AxisWeights] = {
    # Confluence: PageTrustScorer — специализированный и надёжный сигнал
    # жизненного цикла страницы, поэтому доверию отдан наибольший вес.
    SourceType.CONFLUENCE: AxisWeights(structure=0.25, trust=0.45, axis_alignment=0.30),
    # Jira: enforced-поля workflow делают структуру почти гарантированной,
    # а сам факт "это Jira" сильнее всего сигнализирует об операционном
    # статусе — поэтому наибольший вес у согласованности с осью правды.
    SourceType.JIRA: AxisWeights(structure=0.20, trust=0.30, axis_alignment=0.50),
    # Slack: ни PageTrustScorer, ни ось правды не специализированы под
    # переписку напрямую — структура и доверие через ThreadNoiseFilter
    # несут основную нагрузку.
    SourceType.SLACK: AxisWeights(structure=0.30, trust=0.50, axis_alignment=0.20),
    # OTHER: намеренно сбалансированные веса по умолчанию для источников,
    # которые ещё не были явно откалиброваны под конкретное предприятие.
    SourceType.OTHER: AxisWeights(structure=0.34, trust=0.33, axis_alignment=0.33),
}

# Множитель, применяемый к итоговому score при ЖЁСТКОМ несовпадении оси
# правды (кандидат — источник противоположного типа тому, что рекомендует
# TruthAxisRouter). Это НЕ абсолютное вето (как is_deprecated в
# PageTrustScorer, урок 2.3) — источник всё ещё может победить, если у
# него нет конкурентов с совпадающей осью, но конкурировать ему заметно
# сложнее. См. инцидент урока 2.4 (design-intent страница отвечала на
# operational-status вопрос).
DEFAULT_AXIS_MISMATCH_PENALTY: float = 0.5

# Нейтральное значение доверия для источников, для которых в app/context/
# ещё нет специализированного скорера жизненного цикла (например, Jira —
# доверие к enforced-полям тикета не измеряется отдельным инструментом в
# этом модуле). Намеренно НЕ 0.0 и НЕ 1.0 — отсутствие специализированного
# сигнала не должно ни наказывать, ни поощрять кандидата.
DEFAULT_NEUTRAL_TRUST: float = 0.6


@dataclass
class KnowledgeCandidate:
    """
    Один фрагмент контекста — кандидат, уже найденный retrieval-слоем для
    конкретного запроса, но ещё не окончательно ранжированный.

    confluence_page задаётся, только если source_type is SourceType.CONFLUENCE
    и есть метаданные жизненного цикла страницы (урок 2.3). thread задаётся,
    только если source_type is SourceType.SLACK и кандидат — целый тред
    (урок 2.2). Для Jira и любого другого типа оба поля остаются None — это
    не ошибка данных, а сигнал резолверу использовать нейтральное значение
    доверия вместо специализированного скорера.
    """

    fragment_id: str
    source_type: SourceType
    text: str
    confluence_page: ConfluencePage | None = None
    thread: Thread | None = None


@dataclass
class PriorityBreakdown:
    """
    Полная раскладка итоговой оценки одного кандидата — не просто число,
    а объяснимый результат: какой вклад дал каждый из трёх сигналов и
    почему. Тот же принцип объяснимости, что и у TextStructureScorer,
    PageTrustScorer и TruthAxisRouter — если ранжирование выглядит
    неожиданным, инженер должен суметь открыть PriorityBreakdown и увидеть,
    какой именно сигнал его вызвал, а не гадать.
    """

    fragment_id: str
    source_type: SourceType
    structure_score: float
    trust_score: float
    axis_alignment_score: float
    axis_mismatch: bool
    final_score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class HierarchyResolution:
    """
    Результат разрешения иерархии знаний для ОДНОГО запроса: полный
    ранжированный список кандидатов плюс явный флаг, нужна ли сверка
    нескольких классов источников вместо единственного победителя
    (см. TruthAxisRouter.should_cross_check, урок 2.4).
    """

    query: str
    axis: TruthAxis
    cross_check_recommended: bool
    ranked: list[PriorityBreakdown]
    top_by_source_type: dict[SourceType, PriorityBreakdown] | None = None


class KnowledgeHierarchyResolver:
    """
    Оркестратор всего модуля M2: сводит четыре независимых кирпича —
    TextStructureScorer (2.1), ThreadNoiseFilter (2.2), PageTrustScorer
    (2.3) и TruthAxisRouter (2.4) — в единую политику приоритета
    источников, вызываемую retrieval-слоем (модуль M3) для одного запроса
    и списка уже найденных кандидатов.

    Как и ContextConflictAnalyzer (урок 1.4), этот класс НЕ переизобретает
    логику своих четырёх зависимостей — он получает их через конструктор
    (dependency injection) и обращается к ним как к готовым инструментам
    с известным контрактом. Собственная логика резолвера — это ТОЛЬКО:

    1. свод трёх независимых сигналов (структура, доверие, ось правды) в
       единый priority score по конфигурируемым весам;
    2. явная, объяснимая обработка случая, когда сигналы противоречат друг
       другу (высокое доверие и аккуратная структура, но неверная ось
       правды — инцидент урока 2.4);
    3. решение о том, нужна ли сверка нескольких классов источников вместо
       единственного победителя.
    """

    def __init__(
        self,
        structure_scorer: TextStructureScorer,
        thread_filter: ThreadNoiseFilter,
        page_trust_scorer: PageTrustScorer,
        truth_axis_router: TruthAxisRouter,
        weights: dict[SourceType, AxisWeights] | None = None,
        axis_mismatch_penalty: float = DEFAULT_AXIS_MISMATCH_PENALTY,
        neutral_trust_default: float = DEFAULT_NEUTRAL_TRUST,
    ) -> None:
        self._structure_scorer = structure_scorer
        self._thread_filter = thread_filter
        self._page_trust_scorer = page_trust_scorer
        self._truth_axis_router = truth_axis_router
        self._weights: dict[SourceType, AxisWeights] = dict(
            weights if weights is not None else DEFAULT_WEIGHTS
        )
        self._axis_mismatch_penalty = axis_mismatch_penalty
        self._neutral_trust_default = neutral_trust_default

    def register_weights(self, source_type: SourceType, weights: AxisWeights) -> None:
        """
        Зарегистрировать (или переопределить) веса для типа источника —
        точка расширения, ради которой enterprise НЕ должен править код
        резолвера при добавлении нового источника (например, Notion или
        внутреннего wiki-движка): достаточно вызвать этот метод один раз
        при инициализации системы, а не добавлять новую ветку if/elif
        внутри score_candidate.

        TODO: сохранить weights в self._weights[source_type].
        """
        ...

    def weights_for(self, source_type: SourceType) -> AxisWeights:
        """
        Вернуть веса для типа источника, а не бросать исключение для ещё
        не сконфигурированного типа — новый, никогда не регистрировавшийся
        тип источника должен получить разумные веса по умолчанию, а не
        сломать резолвер.

        TODO: вернуть self._weights.get(source_type, AxisWeights()) —
        AxisWeights() без аргументов уже задаёт сбалансированный дефолт
        (0.30 / 0.40 / 0.30) на случай ещё не откалиброванного типа
        источника.
        """
        ...

    def structure_component(self, candidate: KnowledgeCandidate) -> float:
        """
        Оценка формы текста кандидата (урок 2.1).

        TODO: вернуть self._structure_scorer.score(candidate.text).
        """
        ...

    def trust_component(
        self, candidate: KnowledgeCandidate, today: date
    ) -> tuple[float, str]:
        """
        Оценка доверия к кандидату тем специализированным инструментом,
        который для него применим, вместе с понятной заметкой для
        PriorityBreakdown.

        TODO:
        1. Если candidate.confluence_page is not None — вернуть
           (self._page_trust_scorer.score(candidate.confluence_page, today),
           "доверие по PageTrustScorer (2.3)").
        2. Иначе если candidate.thread is not None:
           a. Если not self._thread_filter.is_worth_indexing(candidate.thread)
              — вернуть (0.0, "тред не проходит is_worth_indexing — подозрение
              на шум или устаревание (2.2)") и НЕ вызывать signal_ratio.
           b. Иначе вернуть (self._thread_filter.signal_ratio(candidate.thread),
              "доверие по signal_ratio треда (2.2)").
        3. Иначе (Jira и любой другой тип без специализированного скорера)
           — вернуть (self._neutral_trust_default, "нет специализированного
           скорера доверия для этого типа источника — нейтральное значение
           по умолчанию").
        """
        ...

    def axis_alignment_component(
        self,
        candidate: KnowledgeCandidate,
        classification: QueryTruthClassification,
    ) -> tuple[float, bool]:
        """
        Оценка согласованности типа источника кандидата с осью правды
        запроса (урок 2.4). Возвращает (score, is_hard_mismatch).

        TODO:
        1. Если classification.axis is TruthAxis.AMBIGUOUS — вернуть
           (0.5, False): ось не определена уверенно, штрафовать некого.
        2. Если candidate.source_type == classification.recommended_source_type
           — вернуть (1.0, False): кандидат отвечает именно на тот тип
           правды, который спросили.
        3. Если classification.recommended_source_type in
           {SourceType.CONFLUENCE, SourceType.JIRA} и candidate.source_type
           in {SourceType.CONFLUENCE, SourceType.JIRA} и они не равны
           (то есть кандидат — источник ПРОТИВОПОЛОЖНОГО типа правды) —
           вернуть (0.0, True): жёсткое несовпадение оси, ровно инцидент
           урока 2.4 (design-intent страница отвечала на operational-status
           вопрос).
        4. Иначе (кандидат — Slack/Other, для которых эта ось не
           специализирована ни в одну сторону) — вернуть (0.5, False).
        """
        ...

    def score_candidate(
        self,
        candidate: KnowledgeCandidate,
        classification: QueryTruthClassification,
        today: date,
    ) -> PriorityBreakdown:
        """
        Свести три сигнала в один объяснимый PriorityBreakdown для ОДНОГО
        кандидата.

        TODO:
        1. structure = self.structure_component(candidate)
        2. trust, trust_note = self.trust_component(candidate, today)
        3. axis_score, mismatch = self.axis_alignment_component(candidate, classification)
        4. weights = self.weights_for(candidate.source_type)
        5. raw = (weights.structure * structure + weights.trust * trust
           + weights.axis_alignment * axis_score)
        6. Если mismatch — final = raw * self._axis_mismatch_penalty, иначе
           final = raw. Это НЕ абсолютное вето (в отличие от is_deprecated
           в PageTrustScorer) — кандидат всё ещё может победить при
           отсутствии альтернатив с совпадающей осью, но конкурировать ему
           заметно сложнее.
        7. notes = [trust_note]; если mismatch — добавить в notes
           "ось правды запроса не совпадает с типом источника (2.4)".
        8. Вернуть PriorityBreakdown(fragment_id=candidate.fragment_id,
           source_type=candidate.source_type, structure_score=structure,
           trust_score=trust, axis_alignment_score=axis_score,
           axis_mismatch=mismatch, final_score=final, notes=notes).
        """
        ...

    def rank(
        self,
        candidates: list[KnowledgeCandidate],
        query: str,
        today: date,
    ) -> list[PriorityBreakdown]:
        """
        Ранжировать всех кандидатов для одного запроса.

        TODO:
        1. classification = self._truth_axis_router.classify(query)
        2. breakdowns = [self.score_candidate(c, classification, today)
           for c in candidates]
        3. Отсортировать по убыванию final_score; при равенстве — по
           fragment_id по возрастанию (детерминированность, тот же принцип,
           что в SourceRegistry.rank_by_authority и PageTrustScorer.rank_by_trust).
        4. Вернуть отсортированный список.
        """
        ...

    def resolve(
        self,
        candidates: list[KnowledgeCandidate],
        query: str,
        today: date,
    ) -> HierarchyResolution:
        """
        Полное разрешение иерархии знаний для одного запроса — точка
        входа, которую будет вызывать планировщик retrieval модуля M3.

        TODO:
        1. classification = self._truth_axis_router.classify(query)
        2. ranked = self.rank(candidates, query, today) (внутри rank()
           classify(query) будет вызван ещё раз — для целей этого
           учебного упражнения это допустимо, оба вызова детерминированы
           и дёшевы; в проде стоило бы переиспользовать одну и ту же
           classification вместо повторного вызова).
        3. cross_check = self._truth_axis_router.should_cross_check(classification)
        4. Если cross_check: собрать top_by_source_type — словарь
           {source_type: лучший PriorityBreakdown этого типа}, проходя по
           ranked ПО ПОРЯДКУ (он уже отсортирован по убыванию final_score)
           и беря для каждого source_type первое встреченное вхождение.
           Это и есть "сверить оба класса источников" из урока 2.4 вместо
           единственного победителя. Если не cross_check —
           top_by_source_type = None.
        5. Вернуть HierarchyResolution(query=query, axis=classification.axis,
           cross_check_recommended=cross_check, ranked=ranked,
           top_by_source_type=top_by_source_type).
        """
        ...
