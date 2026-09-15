from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.context.multi_hop_retriever import HopRecord, StopReason
from app.context.retrieval_planner import RetrievalBranch, RetrievalPlan, RetrievalPlanner, RetrievalStrategy
from app.context.source_registry import SourceType
from app.context.source_selector import SourceSelector
from app.context.truth_axis_router import TruthAxis


__all__ = [
    "IsSufficientFn",
    "SourceSelectionEvent",
    "MultiHopEvent",
    "EscalationEvent",
    "RetrievalTrace",
    "RetrievalTracer",
]


# Колбэк, которым внешний вызывающий код сообщает трассировщику, оказался ли
# уже построенный RetrievalPlan достаточным — та же ответственность, что
# RetrievalPlanner.escalate() (урок 3.4) оставляет вызывающей стороне: сам
# планировщик не решает, была ли попытка недостаточной, и трассировщик
# наследует это же правило, а не берёт решение на себя.
IsSufficientFn = Callable[[RetrievalPlan], bool]


@dataclass
class SourceSelectionEvent:
    """
    Сигналы SourceSelector.select() (3.3) для ОДНОЙ ветки плана, записанные
    для диагностики. RetrievalBranch (3.4) хранит только selected_sources —
    entity_hint, axis и used_fallback до этого класса нигде не сохранялись.
    """

    branch_query: str
    selected_sources: list[SourceType]
    entity_hint: Optional[SourceType]
    axis: TruthAxis
    used_fallback: bool


@dataclass
class MultiHopEvent:
    """Полный результат MultiHopRetriever.run() (3.2) для одной ветки плана."""

    branch_query: str
    hops: list[HopRecord]
    stop_reason: StopReason
    final_entity: Optional[str]


@dataclass
class EscalationEvent:
    """Один шаг подъёма по лестнице SIMPLE -> DECOMPOSE -> MULTI_HOP (3.4)."""

    from_strategy: RetrievalStrategy
    to_strategy: RetrievalStrategy
    reason: str


@dataclass
class RetrievalTrace:
    """
    Полная диагностическая запись одного вызова RetrievalTracer.run() —
    не замена RetrievalPlan, а надстройка над ним: RetrievalPlan остаётся
    единственным источником правды о том, ЧТО делать дальше по конвейеру
    (retrieval по веткам), а RetrievalTrace отвечает на вопрос, который
    задают только постфактум, при разборе инцидента: ПОЧЕМУ конвейер
    принял именно эти решения.
    """

    original_query: str
    strategy_history: list[RetrievalStrategy] = field(default_factory=list)
    escalations: list[EscalationEvent] = field(default_factory=list)
    source_events: list[SourceSelectionEvent] = field(default_factory=list)
    multi_hop_events: list[MultiHopEvent] = field(default_factory=list)
    final_plan: Optional[RetrievalPlan] = None

    def as_dict(self) -> dict:
        """
        Сериализовать трассу в JSON-совместимый словарь для записи в
        лог-хранилище (тот же принцип структурированного лога, который в
        курсе 8 применяется уже ко всему конвейеру, а не только к retrieval).

        TODO:
        1. Вернуть dict с ключами:
           - "original_query": self.original_query
           - "strategy_history": [s.value for s in self.strategy_history]
           - "escalations": список dict с полями "from_strategy" (.value),
             "to_strategy" (.value), "reason" — по одному на каждый
             EscalationEvent из self.escalations
           - "source_events": список dict с полями "branch_query",
             "selected_sources" (список .value), "entity_hint" (.value или
             None, если entity_hint is None), "axis" (.value),
             "used_fallback" — по одному на каждый SourceSelectionEvent
           - "multi_hop_events": список dict с полями "branch_query",
             "hop_count" (len(event.hops)), "stop_reason" (.value),
             "final_entity" — по одному на каждый MultiHopEvent
           - "final_strategy": self.final_plan.strategy.value, если
             self.final_plan is not None, иначе None
        2. Никаких сырых dataclass-инстансов или Enum-объектов в
           результате — только примитивы, готовые к json.dumps().
        """
        ...

    def flag_risks(self) -> list[str]:
        """
        Просмотреть уже заполненную трассу и вернуть список
        человекочитаемых предупреждений о потенциально слабых местах —
        не доказанная ошибка, а кандидаты на первоочередную проверку
        инженером при разборе похожего инцидента в будущем.

        TODO:
        1. risks: list[str] = []
        2. Если len(self.escalations) > 0: добавить в risks сообщение о
           том, что план был эскалирован len(self.escalations) раз(а),
           и что исходная классификация стратегии могла быть неверной
           (упомянуть self.strategy_history в сообщении).
        3. Для каждого event in self.source_events: если
           event.used_fallback is True — добавить сообщение вида
           "ветка '{event.branch_query}': источник выбран через fallback,
           а не по уверенному сигналу".
        4. Для каждого event in self.multi_hop_events: если
           event.stop_reason is не StopReason.ANSWER_FOUND — добавить
           сообщение вида "ветка '{event.branch_query}': multi-hop
           остановился по {event.stop_reason.value}, а не по найденному
           ответу".
        5. Вернуть risks (пустой список, если ничего не найдено).
        """
        ...


class RetrievalTracer:
    """
    Диагностический слой поверх уже готового RetrievalPlanner (3.4) —
    не дублирует ни одну из его обязанностей, а лишь записывает, какие
    решения были приняты при обработке конкретного запроса, чтобы их
    можно было разобрать постфактум (см. текст урока 3.5, разделы 1 и 3).

    RetrievalTracer сознательно принимает source_selector отдельным
    параметром конструктора — тем же экземпляром, что уже передан внутрь
    planner при его создании. Это осознанное архитектурное решение, а не
    доступ к приватному атрибуту planner: RetrievalBranch (3.4) хранит
    только selected_sources, а entity_hint/axis/used_fallback, вычисленные
    SourceSelector.select() при построении ветки, нигде не сохраняются.
    Чтобы восстановить эти сигналы для трассы, RetrievalTracer повторно
    вызывает select() для запроса каждой ветки — дешёвая, детерминированная,
    rule-based операция (3.3), которая не меняет уже построенный план, а
    только заново вычисляет то же самое решение ради записи его причины.
    """

    def __init__(
        self,
        planner: RetrievalPlanner,
        source_selector: SourceSelector,
        is_sufficient_fn: Optional[IsSufficientFn] = None,
        max_escalations: int = 2,
    ) -> None:
        self._planner = planner
        self._source_selector = source_selector
        self._is_sufficient_fn = is_sufficient_fn
        self._max_escalations = max_escalations

    def run(self, query: str) -> tuple[RetrievalPlan, RetrievalTrace]:
        """
        Выполнить query через RetrievalPlanner, попутно записывая каждое
        решение в RetrievalTrace.

        TODO:
        1. trace = RetrievalTrace(original_query=query)
        2. plan = self._planner.build_plan(query)
        3. trace.strategy_history.append(plan.strategy)
        4. self._record_plan(plan, trace)
        5. attempts = 0
        6. Пока (self._is_sufficient_fn is not None) и
           (self._is_sufficient_fn(plan) is False) и
           (attempts < self._max_escalations):
           a. previous_strategy = plan.strategy
           b. escalated_plan = self._planner.escalate(plan)
           c. Если escalated_plan.strategy == previous_strategy — эскалировать
              уже некуда (RetrievalPlanner.escalate уже на MULTI_HOP,
              см. 3.4) — прервать цикл (break), не добавляя EscalationEvent
           d. plan = escalated_plan
           e. trace.escalations.append(EscalationEvent(
              from_strategy=previous_strategy, to_strategy=plan.strategy,
              reason="is_sufficient_fn вернул False"))
           f. trace.strategy_history.append(plan.strategy)
           g. self._record_plan(plan, trace)
           h. attempts += 1
        7. trace.final_plan = plan
        8. Вернуть (plan, trace)
        """
        ...

    def _record_plan(self, plan: RetrievalPlan, trace: RetrievalTrace) -> None:
        """
        Записать в trace сигналы SourceSelector и результаты
        MultiHopRetriever для КАЖДОЙ ветки уже построенного плана.

        TODO:
        1. Для каждой branch in plan.branches:
           a. selection = self._source_selector.select(branch.query)
           b. trace.source_events.append(SourceSelectionEvent(
              branch_query=branch.query,
              selected_sources=selection.selected_sources,
              entity_hint=selection.entity_hint,
              axis=selection.axis,
              used_fallback=selection.used_fallback,
           ))
           c. Если branch.requires_multi_hop is True:
              result = self._planner.run_branch(branch)
              Если result is not None:
              trace.multi_hop_events.append(MultiHopEvent(
                branch_query=branch.query,
                hops=result.hops,
                stop_reason=result.stop_reason,
                final_entity=result.final_entity,
              ))
        """
        ...
