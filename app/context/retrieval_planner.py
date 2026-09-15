from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from app.context.multi_hop_retriever import MultiHopResult, MultiHopRetriever
from app.context.query_decomposer import QueryDecomposer
from app.context.source_registry import SourceType
from app.context.source_selector import SourceSelector


__all__ = [
    "RetrievalStrategy",
    "RetrievalBranch",
    "RetrievalPlan",
    "RetrievalPlanner",
]


class RetrievalStrategy(Enum):
    """
    Минимально достаточная стратегия retrieval для одного (под)запроса.

    Порядок значений совпадает с лестницей эскалации SIMPLE -> DECOMPOSE ->
    MULTI_HOP (см. _STRATEGY_LADDER ниже) — каждая следующая стратегия
    строго тяжелее предыдущей по числу вызовов retrieval, которые она
    порождает.
    """

    SIMPLE = "simple"
    DECOMPOSE = "decompose"
    MULTI_HOP = "multi_hop"


# Фиксированный порядок эскалации. Эскалация всегда двигается ровно на одну
# ступень вперёд по этому кортежу — никогда не перепрыгивает через ступень
# и никогда не откатывается назад.
_STRATEGY_LADDER: tuple[RetrievalStrategy, ...] = (
    RetrievalStrategy.SIMPLE,
    RetrievalStrategy.DECOMPOSE,
    RetrievalStrategy.MULTI_HOP,
)

# Маркеры ссылки на сущность, которая раньше занимала какую-то роль, но
# сейчас неизвестна — тот же класс сигнала, что и в вопросе Марины про
# Ковтуна (урок 3.2): "кто сейчас отвечает за то, чем раньше руководил
# Ковтун". classify_pattern() из QueryDecomposer (3.1) видит только число
# независимых информационных потребностей (здесь их одна — запрос ATOMIC),
# а не то, что второй логический шаг зависит от факта, которого ещё нет
# в тексте вопроса. Эти маркеры закрывают именно этот пробел.
_DEFAULT_MULTI_HOP_MARKERS: tuple[str, ...] = (
    "которым раньше",
    "который раньше",
    "которая раньше",
    "которое раньше",
    "бывший",
    "бывшего",
    "прежний",
    "прежнего",
    "предыдущий",
    "предыдущего",
)


@dataclass
class RetrievalBranch:
    """
    Одна независимо обрабатываемая ветка плана — либо единственный запрос
    (SIMPLE, MULTI_HOP), либо один из подзапросов декомпозиции (DECOMPOSE).

    requires_multi_hop=True означает, что эта ветка должна быть передана в
    MultiHopRetriever.run(), а не обработана одним вызовом retrieval —
    RetrievalPlanner сам не выполняет цикл hop'ов, он лишь помечает, какой
    ветке он нужен (см. RetrievalPlanner.run_branch()).
    """

    query: str
    selected_sources: list[SourceType]
    requires_multi_hop: bool = False


@dataclass
class RetrievalPlan:
    """
    Полный результат планирования для одного пользовательского запроса.

    escalated_from не None только если план — результат вызова escalate(),
    а не build_plan() с нуля: хранит стратегию, которая была признана
    недостаточной на предыдущей попытке.
    """

    original_query: str
    strategy: RetrievalStrategy
    branches: list[RetrievalBranch] = field(default_factory=list)
    escalated_from: Optional[RetrievalStrategy] = None


class RetrievalPlanner:
    """
    Оркестратор трёх инструментов модуля M3 "Планирование retrieval":
    QueryDecomposer (3.1), MultiHopRetriever (3.2) и SourceSelector (3.3).

    RetrievalPlanner не переизобретает логику ни одного из них — все три
    получены через конструктор (dependency injection, тот же принцип, что
    и у SourceSelector с TruthAxisRouter). Единственная новая ответственность
    этого класса — решить, КАКУЮ минимально достаточную стратегию применить
    к входящему запросу, ДО того как сделан хоть один вызов retrieval:

    - SIMPLE: одна информационная потребность, нет зависимости между
      шагами -> только SourceSelector.select() для исходного запроса.
    - DECOMPOSE: QueryDecomposer.classify_pattern() вернул не ATOMIC ->
      декомпозировать и вызвать SourceSelector.select() для каждого
      подзапроса отдельно.
    - MULTI_HOP: classify_pattern() вернул ATOMIC, но в тексте есть маркер
      ссылки на сущность, которая раньше занимала роль, но сейчас
      неизвестна -> SourceSelector.select() для запроса первого hop'а,
      ветка помечается requires_multi_hop=True для последующего вызова
      MultiHopRetriever.run().

    Если результат build_plan() оказался недостаточным (see escalate()),
    планировщик поднимается ровно на одну ступень по фиксированной
    лестнице SIMPLE -> DECOMPOSE -> MULTI_HOP, а не переклассифицирует
    запрос заново с нуля и не перепрыгивает через ступень.
    """

    def __init__(
        self,
        decomposer: QueryDecomposer,
        multi_hop_retriever: MultiHopRetriever,
        source_selector: SourceSelector,
        multi_hop_markers: Optional[Iterable[str]] = None,
    ) -> None:
        self._decomposer = decomposer
        self._multi_hop_retriever = multi_hop_retriever
        self._source_selector = source_selector
        self._multi_hop_markers: tuple[str, ...] = tuple(
            multi_hop_markers if multi_hop_markers is not None else _DEFAULT_MULTI_HOP_MARKERS
        )

    def classify_strategy(self, query: str) -> RetrievalStrategy:
        """
        Определить минимально достаточную стратегию для query.

        TODO:
        1. pattern = self._decomposer.classify_pattern(query)
        2. Если pattern is не DecompositionPattern.ATOMIC — вернуть
           RetrievalStrategy.DECOMPOSE (запрос уже составной — вопрос
           "нужен ли MULTI_HOP" в этом случае не рассматривается на этом
           уровне: декомпозиция сначала разрежет запрос на независимые
           подзапросы, и цепочечная зависимость внутри одного из них — это
           уже отдельная, более глубокая задача, вне контракта этого
           упражнения).
        3. Иначе (pattern is ATOMIC): lower = query.lower(); если хотя бы
           один marker из self._multi_hop_markers встречается в lower —
           вернуть RetrievalStrategy.MULTI_HOP.
        4. Иначе — вернуть RetrievalStrategy.SIMPLE.
        """
        ...

    def _build_branches(
        self, query: str, strategy: RetrievalStrategy
    ) -> list[RetrievalBranch]:
        """
        Построить ветки плана для уже выбранной стратегии. Общий приватный
        метод, используемый и build_plan(), и escalate() — чтобы правило
        "как из стратегии получить ветки" было определено ровно в одном
        месте.

        TODO:
        1. Если strategy is RetrievalStrategy.SIMPLE:
           result = self._source_selector.select(query)
           вернуть [RetrievalBranch(query=query,
                                     selected_sources=result.selected_sources,
                                     requires_multi_hop=False)]
        2. Если strategy is RetrievalStrategy.DECOMPOSE:
           decomposition = self._decomposer.decompose(query)
           для каждого sub_query в decomposition.sub_queries (в порядке
           sub_query.order): result = self._source_selector.select(sub_query.text)
           вернуть список RetrievalBranch(query=sub_query.text,
                                           selected_sources=result.selected_sources,
                                           requires_multi_hop=False)
        3. Если strategy is RetrievalStrategy.MULTI_HOP:
           result = self._source_selector.select(query) — источник для
           запроса ПЕРВОГО hop'а (переформулированные запросы следующих
           hop'ов подбирает сам MultiHopRetriever своим reformulate_fn,
           а не SourceSelector)
           вернуть [RetrievalBranch(query=query,
                                     selected_sources=result.selected_sources,
                                     requires_multi_hop=True)]
        """
        ...

    def build_plan(self, query: str) -> RetrievalPlan:
        """
        Точка входа: классифицировать запрос и построить полный
        RetrievalPlan с нуля (без предположения о предыдущей попытке).

        TODO:
        1. strategy = self.classify_strategy(query)
        2. branches = self._build_branches(query, strategy)
        3. Вернуть RetrievalPlan(original_query=query, strategy=strategy,
           branches=branches, escalated_from=None).
        """
        ...

    def escalate(self, plan: RetrievalPlan) -> RetrievalPlan:
        """
        Поднять план на одну ступень тяжелее по лестнице
        SIMPLE -> DECOMPOSE -> MULTI_HOP, потому что попытка plan.strategy
        была признана недостаточной вызывающей стороной (например,
        SourceSelector-выбранный источник не вернул ничего релевантного).

        RetrievalPlanner сам не решает, была ли попытка недостаточной —
        это решение принимает вызывающий код по результату фактического
        retrieval; escalate() только строит план для следующей ступени.

        TODO:
        1. current_index = _STRATEGY_LADDER.index(plan.strategy)
        2. Если current_index + 1 >= len(_STRATEGY_LADDER) — эскалировать
           уже некуда (MULTI_HOP — самая тяжёлая ступень); вернуть plan
           БЕЗ ИЗМЕНЕНИЙ (тот же объект или его копию с тем же strategy).
        3. Иначе: next_strategy = _STRATEGY_LADDER[current_index + 1]
           branches = self._build_branches(plan.original_query, next_strategy)
           вернуть RetrievalPlan(original_query=plan.original_query,
                                  strategy=next_strategy,
                                  branches=branches,
                                  escalated_from=plan.strategy).
        """
        ...

    def run_branch(self, branch: RetrievalBranch) -> Optional[MultiHopResult]:
        """
        Выполнить одну ветку плана, если она требует MultiHopRetriever.

        Ветки SIMPLE/DECOMPOSE не выполняются этим методом — у
        RetrievalPlanner нет собственного одноразового retrieve_fn, вызов
        источников для них выполняет внешний слой конвейера, используя
        branch.selected_sources. Единственная ветка, которую планировщик
        умеет довести до конца сам, — та, что помечена
        requires_multi_hop=True, потому что для неё уже есть готовый,
        полностью сконфигурированный self._multi_hop_retriever.

        TODO:
        1. Если branch.requires_multi_hop is False — вернуть None.
        2. Иначе — вернуть self._multi_hop_retriever.run(branch.query).
        """
        ...
