from __future__ import annotations

from dataclasses import dataclass, field

from app.context.token_budget_allocator import (
    BudgetReport,
    BudgetCategory,
    ContextComponent,
    TokenBudgetAllocator,
)
from app.context.context_prioritizer import (
    ContextPrioritizer,
    PrioritizationResult,
    PriorityItem,
)


__all__ = [
    "CategoryContent",
    "AssemblyTrace",
    "AssembledPrompt",
    "ContextAssembler",
]


@dataclass(frozen=True)
class CategoryContent:
    """
    Всё содержимое ОДНОЙ категории бюджета (BudgetCategory) — список уже
    оценённых кандидатов (PriorityItem с посчитанным score), полученных
    от KnowledgeHierarchyResolver (M2/M3) или соответствующего
    компонента памяти (M5). ContextAssembler не пересчитывает score
    внутри items — это чужая обязанность, как и во всех уроках модуля
    M6 (см. docstring PriorityItem, урок 6.2).
    """

    category: BudgetCategory
    items: list[PriorityItem]


@dataclass
class AssemblyTrace:
    """
    Полная объяснимость одного вызова ContextAssembler.assemble():
    что решил TokenBudgetAllocator (6.1), что решил ContextPrioritizer
    на уровне каждой отдельной категории (6.2, шаг "внутри одной
    категории") и что решил финальный межкатегорийный проход (6.2, шаг
    "по всем категориям сразу"). Это тот же принцип объяснимости, что
    у PriorityBreakdown.notes (2.5) и RetrievalTracer (3.5) — при
    неожиданном ответе инженер разбирает trace, а не гадает по логам.
    """

    budget_report: BudgetReport
    category_results: dict[BudgetCategory, PrioritizationResult]
    final_result: PrioritizationResult
    all_dropped: list[PriorityItem] = field(default_factory=list)


@dataclass(frozen=True)
class AssembledPrompt:
    """Итог одного вызова assemble() — то, что реально уйдёт модели."""

    text: str
    trace: AssemblyTrace


class ContextAssembler:
    """
    Модуль M6, урок 6.3: финальная точка входа сборки промпта в
    app/context/ — единственное место, которое остальная система
    (app/api, а в будущих курсах app/agents) вызывает, чтобы получить
    готовый текст промпта. Компонует TokenBudgetAllocator (6.1) и
    ContextPrioritizer (6.2) в правильном порядке — тот самый шаг,
    которого не хватало в инциденте урока 6.1 ("код собирал промпт в
    фиксированном порядке без агрегированной проверки бюджета").

    Не реализует ни одну задачу заново: не считает токены сам (это
    TokenBudgetAllocator), не отбирает и не расставляет элементы сам
    (это ContextPrioritizer), не решает, что относится к запросу (это
    M1-M5). Композиция, а не пятый независимый инструмент — тот же
    принцип, что у KnowledgeHierarchyResolver (2.5).
    """

    def __init__(
        self,
        allocator: TokenBudgetAllocator,
        prioritizer: ContextPrioritizer,
        separator: str = "\n\n",
    ) -> None:
        """
        TODO: сохранить allocator, prioritizer и separator в
        self._allocator, self._prioritizer, self._separator.
        """
        ...

    def assemble(self, category_contents: list[CategoryContent]) -> AssembledPrompt:
        """
        Собрать финальный текст промпта из содержимого всех категорий.

        TODO:
        1. Построить components: list[ContextComponent] — по одному
           ContextComponent(category=cc.category, text=item.text) на
           КАЖДЫЙ item внутри КАЖДОГО CategoryContent, сохраняя
           исходный порядок category_contents и items внутри каждой
           категории. TokenBudgetAllocator.build_report сам суммирует
           несколько компонентов одной категории.
        2. report = self._allocator.build_report(components).
        3. Пройти по category_contents В ИСХОДНОМ ПОРЯДКЕ. Для каждой
           cc:
           - если cc.category встречается в
             report.categories_needing_truncation: найти её
             CategoryAllocation в report.allocations (совпадение по
             cc.category), вызвать self._prioritizer.prioritize(
             cc.items, allocation.allocated_tokens); сохранить
             результат в category_results[cc.category]; добавить
             result.kept к списку survivors (в порядке result.kept).
           - иначе (категория уложилась в свой бюджет целиком): ВСЕ
             cc.items добавляются в survivors напрямую, БЕЗ вызова
             prioritizer — нет смысла отбирать то, что не превышает
             свой лимит (см. текст урока 6.2, "категории, которые уже
             уложились в бюджет, этот шаг не трогает вообще").
        4. final_result = self._prioritizer.prioritize(survivors,
           report.content_budget_tokens) — межкатегорийный трейд-офф
           и итоговая начало/конец-взвешенная расстановка ПО ВСЕМ
           выжившим элементам сразу, независимо от исходной категории
           (см. текст урока 6.2, шаг "по всем категориям сразу, в
           конце").
        5. text = self._separator.join(item.text for item in
           final_result.kept).
        6. all_dropped = конкатенация: сначала dropped из ВСЕХ
           category_results (в порядке появления категории в
           category_contents), затем final_result.dropped.
        7. Вернуть AssembledPrompt(
             text=text,
             trace=AssemblyTrace(
               budget_report=report,
               category_results=category_results,
               final_result=final_result,
               all_dropped=all_dropped,
             ),
           )

        При пустом category_contents вернуть AssembledPrompt с
        text="" и AssemblyTrace с пустыми category_results, пустым
        all_dropped и final_result от вызова prioritizer с пустым
        списком items.
        """
        ...
