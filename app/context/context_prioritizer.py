from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.context.context_summarizer import estimate_tokens


__all__ = [
    "PriorityItem",
    "PrioritizationResult",
    "ContextPrioritizer",
]


@dataclass(frozen=True)
class PriorityItem:
    """
    Один кандидат на попадание в финальный промпт — с уже посчитанным
    приоритетом (например, KnowledgeHierarchyResolver.final_score, урок
    2.5, оценкой ветки RetrievalPlanner, урок 3.3, или релевантностью от
    компонента памяти, модуль M5). ContextPrioritizer НЕ пересчитывает
    score — он только решает, кто из уже оценённых кандидатов остаётся в
    бюджете и где именно окажется в итоговом порядке (см. текст урока,
    раздел "приоритизация внутри категории").

    protected=True означает, что элемент никогда не должен быть отброшен
    независимо от бюджета — типично system prompt и сообщение
    пользователя (см. текст урока, раздел "защищённые элементы"). Это
    НЕ означает, что элемент игнорирует бюджет — его токены всё равно
    вычитаются из budget_tokens, просто он не может оказаться в dropped.
    """

    item_id: str
    text: str
    score: float               # выше = важнее; шкала произвольная, но монотонная
    protected: bool = False


@dataclass(frozen=True)
class PrioritizationResult:
    """
    Итог одного вызова ContextPrioritizer.prioritize().

    kept — выжившие элементы, УЖЕ в финальном порядке размещения (после
    _arrange), а не в порядке отбора по score. dropped — отброшенные
    элементы, в порядке убывания score (первый dropped — самый
    приоритетный из отброшенных, см. текст урока и exercise).
    """

    kept: list[PriorityItem]
    dropped: list[PriorityItem]
    kept_tokens: int
    budget_tokens: int


class ContextPrioritizer:
    """
    Модуль M6, урок 6.2: работает СРАЗУ ПОСЛЕ TokenBudgetAllocator (6.1) —
    решает, какие конкретно элементы (не категории целиком) остаются в
    рамках бюджета, и в каком порядке их разместить в финальном промпте,
    чтобы минимизировать эффект "lost in the middle" (урок 4.5) на
    уровне ИТОГОВОГО промпта, а не одной категории.

    Вызывается дважды в конвейере (см. текст урока, диаграмма
    prioritization_pipeline):
    1. Внутри ОДНОЙ категории, помеченной TokenBudgetAllocator как
       categories_needing_truncation — budget_tokens тогда равен
       allocated_tokens этой категории из CategoryAllocation (6.1).
    2. Один раз на ВСЕХ элементах, выживших после шага 1, сразу по всем
       категориям — budget_tokens тогда равен content_budget_tokens
       целиком, для финального межкатегорийного трейд-оффа и итоговой
       расстановки перед сборкой промпта.

    Не пересчитывает score и не обращается к LLM — работает только с
    уже переданными числами, поэтому детерминирован и проверяем без
    живых вызовов модели.
    """

    def __init__(self, token_counter: Callable[[str], int] = estimate_tokens) -> None:
        """
        TODO: сохранить token_counter в self._token_counter.
        """
        ...

    def prioritize(self, items: list[PriorityItem], budget_tokens: int) -> PrioritizationResult:
        """
        Отобрать и расставить элементы в рамках budget_tokens.

        TODO:
        1. Разделить items на protected и droppable
           (protected — item.protected is True).
        2. Отсортировать droppable по score по убыванию. Используйте
           sorted(droppable, key=lambda item: item.score, reverse=True) —
           встроенная сортировка Python устойчива, поэтому элементы с
           одинаковым score сохраняют исходный относительный порядок.
        3. Посчитать self._token_counter(item.text) для КАЖДОГО item
           (и protected, и droppable) один раз — не пересчитывать
           повторно на следующих шагах.
        4. protected элементы ВСЕГДА остаются в kept, независимо от
           бюджета (даже если это приводит к kept_tokens > budget_tokens
           — см. docstring PriorityItem.protected). Их токены вычитаются
           из budget_tokens первыми: remaining_budget = budget_tokens -
           sum(токены protected).
        5. Пройти droppable в порядке убывания score (шаг 2), добавляя
           каждый item в kept, пока накопленные токены droppable-элементов
           не превышают remaining_budget. Первый элемент, который бы
           превысил remaining_budget, и все последующие (уже отсортированные
           как менее приоритетные) — в dropped, в том же порядке убывания
           score, в котором их обходили.
        6. Собрать итоговый список kept_before_arrangement = protected +
           отобранные droppable (порядок здесь неважен — финальный порядок
           задаст _arrange на шаге 7).
        7. Вызвать self._arrange(kept_before_arrangement), чтобы получить
           итоговый порядок размещения (см. docstring _arrange).
        8. kept_tokens = сумма токенов ВСЕХ элементов в итоговом kept
           (protected + отобранные droppable).
        9. Вернуть PrioritizationResult(
             kept=<результат _arrange>,
             dropped=<список droppable, не попавших в kept, в порядке
               убывания score>,
             kept_tokens=kept_tokens,
             budget_tokens=budget_tokens,
           )

        При пустом items вернуть PrioritizationResult с пустыми kept и
        dropped, kept_tokens=0, budget_tokens=budget_tokens.
        """
        ...

    def _arrange(self, items: list[PriorityItem]) -> list[PriorityItem]:
        """
        Расставляет выжившие элементы по принципу "лучшие — на края,
        худшие — в середину" (см. текст урока, раздел "начало/конец-
        взвешенная расстановка"), чтобы противодействовать эффекту
        "lost in the middle" (урок 4.5) на уровне ИТОГОВОГО промпта, а
        не только внутри одной категории — это уже делает
        reorder_context_for_attention (app/rag/context_builder.py,
        курс 2), но только для одной категории retrieved_context.

        TODO:
        1. Отсортировать items по score по убыванию (тот же принцип
           устойчивой сортировки, что и в prioritize()).
        2. Разложить по позициям чередованием начала и конца результата:
           - rank 0 (лучший) -> позиция 0 (начало)
           - rank 1 -> последняя позиция
           - rank 2 -> позиция 1 (вторая с начала)
           - rank 3 -> предпоследняя позиция
           - ...и так далее, до заполнения всех позиций.

           Пример для 5 элементов [A, B, C, D, E] по убыванию score:
           результат — [A, C, E, D, B].
        3. При пустом входном списке вернуть пустой список.
        """
        ...
