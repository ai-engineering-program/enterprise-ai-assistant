from __future__ import annotations

from dataclasses import dataclass


__all__ = ["PageProfile", "RoutingDecision", "LayoutComplexityAdvisor"]


@dataclass
class PageProfile:
    """Сводка о структуре документа, нужная для решения о маршруте OCR.

    has_multiple_columns / has_table_regions — итог быстрой геометрической
    проверки (см. ColumnLayoutAnalyzer.find_column_gap) до полного
    layout-анализа: есть ли в документе структура, которую построчный
    OCR может перепутать.

    is_safety_critical — документ, где перестановка числового значения
    между полями ведёт к материальному риску (допуски, ГОСТы, паспорта
    изделий) — см. историю урока с допуском на вал редуктора.

    page_count — объём документа/потока документов одного типа, на
    который распределяется дополнительная стоимость layout-анализа.
    """

    has_multiple_columns: bool
    has_table_regions: bool
    page_count: int
    is_safety_critical: bool = False


@dataclass
class RoutingDecision:
    """Результат решения LayoutComplexityAdvisor по одному профилю."""

    use_layout_aware: bool
    reason: str
    estimated_extra_cost: float


class LayoutComplexityAdvisor:
    """
    Экономическое решение: пускать документ через простой построчный OCR
    (урок 3.2) или через layout-aware пайплайн (этот урок).

    Три вопроса из текста урока, в этом порядке:
    1. Есть ли в документе структура, которую можно перепутать?
    2. Является ли документ safety-critical?
    3. Если нет — оправдывает ли объём потока дополнительную стоимость?
    """

    def __init__(
        self, cost_per_page_simple: float, cost_per_page_layout_aware: float
    ) -> None:
        # TODO: сохранить оба параметра как атрибуты экземпляра
        # (self.cost_per_page_simple, self.cost_per_page_layout_aware).
        # Ожидается cost_per_page_layout_aware >= cost_per_page_simple,
        # но проверять это в конструкторе не нужно.
        ...

    def estimate_extra_cost(self, profile: PageProfile) -> float:
        """
        Оценить дополнительную стоимость layout-aware пайплайна по
        сравнению с простым OCR для всего объёма profile.page_count.

        TODO: вернуть
            (self.cost_per_page_layout_aware - self.cost_per_page_simple)
            * profile.page_count
        """
        ...

    def decide(
        self, profile: PageProfile, max_acceptable_extra_cost: float
    ) -> RoutingDecision:
        """
        Принять решение о маршруте для одного профиля документа.

        TODO, строго в этом порядке:
        1. Если НЕ profile.has_multiple_columns И НЕ
           profile.has_table_regions:
               вернуть RoutingDecision(
                   use_layout_aware=False,
                   reason="simple_layout_no_benefit",
                   estimated_extra_cost=0.0,
               )
           (в документе нет структуры, которую можно перепутать —
           платить за layout-анализ бессмысленно вне зависимости от
           бюджета).
        2. Иначе, если profile.is_safety_critical:
               extra_cost = self.estimate_extra_cost(profile)
               вернуть RoutingDecision(
                   use_layout_aware=True,
                   reason="safety_critical_overrides_cost",
                   estimated_extra_cost=extra_cost,
               )
           (стоимость ошибки перестановки значения несопоставима со
           стоимостью OCR — экономический расчёт не применяется).
        3. Иначе (сложная структура, документ не критичный):
               extra_cost = self.estimate_extra_cost(profile)
               если extra_cost <= max_acceptable_extra_cost:
                   вернуть RoutingDecision(
                       use_layout_aware=True,
                       reason="cost_justified",
                       estimated_extra_cost=extra_cost,
                   )
               иначе:
                   вернуть RoutingDecision(
                       use_layout_aware=False,
                       reason="cost_exceeds_budget",
                       estimated_extra_cost=extra_cost,
                   )
        """
        ...
