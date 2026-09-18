from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from app.context.context_summarizer import estimate_tokens


__all__ = [
    "BudgetCategory",
    "ContextComponent",
    "CategoryAllocation",
    "BudgetReport",
    "DEFAULT_CATEGORY_WEIGHTS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "TokenBudgetAllocator",
]


class BudgetCategory(Enum):
    """
    Категории содержимого промпта, конкурирующие за одно и то же
    контекстное окно модели (см. текст урока 6.1, раздел "что на самом
    деле занимает бюджет токенов"). Резерв на ОТВЕТ модели сюда
    сознательно не входит — это не содержимое промпта, а место,
    вычитаемое из общего бюджета ДО деления между этими категориями
    (см. TokenBudgetAllocator.content_budget_tokens).
    """

    SYSTEM_PROMPT = "system_prompt"
    TOOL_DEFINITIONS = "tool_definitions"
    RETRIEVED_CONTEXT = "retrieved_context"       # RetrievalPlanner (M3) + компрессия (M4)
    SESSION_HISTORY = "session_history"           # SessionMemory (5.1)
    LONG_TERM_FACTS = "long_term_facts"            # LongTermMemoryStore (5.2), FactJournal (5.4)
    EPISODIC_MEMORY = "episodic_memory"             # EpisodicMemoryStore (5.3)
    USER_MESSAGE = "user_message"


@dataclass(frozen=True)
class ContextComponent:
    """
    Один кусок содержимого промпта, готовый к отправке модели — уже
    прошедший через RetrievalPlanner/компрессию (M3-M4) или через
    соответствующий компонент памяти (M5). TokenBudgetAllocator НЕ решает,
    что попало в text, — он только проверяет, укладывается ли это в
    бюджет своей категории.

    В одном вызове build_report на одну категорию обычно приходится один
    ContextComponent (например, весь retrieved context уже склеен в одну
    строку компрессией модуля M4) — список компонентов допускает и
    несколько записей на категорию, они суммируются.
    """

    category: BudgetCategory
    text: str


@dataclass(frozen=True)
class CategoryAllocation:
    """Итог сравнения бюджета и факта для ОДНОЙ категории."""

    category: BudgetCategory
    allocated_tokens: int
    actual_tokens: int
    over_budget: bool
    overflow_tokens: int


@dataclass
class BudgetReport:
    """
    Итог одного вызова TokenBudgetAllocator.build_report().

    categories_needing_truncation — категории, у которых actual_tokens >
    allocated_tokens, в порядке появления в списке components, переданном
    в build_report (см. текст урока 6.1, раздел "от жадного заполнения к
    бюджету по категориям"). Этот список — вход для урока 6.2
    "Приоритизация" (какую именно категорию обрезать первой и как).
    """

    context_window_tokens: int
    reserved_output_tokens: int
    content_budget_tokens: int
    allocations: list[CategoryAllocation]
    total_actual_tokens: int
    fits: bool
    categories_needing_truncation: list[BudgetCategory] = field(default_factory=list)


# Процентное распределение оставшегося (после вычета резерва на ответ)
# бюджета между категориями — отправная точка, а не универсальная
# константа (см. текст урока, таблица "процентный vs приоритетный
# подход"). Сумма весов равна 1.0 — весь content_budget_tokens
# распределяется без остатка.
DEFAULT_CATEGORY_WEIGHTS: dict[BudgetCategory, float] = {
    BudgetCategory.SYSTEM_PROMPT: 0.05,
    BudgetCategory.TOOL_DEFINITIONS: 0.05,
    BudgetCategory.RETRIEVED_CONTEXT: 0.35,
    BudgetCategory.SESSION_HISTORY: 0.15,
    BudgetCategory.LONG_TERM_FACTS: 0.10,
    BudgetCategory.EPISODIC_MEMORY: 0.10,
    BudgetCategory.USER_MESSAGE: 0.20,
}

# Резерв под генерацию ответа модели (см. текст урока, раздел про
# "регулярно забывается") — вычитается из context_window_tokens ДО того,
# как остаток делится между категориями промпта.
DEFAULT_RESERVED_OUTPUT_TOKENS: int = 1024


class TokenBudgetAllocator:
    """
    Модуль M6, урок 6.1: детерминированный слой ПОСЛЕ RetrievalPlanner +
    компрессии (M3-M4) и компонентов памяти (M5), ПЕРЕД финальной сборкой
    промпта. Не решает, что попало в контекст (это уже сделали M3-M5), и
    не обрезает содержимое сам (это урок 6.2) — его единственная задача:
    превратить вопрос "поместится ли всё это в контекстное окно?" в
    проверяемый расчёт бюджета по категориям ДО отправки запроса модели,
    а не в угадывание постфактум (см. инцидент урока, раздел 1).

    Работает без единого вызова LLM и без токенизатора конкретной модели:
    token_counter по умолчанию — estimate_tokens из ContextSummarizer
    (урок 4.1), та же эвристика "~4 символа на токен". Для точного
    биллинга в production сюда подставляется настоящий токенизатор модели
    через тот же dependency injection, что использовался у
    ContextSummarizer и RetrievalPlanner.
    """

    def __init__(
        self,
        context_window_tokens: int,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
        category_weights: Optional[dict[BudgetCategory, float]] = None,
        token_counter: Callable[[str], int] = estimate_tokens,
    ) -> None:
        """
        TODO:
        1. Если category_weights is None -> использовать
           dict(DEFAULT_CATEGORY_WEIGHTS) (копию, а не сам модуль-уровневый
           словарь — чтобы изменение экземпляра не портило константу).
        2. Валидация (обе проверки — до сохранения полей, поднимать
           ValueError с понятным сообщением):
           - reserved_output_tokens >= context_window_tokens ->
             ValueError ("резерв на ответ не может занимать весь бюджет
             или больше").
           - sum(category_weights.values()) > 1.0 (с учётом небольшой
             погрешности float, например > 1.0 + 1e-6) -> ValueError
             ("сумма весов категорий не может превышать 1.0").
        3. Сохранить все параметры в self._context_window_tokens,
           self._reserved_output_tokens, self._category_weights,
           self._token_counter.
        """
        ...

    @property
    def content_budget_tokens(self) -> int:
        """
        TODO: вернуть self._context_window_tokens -
        self._reserved_output_tokens — бюджет, доступный ДЛЯ СОДЕРЖИМОГО
        промпта, уже без учёта резерва на ответ модели (см. текст урока,
        раздел "резерв на ответ... не ещё одна категория содержимого").
        """
        ...

    def allocate_category_budgets(self) -> dict[BudgetCategory, int]:
        """
        Разбить content_budget_tokens между категориями согласно
        self._category_weights.

        TODO: вернуть {category: int(weight * self.content_budget_tokens)
        for category, weight in self._category_weights.items()}.
        Целочисленное усечение (не округление) — сознательный выбор:
        бюджет категории не должен случайно оказаться БОЛЬШЕ, чем
        позволяет её доля, из-за округления вверх.

        Категории, отсутствующие в self._category_weights, в этот словарь
        не попадают — что моделирует ровно ту ошибку из текста урока,
        когда категорию забыли включить в бюджет вообще: у неё нет
        выделенного места, и build_report должен обработать это как
        allocated_tokens == 0, а не как отсутствие проверки.
        """
        ...

    def build_report(self, components: list[ContextComponent]) -> BudgetReport:
        """
        Построить полный отчёт: бюджет vs факт для каждой категории,
        встретившейся в components, и сигнал "что нужно обрезать", если
        что-то превышает СВОЙ бюджет.

        TODO:
        1. category_budgets = self.allocate_category_budgets()
        2. Для каждого component в components (в исходном порядке):
           - actual = self._token_counter(component.text)
           - allocated = category_budgets.get(component.category, 0)
             (0, если категория не была включена в category_weights —
             см. docstring allocate_category_budgets)
           - over_budget = actual > allocated
           - overflow_tokens = max(0, actual - allocated)
           - собрать CategoryAllocation(category=component.category,
             allocated_tokens=allocated, actual_tokens=actual,
             over_budget=over_budget, overflow_tokens=overflow_tokens)

           Если одна и та же категория встречается в components несколько
           раз — суммировать actual_tokens ПЕРЕД сравнением с allocated
           (то есть считать общий факт по категории, а не создавать
           несколько отдельных CategoryAllocation на одну категорию).
        3. total_actual = сумма actual_tokens по ВСЕМ components (после
           объединения по категориям на шаге 2).
        4. fits = total_actual <= self.content_budget_tokens.
        5. categories_needing_truncation = список category из allocations
           (в порядке первого появления соответствующей категории в
           components), где over_budget is True.
        6. Вернуть BudgetReport(
             context_window_tokens=self._context_window_tokens,
             reserved_output_tokens=self._reserved_output_tokens,
             content_budget_tokens=self.content_budget_tokens,
             allocations=<список CategoryAllocation из шага 2,
               по одной записи на категорию>,
             total_actual_tokens=total_actual,
             fits=fits,
             categories_needing_truncation=categories_needing_truncation,
           )
        """
        ...
