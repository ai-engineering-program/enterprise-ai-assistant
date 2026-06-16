from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


__all__ = [
    "SearchResult",
    "QueryGenerator",
    "MockQueryGenerator",
    "deduplicate_results",
    "MultiQueryRetriever",
    "SubqueryDecomposer",
    "TransformMode",
    "QueryTransformer",
]


# ---------------------------------------------------------------------------
# Общие типы
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Унифицированный результат поиска.

    Attributes:
        text:   текст документа (чанка).
        source: уникальный идентификатор документа (имя файла, URL и т.п.).
        score:  оценка релевантности (выше — лучше).
    """

    text: str
    source: str
    score: float


class Retriever(Protocol):
    """Протокол совместимости для любого поисковика."""

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ...


# ---------------------------------------------------------------------------
# Протокол и реализации генераторов запросов
# ---------------------------------------------------------------------------

class QueryGenerator(Protocol):
    """Протокол для компонента, преобразующего запрос в список запросов."""

    def generate(self, query: str, n: int = 3) -> list[str]:
        """Генерирует n перефразировок исходного запроса.

        Возвращает список, включающий исходный запрос в первом элементе.
        """
        ...

    def decompose(self, query: str) -> list[str]:
        """Разбивает составной запрос на список простых подзапросов."""
        ...


class MockQueryGenerator:
    """Детерминированный генератор запросов для тестов (без вызова LLM).

    generate() добавляет к исходному запросу фиксированные суффиксы.
    decompose() разбивает запрос по слову «и» или возвращает два подзапроса.
    """

    def generate(self, query: str, n: int = 3) -> list[str]:
        """Возвращает исходный запрос и n вариантов с суффиксами."""
        variants = [query]
        suffixes = [" (вариант 1)", " (вариант 2)", " (вариант 3)"]
        for i in range(min(n, len(suffixes))):
            variants.append(query + suffixes[i])
        return variants

    def decompose(self, query: str) -> list[str]:
        """Разбивает запрос по союзу «и» или возвращает два подзапроса."""
        parts = [p.strip() for p in query.split(" и ") if p.strip()]
        if len(parts) >= 2:
            return parts
        # если разбить не получилось — добавляем второй подзапрос-суффикс
        return [query, query + " диагностика"]


class OpenAIQueryGenerator:
    """Генератор запросов через OpenAI API.

    Требует переменной окружения OPENAI_API_KEY.
    Используйте только в интеграционных тестах.
    """

    MULTI_QUERY_PROMPT = (
        "Ты помощник, который генерирует разные формулировки вопроса.\n"
        "Исходный вопрос: {query}\n\n"
        "Сгенерируй {n} альтернативных формулировок того же вопроса. "
        "Используй разные слова и синонимы, но сохраняй смысл. "
        "Каждую формулировку выведи с новой строки, без нумерации."
    )

    DECOMPOSE_PROMPT = (
        "Ты эксперт по анализу вопросов.\n"
        "Разбей следующий сложный вопрос на простые самостоятельные подвопросы. "
        "Каждый подвопрос должен касаться одной конкретной темы. "
        "Если вопрос уже простой — верни его как есть (один элемент).\n\n"
        "Вопрос: {query}\n\n"
        "Выведи каждый подвопрос с новой строки, без нумерации и маркеров."
    )

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        # TODO: сохранить model в self.model
        # TODO: инициализировать self._client = OpenAI()
        #       (импорт: from openai import OpenAI)
        ...

    def generate(self, query: str, n: int = 3) -> list[str]:
        """Генерирует n перефразировок через OpenAI API.

        TODO: реализуйте этот метод.
        Шаги:
        1. Вызвать self._client.chat.completions.create() с MULTI_QUERY_PROMPT.
        2. Разбить ответ по строкам, убрать пустые.
        3. Вернуть [query] + variants[:n].
        """
        ...

    def decompose(self, query: str) -> list[str]:
        """Разбивает составной запрос на подзапросы через OpenAI API.

        TODO: реализуйте этот метод.
        Шаги:
        1. Вызвать self._client.chat.completions.create() с DECOMPOSE_PROMPT,
           temperature=0.2 (нужна точность, не разнообразие).
        2. Разбить ответ по строкам, убрать пустые.
        3. Вернуть subqueries или [query] если список пустой.
        """
        ...


# ---------------------------------------------------------------------------
# Утилита дедупликации
# ---------------------------------------------------------------------------

def deduplicate_results(
    all_results: list[list[SearchResult]],
) -> list[SearchResult]:
    """Объединяет результаты нескольких поисков, убирает дубликаты.

    Ключ дедупликации — поле source. При дублировании сохраняет
    экземпляр с наибольшим score. Итоговый список отсортирован
    по убыванию score.

    TODO: реализуйте эту функцию.
    Шаги:
    1. Создать dict[str, SearchResult] seen = {}.
    2. Пройти по all_results -> results -> result:
       если result.source не в seen или result.score > seen[result.source].score
       — записать result в seen[result.source].
    3. Вернуть sorted(seen.values(), key=lambda r: r.score, reverse=True).
    """
    ...


# ---------------------------------------------------------------------------
# MultiQueryRetriever
# ---------------------------------------------------------------------------

class MultiQueryRetriever:
    """Поисковик с множественными формулировками запроса.

    Генерирует n вариантов исходного запроса, выполняет поиск для каждого
    и возвращает дедуплицированный список результатов.

    Пример:
        generator = MockQueryGenerator()
        retriever = BM25Retriever()
        retriever.index_documents(docs)
        mq = MultiQueryRetriever(retriever=retriever, query_generator=generator)
        results = mq.retrieve("комиссия при возврате", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        query_generator: QueryGenerator,
        n_variants: int = 3,
    ) -> None:
        # TODO: сохранить retriever в self.retriever
        # TODO: сохранить query_generator в self.query_generator
        # TODO: сохранить n_variants в self.n_variants
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Поиск с множественными формулировками.

        TODO: реализуйте этот метод.
        Шаги:
        1. Вызвать self.query_generator.generate(query, n=self.n_variants)
           — получить список строк variants.
        2. Для каждого варианта вызвать self.retriever.retrieve(v, top_k=top_k)
           и собрать список all_results.
        3. Вызвать deduplicate_results(all_results).
        4. Вернуть merged[:top_k].
        """
        ...


# ---------------------------------------------------------------------------
# SubqueryDecomposer
# ---------------------------------------------------------------------------

class SubqueryDecomposer:
    """Поисковик с декомпозицией составного запроса на подзапросы.

    Разбивает сложный вопрос на простые части, выполняет поиск для каждой
    и объединяет результаты с дедупликацией.

    Пример:
        decomposer = SubqueryDecomposer(
            retriever=sparse,
            query_generator=MockQueryGenerator(),
            top_k_per_subquery=5,
        )
        results = decomposer.retrieve("задержки и поды падают в k8s", top_k=10)
    """

    def __init__(
        self,
        retriever: Retriever,
        query_generator: QueryGenerator,
        top_k_per_subquery: int = 5,
        max_total: int = 15,
    ) -> None:
        # TODO: сохранить retriever, query_generator, top_k_per_subquery, max_total
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Поиск с декомпозицией на подзапросы.

        TODO: реализуйте этот метод.
        Шаги:
        1. Вызвать self.query_generator.decompose(query)
           — получить список подзапросов subqueries.
        2. Для каждого подзапроса вызвать self.retriever.retrieve(sq,
           top_k=self.top_k_per_subquery) и собрать all_results.
        3. Дедупликация: merged = deduplicate_results(all_results).
        4. Вернуть merged[:top_k].
        """
        ...


# ---------------------------------------------------------------------------
# QueryTransformer: единый интерфейс
# ---------------------------------------------------------------------------

class TransformMode(str, Enum):
    NONE = "none"
    MULTI_QUERY = "multi_query"
    DECOMPOSE = "decompose"
    HYDE = "hyde"


@dataclass
class TransformConfig:
    mode: TransformMode = TransformMode.NONE
    n_variants: int = 3
    top_k_per_subquery: int = 5
    llm_model: str = "gpt-4o-mini"


class QueryTransformer:
    """Единая точка управления преобразованиями запросов.

    При mode=NONE работает как прозрачная обёртка вокруг retriever.
    При других режимах применяет соответствующий паттерн.

    Пример:
        cfg = TransformConfig(mode=TransformMode.MULTI_QUERY, n_variants=3)
        transformer = QueryTransformer(retriever=sparse, config=cfg,
                                       query_generator=MockQueryGenerator())
        results = transformer.retrieve("запрос пользователя", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        config: TransformConfig = field(default_factory=TransformConfig),
        query_generator: QueryGenerator | None = None,
    ) -> None:
        # TODO: сохранить retriever в self.retriever
        # TODO: сохранить config в self.config
        # TODO: сохранить query_generator в self.query_generator
        #       (если None и mode != NONE — raise ValueError с понятным сообщением)
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Поиск с применением выбранного паттерна трансформации.

        TODO: реализуйте этот метод.
        Шаги:
        1. Если self.config.mode == TransformMode.NONE:
           вернуть self.retriever.retrieve(query, top_k=top_k).
        2. Если MULTI_QUERY: использовать MultiQueryRetriever.
        3. Если DECOMPOSE: использовать SubqueryDecomposer.
        4. Если HYDE: вызвать self.query_generator.decompose(query)[0]
           как «гипотетический документ» (упрощённая версия без реального LLM).
        """
        ...
