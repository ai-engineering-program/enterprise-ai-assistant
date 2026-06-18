from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


__all__ = [
    "SearchResult",
    "GoldenItem",
    "RetrievalReport",
    "RetrievalAnalyzer",
]


@dataclass
class SearchResult:
    """Один результат поиска из топ-K."""

    doc_id: str
    chunk_id: str
    score: float
    text_preview: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldenItem:
    """Элемент золотого набора для оценки Recall@K."""

    question: str
    expected_doc_id: str


@dataclass
class RetrievalReport:
    """Диагностический отчёт по одному поисковому запросу."""

    query: str
    results: list[SearchResult]
    expected_doc_id: str | None

    # Результаты диагностики
    found: bool = False
    rank: int | None = None
    low_recall: bool = False
    context_pollution: bool = False
    conflicts: list[tuple[SearchResult, SearchResult]] = field(default_factory=list)


class RetrievalAnalyzer:
    """
    Инструмент диагностики поиска.

    Выявляет симптомы низкой полноты и засорения контекста
    по результатам одного запроса, а также считает Recall@K
    по «золотому набору» с известными ответами.
    """

    def analyze(
        self,
        query: str,
        results: list[SearchResult],
        expected_doc_id: str | None = None,
    ) -> RetrievalReport:
        """
        Анализирует результаты поиска и возвращает диагностический отчёт.

        Параметры
        ---------
        query:
            Текст поискового запроса.
        results:
            Список результатов поиска в порядке убывания оценки.
        expected_doc_id:
            Идентификатор документа, который должен присутствовать
            в результатах (задаётся при наличии золотого ответа).

        Возвращает
        ----------
        RetrievalReport с заполненными полями found, rank,
        low_recall, context_pollution, conflicts.
        """
        # TODO: найти expected_doc_id в results
        # TODO: заполнить found=True и rank (1-based) если найден
        # TODO: установить low_recall=True если expected_doc_id задан, но не найден
        # TODO: вызвать self._detect_conflicts(results) для поиска конфликтов версий
        # TODO: установить context_pollution=True если список conflicts непустой
        # TODO: вернуть заполненный RetrievalReport
        ...

    def _detect_conflicts(
        self,
        results: list[SearchResult],
    ) -> list[tuple[SearchResult, SearchResult]]:
        """
        Находит пары фрагментов, относящихся к одному базовому документу,
        но имеющих разные версии — признак засорения контекста.

        Версия берётся из metadata["document_version"],
        базовый идентификатор — из metadata["base_doc_id"].
        Пары с отсутствующими ключами пропускаются.
        """
        # TODO: перебрать все пары (i, j) где i < j
        # TODO: для каждой пары проверить:
        #       - оба содержат "base_doc_id" в metadata
        #       - base_doc_id совпадает
        #       - "document_version" не совпадает
        # TODO: добавить такие пары в список и вернуть его
        ...

    def compute_recall_at_k(
        self,
        golden_set: list[GoldenItem],
        search_fn: Callable[[str, int], list[SearchResult]],
        k: int = 5,
    ) -> float:
        """
        Считает долю запросов из golden_set, для которых
        правильный документ оказался в топ-K результатов.

        Параметры
        ---------
        golden_set:
            Список элементов GoldenItem с известными ответами.
        search_fn:
            Вызываемый объект с сигнатурой (query: str, k: int) -> list[SearchResult].
        k:
            Размер топа для проверки.

        Возвращает
        ----------
        float в диапазоне [0.0, 1.0].
        Возвращает 0.0 при пустом golden_set.
        """
        # TODO: при пустом golden_set вернуть 0.0
        # TODO: для каждого GoldenItem вызвать search_fn(item.question, k)
        # TODO: проверить, содержит ли результат item.expected_doc_id
        # TODO: посчитать долю успешных запросов и вернуть float
        ...
