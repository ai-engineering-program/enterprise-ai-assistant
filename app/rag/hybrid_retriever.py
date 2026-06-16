from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.rag.dense_retriever import DenseRetriever
from app.rag.sparse_retriever import BM25Retriever


__all__ = ["HybridRetriever", "HybridResult", "reciprocal_rank_fusion", "weighted_hybrid_retrieve"]


@dataclass
class HybridResult:
    """Результат гибридного поиска с RRF-оценкой.

    Содержит информацию об источнике сигнала:
      dense_rank  — позиция в списке DenseRetriever (None если не вошёл)
      sparse_rank — позиция в списке BM25Retriever (None если не вошёл)
    """

    text: str
    source: str
    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = 60,
) -> list[HybridResult]:
    """Объединяет два рейтинговых списка через алгоритм слияния рейтинговых списков (RRF).

    Для каждого документа d вычисляется:
        RRF(d) = Σ  1 / (k + rank_i(d))
    где сумма по всем поисковым системам, rank_i — позиция (с 1) в i-м списке.

    Args:
        dense_results:  список SearchResult от DenseRetriever (убывающий порядок score).
        sparse_results: список SearchResult от BM25Retriever (убывающий порядок score).
        k: константа сглаживания, по умолчанию 60 (из оригинальной статьи RRF).

    Returns:
        Список HybridResult, отсортированный по убыванию rrf_score.

    TODO: реализуйте эту функцию.
    Шаги:
    1. Создать словарь scores: source -> {"text", "rrf_score", "dense_rank", "sparse_rank"}.
    2. Пройти по dense_results (enumerate начиная с 1):
       - добавить 1/(k+rank) к rrf_score
       - записать dense_rank
    3. Пройти по sparse_results (enumerate начиная с 1):
       - добавить 1/(k+rank) к rrf_score
       - записать sparse_rank
    4. Собрать список HybridResult из словаря.
    5. Отсортировать по убыванию rrf_score и вернуть.
    """
    ...


def weighted_hybrid_retrieve(
    dense: DenseRetriever,
    sparse: BM25Retriever,
    query: str,
    alpha: float = 0.5,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[HybridResult]:
    """Гибридный поиск через нормализацию оценок и взвешенную сумму.

    Альтернатива RRF: явный контроль вклада каждого метода через параметр alpha.

        hybrid_score = alpha * dense_norm + (1 - alpha) * sparse_norm

    Где dense_norm и sparse_norm — оценки, нормализованные в [0, 1].

    Args:
        dense:      инициализированный и проиндексированный DenseRetriever.
        sparse:     инициализированный и проиндексированный BM25Retriever.
        query:      текст запроса пользователя.
        alpha:      вес dense-поиска [0, 1]. alpha=0 — только BM25, alpha=1 — только Dense.
        top_k:      количество итоговых результатов.
        candidate_k: число кандидатов, запрашиваемых у каждого поисковика.

    Returns:
        Список HybridResult, отсортированный по убыванию hybrid_score (записанного в rrf_score).

    TODO: реализуйте эту функцию.
    Шаги:
    1. Получить candidate_k результатов от dense и sparse.
    2. Для каждого из двух списков нормализовать оценки в [0, 1]:
       norm = (score - min_score) / (max_score - min_score + 1e-9)
    3. Построить словарь source -> {"text", "dense_norm", "sparse_norm"}.
    4. Вычислить hybrid_score = alpha * dense_norm + (1 - alpha) * sparse_norm.
    5. Вернуть top_k результатов как HybridResult (записать hybrid_score в rrf_score).
    """
    ...


class HybridRetriever:
    """Гибридный поиск: объединение DenseRetriever и BM25Retriever через RRF.

    Оба ретривера должны быть проиндексированы (index_documents вызван)
    до вызова retrieve().

    Пример использования:
        dense = DenseRetriever(collection_name="hr_docs")
        sparse = BM25Retriever()
        documents = [{"text": "...", "source": "..."}]
        dense.index_documents(documents)
        sparse.index_documents(documents)

        hybrid = HybridRetriever(dense=dense, sparse=sparse)
        results = hybrid.retrieve("оформить отпуск по 123-ФЗ", top_k=5)
        context = hybrid.build_context(results)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        # TODO: сохранить dense, sparse, rrf_k, candidate_k как атрибуты экземпляра
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HybridResult]:
        """Выполняет гибридный поиск и возвращает top_k результатов.

        Запрашивает candidate_k кандидатов от каждого поисковика,
        объединяет через RRF и возвращает первые top_k.

        Args:
            query:  текст запроса пользователя.
            top_k:  количество итоговых результатов.

        Returns:
            Список HybridResult, отсортированный по убыванию rrf_score.

        TODO: реализуйте эту функцию.
        Шаги:
        1. Вызвать self.dense.retrieve(query, top_k=self.candidate_k).
        2. Вызвать self.sparse.retrieve(query, top_k=self.candidate_k).
        3. Вызвать reciprocal_rank_fusion(..., k=self.rrf_k).
        4. Вернуть первые top_k элементов результата.
        """
        ...

    def build_context(self, results: list[HybridResult]) -> str:
        """Форматирует результаты в строку контекста для LLM.

        Формат каждого документа:
            [Документ N] (source, d#dense_rank s#sparse_rank)
            text

        Если dense_rank или sparse_rank равен None — пропустить соответствующую часть.
        Документы разделяются двойным переносом строки.
        При пустом списке вернуть «Релевантных документов не найдено.»

        TODO: реализуйте эту функцию.
        """
        ...
