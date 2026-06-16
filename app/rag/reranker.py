from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder


__all__ = ["Reranker", "RerankResult"]


@dataclass
class RerankResult:
    """Результат переранжирования с сохранением исходного ранга.

    Attributes:
        text:          текст документа.
        source:        идентификатор источника (имя файла, URL и т.п.).
        rerank_score:  оценка перекрёстной модели (выше = релевантнее).
        original_rank: позиция в исходном списке кандидатов (с 1).
    """

    text: str
    source: str
    rerank_score: float
    original_rank: int


class Reranker:
    """Уточняющее ранжирование через перекрёстную модель (cross-encoder).

    Принимает пул кандидатов из первого этапа поиска (HybridRetriever),
    оценивает каждую пару «запрос + документ» совместно и возвращает
    top_k наиболее релевантных результатов.

    Пример использования:
        reranker = Reranker()
        candidates = hybrid_retriever.retrieve(query, top_k=20)
        results = reranker.rerank(query, candidates, top_k=5)
        context = reranker.build_context(results)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # TODO: сохранить model_name в self.model_name
        # TODO: загрузить модель: self._model = CrossEncoder(model_name)
        ...

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Переранжирует кандидатов с помощью перекрёстной модели.

        Args:
            query:      текст запроса пользователя.
            candidates: список объектов с полями .text и .source
                        (HybridResult, SearchResult или любой совместимый тип).
            top_k:      число результатов в итоговом списке.

        Returns:
            Список RerankResult, отсортированный по убыванию rerank_score.
            Возвращает пустой список если candidates пуст.

        TODO: реализуйте эту функцию.
        Шаги:
        1. Если candidates пуст — вернуть [].
        2. Сформировать список пар: [(query, c.text) for c in candidates].
        3. Вызвать self._model.predict(pairs) — возвращает массив оценок.
        4. Для каждого кандидата создать RerankResult с rerank_score и
           original_rank (enumerate с 1).
        5. Отсортировать results по убыванию rerank_score.
        6. Вернуть первые top_k элементов.
        """
        ...

    def build_context(self, results: list[RerankResult]) -> str:
        """Форматирует переранжированные результаты в строку контекста для LLM.

        Формат каждого документа:
            [Документ N] (source, оценка=X.XXX, исх.позиция=N)
            text

        Документы разделяются двойным переносом строки.
        При пустом списке возвращает «Релевантных документов не найдено.»

        TODO: реализуйте эту функцию.
        """
        ...
