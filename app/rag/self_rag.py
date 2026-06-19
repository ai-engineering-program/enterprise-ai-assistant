from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import openai

__all__ = [
    "RetrieveDecision",
    "RelevanceLabel",
    "SupportLabel",
    "RetrievedChunk",
    "RankedCandidate",
    "RetrieverProtocol",
    "SelfRAGPipeline",
]


class RetrieveDecision(str, Enum):
    """Токен [Retrieve]: нужен ли поиск для данного запроса?"""

    YES = "yes"
    NO = "no"
    CONTINUE = "continue"


class RelevanceLabel(str, Enum):
    """Токен [IsREL]: оценка релевантности фрагмента."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportLabel(str, Enum):
    """Токен [IsSUP]: степень поддержки ответа документом."""

    FULLY = "fully_supported"
    PARTIALLY = "partially_supported"
    NOT = "not_supported"


@dataclass
class RetrievedChunk:
    """Фрагмент документа, возвращённый поисковым компонентом."""

    text: str
    source: str
    score: float = 0.0


@dataclass
class RankedCandidate:
    """Оценённый ответ-кандидат, сгенерированный по одному фрагменту."""

    answer: str
    chunk: RetrievedChunk
    relevance: RelevanceLabel
    support: SupportLabel
    usefulness: int  # 1..5


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Интерфейс поискового компонента, совместимого с SelfRAGPipeline."""

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        ...


# Приоритет для ранжирования кандидатов по токену [IsSUP]
_SUPPORT_RANK: dict[SupportLabel, int] = {
    SupportLabel.FULLY: 2,
    SupportLabel.PARTIALLY: 1,
    SupportLabel.NOT: 0,
}


class SelfRAGPipeline:
    """
    Self-RAG конвейер с явными шагами оценки через языковую модель.

    Каждый шаг оценки ([Retrieve], [IsREL], [IsSUP], [IsUSE]) реализован
    как отдельный вызов языковой модели — без дообучения базовой модели.

    Параметры:
        retriever: объект с методом search(query, top_k) -> list[RetrievedChunk]
        model: идентификатор модели OpenAI (по умолчанию gpt-4o-mini)
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.client = openai.OpenAI()

    # ------------------------------------------------------------------
    # Вспомогательные методы (реализованы — не изменяйте)
    # ------------------------------------------------------------------

    def _llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 64,
    ) -> str:
        """Базовый вызов языковой модели. Используйте во всех TODO-методах."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def generate_answer(self, query: str, chunk: RetrievedChunk) -> str:
        """
        Генерация ответа-кандидата по одному фрагменту.
        Метод уже реализован — используйте его в run().
        """
        return self._llm(
            system=(
                "Ты — точный ассистент. Отвечай ТОЛЬКО на основе "
                "предоставленного документа. Если документ не содержит "
                "ответа — прямо скажи об этом."
            ),
            user=f"Вопрос: {query}\n\nДокумент:\n{chunk.text[:2000]}",
            max_tokens=512,
        )

    # ------------------------------------------------------------------
    # TODO: реализуйте следующие методы
    # ------------------------------------------------------------------

    def decide_retrieve(self, query: str) -> RetrieveDecision:
        """
        Токен [Retrieve]: нужен ли внешний поиск для ответа на запрос?

        Требования:
        - Вызвать self._llm() с промптом-классификатором.
        - Промпт должен требовать ответ ТОЛЬКО одним словом: yes, no или continue.
        - При ValueError (ответ модели не распознан) — вернуть RetrieveDecision.YES.
        - max_tokens для классификации: 16.
        """
        # TODO: реализовать
        ...

    def assess_relevance(
        self, query: str, chunk: RetrievedChunk
    ) -> RelevanceLabel:
        """
        Токен [IsREL]: релевантен ли фрагмент данному запросу?

        Требования:
        - Передавать chunk.text[:1000] (не весь текст).
        - Ожидаемый ответ модели: "relevant" или "irrelevant".
        - При ValueError — вернуть RelevanceLabel.IRRELEVANT (осторожное умолчание).
        - max_tokens: 16.
        """
        # TODO: реализовать
        ...

    def assess_support(
        self, answer: str, chunk: RetrievedChunk
    ) -> SupportLabel:
        """
        Токен [IsSUP]: насколько ответ подтверждается документом?

        Требования:
        - Ожидаемый ответ модели: "fully_supported", "partially_supported"
          или "not_supported".
        - При ValueError — вернуть SupportLabel.NOT.
        - max_tokens: 16.
        """
        # TODO: реализовать
        ...

    def assess_usefulness(self, query: str, answer: str) -> int:
        """
        Токен [IsUSE]: насколько ответ полезен пользователю? Оценка 1–5.

        Требования:
        - Промпт должен требовать ТОЛЬКО цифру от 1 до 5.
        - Ограничить результат: max(1, min(5, int(score_str))).
        - При ValueError — вернуть 1.
        - max_tokens: 8.
        """
        # TODO: реализовать
        ...

    def run(self, query: str, top_k: int = 3) -> dict:
        """
        Основной метод: запускает полный Self-RAG конвейер.

        Алгоритм:
        1. decide_retrieve() — если NO, ответить напрямую (без поиска).
        2. self.retriever.search(query, top_k=top_k).
        3. Для каждого фрагмента: assess_relevance() → пропустить IRRELEVANT.
        4. Для каждого релевантного: generate_answer() → assess_support()
           → assess_usefulness().
        5. Если кандидатов нет — вернуть ответ об отсутствии данных.
        6. Выбрать лучшего кандидата: приоритет по _SUPPORT_RANK,
           затем по usefulness.

        Формат возвращаемого словаря:
        - retrieve_used: bool
        - candidates_evaluated: int
        - answer: str
        - source: str (только если есть кандидаты)
        - support: str (только если есть кандидаты)
        - usefulness: int (только если есть кандидаты)
        """
        # TODO: реализовать
        ...
