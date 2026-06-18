from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureLayer(str, Enum):
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    UNKNOWN = "unknown"


class FailureClass(str, Enum):
    # Слой индексирования
    I1_DESTRUCTIVE_CHUNKING = "I-1: Деструктивное разбиение"
    I2_STALE_INDEX = "I-2: Устаревший индекс"
    I3_INCOMPATIBLE_VECTORS = "I-3: Несовместимые векторные представления"
    # Слой поиска
    R1_LOW_RECALL = "R-1: Низкая полнота"
    R2_LOW_PRECISION = "R-2: Низкая точность"
    R3_CONTEXT_OVERFLOW = "R-3: Засорение контекста"
    # Слой генерации
    G1_HALLUCINATION = "G-1: Галлюцинация поверх контекста"
    G2_CONTEXT_IGNORED = "G-2: Игнорирование контекста"
    G3_WRONG_ATTRIBUTION = "G-3: Неверная атрибуция"
    UNKNOWN = "Неизвестно"


@dataclass
class DiagnosticResult:
    failure_class: FailureClass
    layer: FailureLayer
    confidence: float  # 0.0–1.0
    recommendation: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class RAGSymptoms:
    # Слой индексирования
    index_model_mismatch: bool = False
    index_outdated_days: Optional[int] = None
    chunk_boundary_breaks_fact: bool = False

    # Слой поиска
    relevant_doc_in_top_k: Optional[bool] = None
    precision_at_k: Optional[float] = None
    context_tokens_exceed_limit: bool = False

    # Слой генерации
    correct_context_given_wrong_answer: bool = False
    answer_ignores_provided_context: bool = False
    attribution_mismatch: bool = False


class RAGDiagnostics:
    """
    Диагностический инструмент для RAG-систем.

    Принимает набор наблюдаемых симптомов (RAGSymptoms) и возвращает
    список предполагаемых классов отказа с уровнем уверенности
    и рекомендациями по устранению.

    Параметры:
        stale_index_threshold_days: порог устаревания индекса в днях (по умолчанию 7).
        precision_threshold: минимально допустимая точность топ-K (по умолчанию 0.5).
    """

    def __init__(
        self,
        stale_index_threshold_days: int = 7,
        precision_threshold: float = 0.5,
    ) -> None:
        self.stale_index_threshold_days = stale_index_threshold_days
        self.precision_threshold = precision_threshold

    def classify(self, symptoms: RAGSymptoms) -> list[DiagnosticResult]:
        """
        По набору симптомов определяет классы отказа.

        Возвращает список DiagnosticResult — один набор симптомов может
        указывать на несколько классов одновременно.
        При отсутствии симптомов возвращает пустой список.
        """
        # TODO: вызвать _check_indexing_layer, _check_retrieval_layer, _check_generation_layer
        # TODO: объединить результаты и вернуть общий список
        ...

    def _check_indexing_layer(self, symptoms: RAGSymptoms) -> list[DiagnosticResult]:
        """
        Проверяет симптомы слоя индексирования.

        Классы для проверки:
        - I-3: index_model_mismatch == True → уверенность 0.95
        - I-2: index_outdated_days задан и > stale_index_threshold_days → уверенность 0.80
        - I-1: chunk_boundary_breaks_fact == True → уверенность 0.85
        """
        # TODO: реализовать логику классификации для каждого класса
        # TODO: для каждого найденного класса создать DiagnosticResult с непустыми
        #       полями recommendation и evidence
        ...

    def _check_retrieval_layer(self, symptoms: RAGSymptoms) -> list[DiagnosticResult]:
        """
        Проверяет симптомы слоя поиска.

        Классы для проверки:
        - R-1: relevant_doc_in_top_k == False → уверенность 0.90
        - R-2: precision_at_k задан и < precision_threshold → уверенность 0.85
        - R-3: context_tokens_exceed_limit == True → уверенность 0.90
        """
        # TODO: реализовать логику классификации для каждого класса
        ...

    def _check_generation_layer(self, symptoms: RAGSymptoms) -> list[DiagnosticResult]:
        """
        Проверяет симптомы слоя генерации.

        Классы для проверки:
        - G-2: answer_ignores_provided_context == True → уверенность 0.90
        - G-1: correct_context_given_wrong_answer == True
               AND answer_ignores_provided_context == False → уверенность 0.85
        - G-3: attribution_mismatch == True → уверенность 0.80
        """
        # TODO: реализовать логику классификации для каждого класса
        # Подсказка: G-2 проверяйте до G-1 — они различаются по флагу
        # answer_ignores_provided_context
        ...


__all__ = [
    "FailureLayer",
    "FailureClass",
    "DiagnosticResult",
    "RAGSymptoms",
    "RAGDiagnostics",
]
