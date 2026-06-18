from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoldenRecord:
    question: str
    reference_answer: str
    reference_contexts: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_questions: int
    per_question: list[dict] = field(default_factory=list)

    def passes_thresholds(self, thresholds: dict[str, float]) -> bool:
        # TODO: вернуть True, если все метрики >= соответствующим порогам.
        # thresholds — словарь вида {"faithfulness": 0.85, "context_precision": 0.70, ...}
        # Если для метрики порог не задан, считать её прошедшей проверку.
        ...

    def to_dict(self) -> dict[str, Any]:
        # TODO: вернуть словарь со всеми полями отчёта.
        # Включить: faithfulness, answer_relevancy, context_precision,
        # context_recall, num_questions, per_question.
        ...


class RagasEvaluator:
    """Конвейер автоматизированной оценки RAG-системы по метрикам RAGAS.

    Использование:
        evaluator = RagasEvaluator.from_json("data/golden_set.json")
        report = evaluator.run(retriever, generator)
        print(report.faithfulness)
    """

    def __init__(
        self,
        golden_records: list[GoldenRecord],
        openai_api_key: str = "",
    ) -> None:
        # TODO: сохранить golden_records в self.golden_records.
        # Если openai_api_key передан, настроить переменную окружения
        # OPENAI_API_KEY (os.environ["OPENAI_API_KEY"] = openai_api_key).
        ...

    @classmethod
    def from_json(cls, path: str, openai_api_key: str = "") -> "RagasEvaluator":
        # TODO: прочитать JSON-файл по пути path.
        # Каждый элемент списка преобразовать в GoldenRecord.
        # Вернуть экземпляр RagasEvaluator.
        # Формат файла:
        # [
        #   {
        #     "question": "...",
        #     "reference_answer": "...",
        #     "reference_contexts": ["...", "..."],
        #     "metadata": {}
        #   }
        # ]
        ...

    def run(self, retriever: Any, generator: Any) -> EvaluationReport:
        # TODO: для каждого вопроса из self.golden_records:
        #   1. Получить фрагменты: chunks = retriever.retrieve(question, top_k=5)
        #   2. Извлечь тексты: contexts = [c["text"] for c in chunks]
        #   3. Получить ответ: answer = generator.generate(question, contexts)
        #
        # Собрать datasets.Dataset из четырёх списков:
        #   question, answer, contexts, ground_truth (= reference_answer)
        #
        # Запустить ragas.evaluate() с метриками:
        #   faithfulness, answer_relevancy, context_precision, context_recall
        #
        # Построить EvaluationReport из результатов evaluate().
        # В per_question сохранить детализацию по каждому вопросу (из result.to_pandas()).
        ...

    def save_report(self, report: EvaluationReport, path: str) -> None:
        # TODO: сохранить report.to_dict() как JSON-файл по пути path.
        # Создать родительские директории при необходимости (pathlib.Path.mkdir).
        ...
