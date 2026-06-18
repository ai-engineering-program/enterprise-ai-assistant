from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class CitationItem:
    """Одна ссылка в ответе модели на конкретный фрагмент."""
    chunk_id: str
    statement: str
    quote: str
    citation_id: int = 0


@dataclass
class AttributedResponse:
    """Ответ языковой модели с атрибутированными источниками."""
    answer: str
    citations: list[CitationItem] = field(default_factory=list)
    unsupported_statements: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class CitationVerification:
    """Результат верификации одной цитаты."""
    chunk_id: str
    quote: str
    is_verified: bool
    similarity_score: float


@dataclass
class VerificationResult:
    """Итог верификации всех цитат в ответе."""
    verified_count: int
    total_count: int
    unverified_citations: list[CitationVerification]

    @property
    def faithfulness_score(self) -> float:
        """Доля подтверждённых утверждений (0.0–1.0)."""
        # TODO: вернуть verified_count / total_count
        # При total_count == 0 вернуть 0.0 (не вызывать ZeroDivisionError)
        ...


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Нормализует текст: нижний регистр, схлопывает пробелы."""
    # TODO: привести text к нижнему регистру,
    # заменить последовательности пробелов одним пробелом,
    # убрать пробелы по краям
    ...


def _token_overlap(a: str, b: str) -> float:
    """Доля токенов из строки a, присутствующих в строке b."""
    # TODO: разбить a и b на множества слов (split)
    # Вернуть len(tokens_a & tokens_b) / len(tokens_a)
    # При пустом a вернуть 0.0
    ...


# ---------------------------------------------------------------------------
# Системный промпт (заполнить корректным шаблоном)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ATTRIBUTION = """
Ты — строгий ассистент. Отвечай ТОЛЬКО на основе предоставленных фрагментов.

ПРАВИЛА:
1. Каждое утверждение из документа должно иметь явную ссылку [N].
2. Если информации в фрагментах нет — укажи это в поле "unsupported".
3. Не добавляй знания из предобучения.

ФОРМАТ ОТВЕТА (строгий JSON):
{{
  "answer": "текст ответа с [1], [2] для ссылок",
  "citations": [
    {{
      "id": 1,
      "chunk_id": "идентификатор фрагмента",
      "statement": "утверждение из ответа",
      "quote": "дословная цитата из фрагмента"
    }}
  ],
  "unsupported": ["утверждение без источника"]
}}

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}
"""


# ---------------------------------------------------------------------------
# Основной класс конвейера
# ---------------------------------------------------------------------------

class AttributionPipeline:
    """
    Конвейер генерации ответов с обязательной атрибуцией источников.

    Параметры:
        client: экземпляр openai.OpenAI (или совместимый клиент)
        model:  название модели (по умолчанию "gpt-4o-mini")
    """

    def __init__(self, client, model: str = "gpt-4o-mini") -> None:
        # TODO: сохранить client и model как атрибуты экземпляра
        ...

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Форматирует фрагменты для подстановки в промпт.

        chunks: список словарей {"chunk_id": str, "text": str}
        Возвращает строку вида:
            [Фрагмент chunk_id_1]
            текст фрагмента 1

            ---

            [Фрагмент chunk_id_2]
            текст фрагмента 2
        """
        # TODO: для каждого chunk сформировать блок
        # "[Фрагмент {chunk_id}]\n{text}"
        # Объединить блоки через "\n\n---\n\n"
        ...

    def _parse_response(self, raw: str) -> AttributedResponse:
        """
        Разбирает JSON-строку ответа модели в AttributedResponse.

        При невалидном JSON возвращает AttributedResponse с пустыми
        citations и unsupported_statements, помещая raw в raw_response.
        """
        # TODO: попытаться json.loads(raw)
        # Если JSONDecodeError — вернуть AttributedResponse(answer=raw, raw_response=raw)
        # Иначе — собрать CitationItem для каждого элемента "citations"
        # и вернуть AttributedResponse с заполненными полями
        ...

    def generate_with_citations(
        self,
        query: str,
        chunks: list[dict],
    ) -> AttributedResponse:
        """
        Генерирует ответ с обязательными ссылками на источники.

        Параметры:
            query:  вопрос пользователя
            chunks: список {"chunk_id": str, "text": str}

        Возвращает AttributedResponse с заполненными citations.
        """
        # TODO: построить context через _build_context
        # Сформировать системный промпт SYSTEM_PROMPT_ATTRIBUTION.format(context=...)
        # Вызвать self.client.chat.completions.create с:
        #   - temperature=0.0 (детерминизм обязателен)
        #   - response_format={"type": "json_object"}
        # Вернуть _parse_response(raw_text)
        ...

    def verify_citations(
        self,
        response: AttributedResponse,
        chunks: list[dict],
    ) -> VerificationResult:
        """
        Верифицирует каждую цитату: содержится ли она в указанном фрагменте?

        Алгоритм:
        1. Построить словарь chunk_id -> text из chunks.
        2. Для каждой CitationItem:
           a. Нормализовать quote и текст фрагмента (_normalize).
           b. Проверить точное вхождение (in).
           c. Если нет — вычислить _token_overlap; is_verified если >= 0.8.
        3. Вернуть VerificationResult с подсчитанными verified_count.
        """
        # TODO: реализовать алгоритм верификации
        ...


__all__ = [
    "CitationItem",
    "AttributedResponse",
    "CitationVerification",
    "VerificationResult",
    "AttributionPipeline",
    "SYSTEM_PROMPT_ATTRIBUTION",
]
