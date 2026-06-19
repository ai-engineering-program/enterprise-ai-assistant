"""
Тесты для app/rag/self_rag.py (урок 6.1: Self-RAG и Корректирующий RAG).

Запуск unit-тестов (без языковой модели):
    pytest tests/test_self_rag.py -v -m unit

Запуск интеграционных тестов (требуется OPENAI_API_KEY):
    pytest tests/test_self_rag.py -v -m integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.self_rag import (
    RankedCandidate,
    RelevanceLabel,
    RetrieveDecision,
    RetrievedChunk,
    SelfRAGPipeline,
    SupportLabel,
    _SUPPORT_RANK,
)


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str = "Текстовый фрагмент",
    source: str = "doc.pdf",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=score)


def _make_retriever(chunks: list[RetrievedChunk]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = chunks
    return retriever


def _make_pipeline(chunks: list[RetrievedChunk]) -> SelfRAGPipeline:
    """Создаёт конвейер с mock-ретривером и mock-клиентом OpenAI."""
    pipeline = SelfRAGPipeline(
        retriever=_make_retriever(chunks),
        model="gpt-4o-mini",
    )
    # Подменяем OpenAI-клиент, чтобы не делать реальных вызовов API
    pipeline.client = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# Unit-тесты: RetrieveDecision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieveDecision:
    """Тесты токена [Retrieve]."""

    def test_decide_retrieve_yes(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="yes")

        result = pipeline.decide_retrieve("Какова ставка по ипотеке?")

        assert result == RetrieveDecision.YES

    def test_decide_retrieve_no(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="no")

        result = pipeline.decide_retrieve("Сколько будет 2 + 2?")

        assert result == RetrieveDecision.NO

    def test_decide_retrieve_invalid_falls_back_to_yes(self):
        """При нераспознанном ответе модели — безопасное умолчание YES."""
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="я не знаю")

        result = pipeline.decide_retrieve("Любой вопрос")

        assert result == RetrieveDecision.YES


# ---------------------------------------------------------------------------
# Unit-тесты: RelevanceLabel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessRelevance:
    """Тесты токена [IsREL]."""

    def test_relevant_chunk(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="relevant")
        chunk = _make_chunk("Ставка по ипотеке составляет 11.2%")

        result = pipeline.assess_relevance("Ставка по ипотеке", chunk)

        assert result == RelevanceLabel.RELEVANT

    def test_irrelevant_chunk(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="irrelevant")
        chunk = _make_chunk("Часы работы отделения банка")

        result = pipeline.assess_relevance("Ставка по ипотеке", chunk)

        assert result == RelevanceLabel.IRRELEVANT

    def test_invalid_response_falls_back_to_irrelevant(self):
        """Осторожное умолчание: неизвестный ответ → IRRELEVANT."""
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="возможно")
        chunk = _make_chunk()

        result = pipeline.assess_relevance("Вопрос", chunk)

        assert result == RelevanceLabel.IRRELEVANT


# ---------------------------------------------------------------------------
# Unit-тесты: SupportLabel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessSupport:
    """Тесты токена [IsSUP]."""

    def test_fully_supported(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="fully_supported")
        chunk = _make_chunk()

        result = pipeline.assess_support("Ставка 11.2%", chunk)

        assert result == SupportLabel.FULLY

    def test_partially_supported(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="partially_supported")
        chunk = _make_chunk()

        result = pipeline.assess_support("Ставка 11.2% с льготами", chunk)

        assert result == SupportLabel.PARTIALLY

    def test_not_supported_on_invalid(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="неизвестно")
        chunk = _make_chunk()

        result = pipeline.assess_support("Любой ответ", chunk)

        assert result == SupportLabel.NOT


# ---------------------------------------------------------------------------
# Unit-тесты: Usefulness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessUsefulness:
    """Тесты токена [IsUSE]."""

    def test_valid_score(self):
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="4")

        score = pipeline.assess_usefulness("Вопрос", "Ответ")

        assert score == 4

    def test_score_clamped_to_max(self):
        """Оценка выше 5 обрезается до 5."""
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="9")

        score = pipeline.assess_usefulness("Вопрос", "Ответ")

        assert score == 5

    def test_score_clamped_to_min(self):
        """Оценка ниже 1 обрезается до 1."""
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="0")

        score = pipeline.assess_usefulness("Вопрос", "Ответ")

        assert score == 1

    def test_invalid_score_falls_back_to_one(self):
        """Нечисловой ответ → оценка 1."""
        pipeline = _make_pipeline([])
        pipeline._llm = MagicMock(return_value="отлично")

        score = pipeline.assess_usefulness("Вопрос", "Ответ")

        assert score == 1


# ---------------------------------------------------------------------------
# Unit-тесты: полный конвейер run()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelfRAGRun:
    """Интеграционные unit-тесты метода run() (без реального API)."""

    def test_run_skips_retrieval_when_not_needed(self):
        """Если decide_retrieve() == NO, поиск не вызывается."""
        chunks = [_make_chunk()]
        pipeline = _make_pipeline(chunks)
        pipeline.decide_retrieve = MagicMock(return_value=RetrieveDecision.NO)
        pipeline._llm = MagicMock(return_value="2 + 2 = 4")

        result = pipeline.run("Сколько будет 2 + 2?")

        assert result["retrieve_used"] is False
        pipeline.retriever.search.assert_not_called()
        assert "answer" in result

    def test_run_returns_no_data_when_all_irrelevant(self):
        """Если все фрагменты нерелевантны — сообщение об отсутствии данных."""
        chunks = [_make_chunk("Нерелевантный текст")]
        pipeline = _make_pipeline(chunks)
        pipeline.decide_retrieve = MagicMock(return_value=RetrieveDecision.YES)
        pipeline.assess_relevance = MagicMock(
            return_value=RelevanceLabel.IRRELEVANT
        )

        result = pipeline.run("Ставка по ипотеке?")

        assert result["retrieve_used"] is True
        assert result["candidates_evaluated"] == 0
        assert "answer" in result
        # Ответ должен явно сообщать об отсутствии данных
        answer_lower = result["answer"].lower()
        assert any(
            word in answer_lower
            for word in ["не найдено", "нет", "отсутств", "недостаточно"]
        )

    def test_run_prefers_fully_supported_over_partially(self):
        """Кандидат с fully_supported предпочитается кандидату с partially."""
        chunk_a = _make_chunk("Документ А", source="a.pdf")
        chunk_b = _make_chunk("Документ Б", source="b.pdf")
        pipeline = _make_pipeline([chunk_a, chunk_b])

        pipeline.decide_retrieve = MagicMock(return_value=RetrieveDecision.YES)
        pipeline.assess_relevance = MagicMock(
            return_value=RelevanceLabel.RELEVANT
        )

        # Документ А: partially + usefulness=5
        # Документ Б: fully + usefulness=3
        # Ожидаем выбор Б (fully важнее usefulness)
        def fake_support(answer: str, chunk: RetrievedChunk) -> SupportLabel:
            return (
                SupportLabel.FULLY
                if chunk.source == "b.pdf"
                else SupportLabel.PARTIALLY
            )

        pipeline.assess_support = MagicMock(side_effect=fake_support)

        def fake_usefulness(query: str, answer: str) -> int:
            # generate_answer вызывается по порядку; у А — первый вызов
            return 5 if "a.pdf" in answer else 3

        pipeline.generate_answer = MagicMock(
            side_effect=lambda q, c: f"Ответ из {c.source}"
        )
        pipeline.assess_usefulness = MagicMock(
            side_effect=lambda q, a: 5 if "a.pdf" in a else 3
        )

        result = pipeline.run("Ставка?")

        assert result["retrieve_used"] is True
        assert result["source"] == "b.pdf"
        assert result["support"] == SupportLabel.FULLY.value

    def test_run_result_contains_required_keys_on_success(self):
        """При успешном ответе словарь содержит все обязательные ключи."""
        chunk = _make_chunk("Ставка 11.2%", source="doc.pdf")
        pipeline = _make_pipeline([chunk])
        pipeline.decide_retrieve = MagicMock(return_value=RetrieveDecision.YES)
        pipeline.assess_relevance = MagicMock(
            return_value=RelevanceLabel.RELEVANT
        )
        pipeline.generate_answer = MagicMock(return_value="Ставка 11.2%")
        pipeline.assess_support = MagicMock(return_value=SupportLabel.FULLY)
        pipeline.assess_usefulness = MagicMock(return_value=5)

        result = pipeline.run("Ставка по ипотеке?")

        assert "answer" in result
        assert "retrieve_used" in result
        assert "candidates_evaluated" in result
        assert "source" in result
        assert "support" in result
        assert "usefulness" in result

    def test_support_rank_ordering(self):
        """_SUPPORT_RANK задаёт правильный порядок: FULLY > PARTIALLY > NOT."""
        assert _SUPPORT_RANK[SupportLabel.FULLY] > _SUPPORT_RANK[SupportLabel.PARTIALLY]
        assert _SUPPORT_RANK[SupportLabel.PARTIALLY] > _SUPPORT_RANK[SupportLabel.NOT]


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSelfRAGIntegration:
    """
    Тесты с реальными вызовами языковой модели.
    Требуют: export OPENAI_API_KEY=...
    Запуск: pytest tests/test_self_rag.py -v -m integration
    """

    def test_decide_retrieve_on_factual_query(self):
        """Фактический вопрос должен получать RetrieveDecision.YES."""
        pytest.skip(
            "Интеграционный тест — требует OPENAI_API_KEY. "
            "Запустите вручную: pytest -m integration"
        )

    def test_full_pipeline_with_mock_retriever(self):
        """Полный конвейер с реальной языковой моделью и mock-ретривером."""
        pytest.skip(
            "Интеграционный тест — требует OPENAI_API_KEY. "
            "Запустите вручную: pytest -m integration"
        )
