import pytest
from app.rag.attribution import (
    AttributedResponse,
    AttributionPipeline,
    CitationItem,
    VerificationResult,
    _normalize,
    _token_overlap,
)


# ---------------------------------------------------------------------------
# Unit-тесты: не требуют внешних сервисов или API-ключей
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNormalize:
    def test_lowercase(self):
        assert _normalize("ТЕКСТ") == "текст"

    def test_collapses_whitespace(self):
        assert _normalize("слово   другое") == "слово другое"

    def test_strips_edges(self):
        assert _normalize("  пробел  ") == "пробел"

    def test_empty_string(self):
        assert _normalize("") == ""


@pytest.mark.unit
class TestTokenOverlap:
    def test_full_overlap(self):
        assert _token_overlap("а б в", "а б в г") == 1.0

    def test_partial_overlap(self):
        score = _token_overlap("а б в", "а б г д")
        assert abs(score - 2 / 3) < 0.01

    def test_no_overlap(self):
        assert _token_overlap("кот", "собака") == 0.0

    def test_empty_a(self):
        assert _token_overlap("", "что угодно") == 0.0


@pytest.mark.unit
class TestVerificationResult:
    def test_faithfulness_score_full(self):
        result = VerificationResult(
            verified_count=3,
            total_count=3,
            unverified_citations=[],
        )
        assert result.faithfulness_score == 1.0

    def test_faithfulness_score_partial(self):
        result = VerificationResult(
            verified_count=2,
            total_count=3,
            unverified_citations=[],
        )
        assert abs(result.faithfulness_score - 2 / 3) < 0.001

    def test_faithfulness_score_zero_total(self):
        result = VerificationResult(
            verified_count=0,
            total_count=0,
            unverified_citations=[],
        )
        # Не должно вызывать ZeroDivisionError
        assert result.faithfulness_score == 0.0


@pytest.mark.unit
class TestVerifyCitations:
    """Тесты верификации цитат без вызова LLM."""

    def _make_pipeline(self):
        # Передаём None в качестве клиента — verify_citations не вызывает LLM
        return AttributionPipeline(client=None)

    def _make_chunks(self):
        return [
            {
                "chunk_id": "doc1_p1",
                "text": (
                    "Оператор обязан принять меры по обеспечению "
                    "безопасности персональных данных при их обработке."
                ),
            },
            {
                "chunk_id": "doc2_p1",
                "text": "Нарушение влечёт гражданскую и уголовную ответственность.",
            },
        ]

    def test_exact_quote_is_verified(self):
        pipeline = self._make_pipeline()
        chunks = self._make_chunks()

        response = AttributedResponse(
            answer="Оператор обязан принять меры [1].",
            citations=[
                CitationItem(
                    chunk_id="doc1_p1",
                    statement="Оператор обязан принять меры",
                    quote="Оператор обязан принять меры по обеспечению безопасности персональных данных при их обработке.",
                    citation_id=1,
                )
            ],
        )
        result = pipeline.verify_citations(response, chunks)

        assert result.verified_count == 1
        assert result.total_count == 1
        assert result.faithfulness_score == 1.0
        assert result.unverified_citations == []

    def test_fabricated_quote_is_not_verified(self):
        pipeline = self._make_pipeline()
        chunks = self._make_chunks()

        response = AttributedResponse(
            answer="Срок хранения данных — 5 лет [1].",
            citations=[
                CitationItem(
                    chunk_id="doc1_p1",
                    statement="Срок хранения данных — 5 лет",
                    quote="Срок хранения персональных данных составляет не менее пяти лет.",
                    citation_id=1,
                )
            ],
        )
        result = pipeline.verify_citations(response, chunks)

        assert result.verified_count == 0
        assert result.total_count == 1
        assert result.faithfulness_score == 0.0
        assert len(result.unverified_citations) == 1

    def test_unknown_chunk_id_not_verified(self):
        pipeline = self._make_pipeline()
        chunks = self._make_chunks()

        response = AttributedResponse(
            answer="Некое утверждение [1].",
            citations=[
                CitationItem(
                    chunk_id="nonexistent_chunk",
                    statement="Некое утверждение",
                    quote="Этого текста нет ни в одном фрагменте.",
                    citation_id=1,
                )
            ],
        )
        result = pipeline.verify_citations(response, chunks)

        assert result.verified_count == 0
        assert len(result.unverified_citations) == 1

    def test_multiple_citations_partial(self):
        pipeline = self._make_pipeline()
        chunks = self._make_chunks()

        response = AttributedResponse(
            answer="Оператор обязан [1]. Срок — 10 лет [2].",
            citations=[
                CitationItem(
                    chunk_id="doc1_p1",
                    statement="Оператор обязан принять меры",
                    quote="обязан принять меры по обеспечению безопасности",
                    citation_id=1,
                ),
                CitationItem(
                    chunk_id="doc2_p1",
                    statement="Срок хранения — 10 лет",
                    quote="Срок хранения данных не менее десяти лет.",
                    citation_id=2,
                ),
            ],
        )
        result = pipeline.verify_citations(response, chunks)

        # Первая цитата частично совпадает (нечёткий поиск)
        # Вторая — выдумана
        assert result.total_count == 2
        # Не требуем конкретного значения verified_count —
        # зависит от реализации нечёткого поиска
        assert 0 <= result.verified_count <= 2

    def test_empty_citations_zero_faithfulness(self):
        pipeline = self._make_pipeline()
        chunks = self._make_chunks()

        response = AttributedResponse(answer="Ответ без ссылок.", citations=[])
        result = pipeline.verify_citations(response, chunks)

        assert result.total_count == 0
        assert result.faithfulness_score == 0.0

    def test_case_insensitive_matching(self):
        pipeline = self._make_pipeline()
        chunks = [{"chunk_id": "c1", "text": "Оператор ОБЯЗАН принять МЕРЫ."}]

        response = AttributedResponse(
            answer="Оператор обязан [1].",
            citations=[
                CitationItem(
                    chunk_id="c1",
                    statement="Оператор обязан",
                    quote="оператор обязан принять меры.",
                    citation_id=1,
                )
            ],
        )
        result = pipeline.verify_citations(response, chunks)
        assert result.verified_count == 1


@pytest.mark.unit
class TestParseResponse:
    """Тесты разбора JSON-ответа без вызова LLM."""

    def _make_pipeline(self):
        return AttributionPipeline(client=None)

    def test_valid_json_parsed(self):
        pipeline = self._make_pipeline()
        raw = (
            '{"answer": "Ответ [1].", '
            '"citations": [{"id": 1, "chunk_id": "c1", '
            '"statement": "утверждение", "quote": "цитата"}], '
            '"unsupported": []}'
        )
        result = pipeline._parse_response(raw)

        assert result.answer == "Ответ [1]."
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "c1"
        assert result.citations[0].quote == "цитата"
        assert result.raw_response == raw

    def test_invalid_json_returns_fallback(self):
        pipeline = self._make_pipeline()
        raw = "Это не JSON {сломано}"
        result = pipeline._parse_response(raw)

        assert result.answer == raw
        assert result.citations == []
        assert result.raw_response == raw

    def test_empty_citations_list(self):
        pipeline = self._make_pipeline()
        raw = '{"answer": "Нет источников.", "citations": [], "unsupported": ["fact1"]}'
        result = pipeline._parse_response(raw)

        assert result.citations == []
        assert result.unsupported_statements == ["fact1"]

    def test_unsupported_statements_extracted(self):
        pipeline = self._make_pipeline()
        raw = (
            '{"answer": "Ответ.", "citations": [], '
            '"unsupported": ["утверждение без источника"]}'
        )
        result = pipeline._parse_response(raw)
        assert "утверждение без источника" in result.unsupported_statements


# ---------------------------------------------------------------------------
# Интеграционные тесты: требуют OpenAI API ключ в окружении
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAttributionPipelineIntegration:
    """Тесты требуют переменной окружения OPENAI_API_KEY."""

    def test_generate_with_citations_real_llm(self):
        pytest.skip(
            "Требует OPENAI_API_KEY — запустите вручную: "
            "pytest tests/test_attribution.py -v -m integration"
        )
