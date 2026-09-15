"""Tests for app/context/extractive_compressor.py (Lesson 4.2, module M4).

ExtractiveCompressor's core is deterministic and rule-based (keyword
overlap + position heuristic), so unlike ContextSummarizer (4.1) it needs
no fake LLM call at all — every test here runs with the real default
scorer or a small stub, no external services required.

Run:
    pytest tests/test_extractive_compressor.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.extractive_compressor import (
    ExtractiveCompressor,
    ExtractionResult,
    keyword_overlap_score,
    split_sentences,
)
from app.rag.context_builder import ContextChunk


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


@pytest.mark.unit
class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        text = "Первое предложение. Второе! Третье?"
        assert split_sentences(text) == [
            "Первое предложение.",
            "Второе!",
            "Третье?",
        ]

    def test_single_sentence_without_terminal_punctuation(self):
        assert split_sentences("Просто текст без точки") == [
            "Просто текст без точки"
        ]


@pytest.mark.unit
class TestKeywordOverlapScore:
    def test_full_overlap(self):
        query = "почему списали абонентскую плату"
        sentence = "почему списали абонентскую плату во время миграции"
        assert keyword_overlap_score(query, sentence) == pytest.approx(1.0)

    def test_partial_overlap(self):
        query = "почему списали абонентскую плату"
        sentence = "почему это правило вообще существует"
        score = keyword_overlap_score(query, sentence)
        assert 0.0 < score < 1.0

    def test_empty_query_returns_zero(self):
        assert keyword_overlap_score("", "любое предложение") == 0.0

    def test_no_overlap_returns_zero(self):
        query = "абонентская плата"
        sentence = "совершенно другая тема без общих слов"
        assert keyword_overlap_score(query, sentence) == 0.0


@pytest.mark.unit
class TestExtractiveCompressorPassthrough:
    def test_no_compression_call_when_under_budget(self):
        def _fail_if_called(query: str, sentence: str) -> float:
            raise AssertionError(
                "sentence_scorer не должен вызываться, когда бюджет не превышен"
            )

        compressor = ExtractiveCompressor(
            sentence_scorer=_fail_if_called,
            token_counter=_char_len,
            chunk_token_threshold=50,
            max_context_tokens=100,
        )
        chunks = [
            ContextChunk(text="короткий фрагмент", score=0.9, source="confluence:policy"),
            ContextChunk(text="ещё один короткий", score=0.7, source="jira:SUPPORT-8842"),
        ]

        result = compressor.compress("вопрос пользователя", chunks)

        assert isinstance(result, ExtractionResult)
        assert result.extracted_indices == []
        assert result.dropped_indices == []
        assert result.chunks == chunks
        assert result.final_token_count == result.original_token_count


@pytest.mark.unit
class TestExtractiveCompressorDispatch:
    def test_compresses_only_chunks_over_threshold(self):
        big_text = (
            "Документ описывает общие принципы биллинга. "
            "Скидки предоставляются по объёму услуг. "
            "При проведении плановых работ по миграции оборудования "
            "начисление абонентской платы приостанавливается на срок работ. "
            "Порядок обжалования начислений описан в отдельном разделе."
        )
        small_text = "короткий текст"

        compressor = ExtractiveCompressor(
            token_counter=_char_len,
            chunk_token_threshold=60,
            max_context_tokens=150,
        )
        chunks = [
            ContextChunk(text=big_text, score=0.9, source="confluence:policy"),
            ContextChunk(text=small_text, score=0.8, source="jira:SUPPORT-8842"),
        ]

        result = compressor.compress(
            "почему списали абонентскую плату во время миграции", chunks
        )

        assert result.extracted_indices == [0]
        assert result.chunks[1].text == small_text  # короткий чанк не тронут
        assert result.dropped_indices == []
        # Верность оригиналу: результат не мог придумать новые слова —
        # каждое оставленное предложение должно дословно входить в исходный текст.
        for sentence in split_sentences(result.chunks[0].text):
            assert sentence in big_text

    def test_kept_sentences_are_verbatim_substrings_of_source(self):
        big_text = (
            "Первое предложение фрагмента про общие условия обслуживания. "
            "Второе предложение содержит критичную деталь про возврат абонентской платы. "
            "Третье предложение — заключительная формальность документа."
        )
        compressor = ExtractiveCompressor(
            token_counter=_char_len,
            chunk_token_threshold=10,
            max_context_tokens=10,
        )
        chunk = ContextChunk(text=big_text, score=0.9, source="confluence:policy")

        compressed = compressor.compress_chunk("возврат абонентской платы", chunk)

        assert compressed.text in big_text or all(
            s in big_text for s in split_sentences(compressed.text)
        )
        assert compressed.metadata.get("extracted") is True

    def test_single_sentence_chunk_is_returned_unchanged(self):
        compressor = ExtractiveCompressor(token_counter=_char_len)
        chunk = ContextChunk(
            text="Единственное предложение без точек с запятой",
            score=0.5,
            source="slack:thread-1",
        )

        result = compressor.compress_chunk("любой запрос", chunk)

        assert result.text == chunk.text


@pytest.mark.unit
class TestExtractiveCompressorScoring:
    def test_position_bonus_favors_first_sentence_on_tie(self):
        def _always_tied(query: str, sentence: str) -> float:
            return 0.5

        compressor = ExtractiveCompressor(
            sentence_scorer=_always_tied,
            position_bonus=0.2,
        )
        sentences = ["Первое.", "Второе.", "Третье."]

        scores = compressor.score_sentences("запрос", sentences)

        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(0.5)


@pytest.mark.unit
class TestExtractiveCompressorTruncationFallback:
    def test_drops_lowest_score_chunk_when_still_over_budget(self):
        low_score_text = "низкий приоритет. " * 10
        high_score_text = "высокий приоритет. " * 10

        compressor = ExtractiveCompressor(
            token_counter=_char_len,
            chunk_token_threshold=5,
            max_context_tokens=30,
        )
        chunks = [
            ContextChunk(text=low_score_text, score=0.2, source="slack:thread-1"),
            ContextChunk(text=high_score_text, score=0.9, source="confluence:policy"),
        ]

        result = compressor.compress("вопрос", chunks)

        assert 0 in result.dropped_indices
        assert 1 not in result.dropped_indices
        assert result.final_token_count <= 30
        assert len(result.chunks) == 1
        assert result.chunks[0].source == "confluence:policy"

    def test_compress_chunk_preserves_score_and_source(self):
        compressor = ExtractiveCompressor(token_counter=_char_len, chunk_token_threshold=5)
        chunk = ContextChunk(
            text="Первое предложение. Второе предложение. Третье предложение.",
            score=0.42,
            source="jira:SUPPORT-8842",
        )

        compressed = compressor.compress_chunk("вопрос", chunk)

        assert compressed.score == 0.42
        assert compressed.source == "jira:SUPPORT-8842"
        assert compressed.metadata.get("extracted") is True


@pytest.mark.integration
class TestExtractiveCompressorIntegration:
    """Требуют реальной модели эмбеддингов для sentence_scorer — не выполняются по умолчанию."""

    def test_with_real_embedding_scorer(self):
        pytest.skip("Требует реальной модели эмбеддингов — запускать вручную")
