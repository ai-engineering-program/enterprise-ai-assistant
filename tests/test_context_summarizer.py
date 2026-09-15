"""Tests for app/context/context_summarizer.py (Lesson 4.1, module M4 opener).

ContextSummarizer is exercised with a fake summarize_fn (a plain Python
callable, not a real LLM call) so every test here is a unit test — no
external services are required.

Run:
    pytest tests/test_context_summarizer.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.context_summarizer import (
    ContextSummarizer,
    SummarizationResult,
    estimate_tokens,
)
from app.rag.context_builder import ContextChunk


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


@pytest.mark.unit
class TestContextSummarizerPassthrough:
    def test_no_summarize_call_when_under_budget(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError(
                "summarize_fn не должен вызываться, когда бюджет не превышен"
            )

        summarizer = ContextSummarizer(
            summarize_fn=_fail_if_called,
            token_counter=_char_len,
            chunk_token_threshold=50,
            max_context_tokens=100,
        )
        chunks = [
            ContextChunk(text="короткий фрагмент", score=0.9, source="confluence:policy"),
            ContextChunk(text="ещё один короткий", score=0.7, source="jira:SUPPORT-8842"),
        ]

        result = summarizer.compress("вопрос пользователя", chunks)

        assert isinstance(result, SummarizationResult)
        assert result.summarized_indices == []
        assert result.dropped_indices == []
        assert result.chunks == chunks
        assert result.final_token_count == result.original_token_count


@pytest.mark.unit
class TestContextSummarizerDispatch:
    def test_summarizes_only_chunks_over_threshold(self):
        calls: list[str] = []

        def _fake_summarize(prompt: str) -> str:
            calls.append(prompt)
            return "сжатая версия"

        big_text = "текст " * 40  # 240 символов — крупный фрагмент
        small_text = "короткий текст"  # 14 символов — короткий фрагмент

        summarizer = ContextSummarizer(
            summarize_fn=_fake_summarize,
            token_counter=_char_len,
            chunk_token_threshold=15,
            max_context_tokens=50,
        )
        chunks = [
            ContextChunk(text=big_text, score=0.9, source="confluence:policy"),
            ContextChunk(text=small_text, score=0.8, source="jira:SUPPORT-8842"),
        ]

        result = summarizer.compress(
            "почему списали абонентскую плату во время миграции", chunks
        )

        assert result.summarized_indices == [0]
        assert len(calls) == 1
        assert result.chunks[0].text == "сжатая версия"
        assert result.chunks[1].text == small_text  # короткий чанк не тронут
        assert result.dropped_indices == []

    def test_prompt_is_query_aware(self):
        captured: dict[str, str] = {}

        def _fake_summarize(prompt: str) -> str:
            captured["prompt"] = prompt
            return "сжатая версия"

        summarizer = ContextSummarizer(
            summarize_fn=_fake_summarize,
            token_counter=_char_len,
            chunk_token_threshold=5,
            max_context_tokens=5,
        )
        query = "почему заморозка не сработала во время миграции DNS"
        chunk = ContextChunk(
            text="длинный фрагмент регламента про приостановку платы",
            score=0.9,
            source="confluence:policy",
        )

        summarizer.compress(query, [chunk])

        assert query in captured["prompt"]
        assert chunk.text in captured["prompt"]


@pytest.mark.unit
class TestContextSummarizerTruncationFallback:
    def test_drops_lowest_score_chunk_when_still_over_budget_after_summarization(self):
        def _fake_summarize(prompt: str) -> str:
            return "x" * 60  # сжатая версия всё ещё крупная

        summarizer = ContextSummarizer(
            summarize_fn=_fake_summarize,
            token_counter=_char_len,
            chunk_token_threshold=10,
            max_context_tokens=70,
        )
        chunks = [
            ContextChunk(text="y" * 80, score=0.4, source="slack:thread-1"),  # низкий score
            ContextChunk(text="z" * 80, score=0.9, source="confluence:policy"),
        ]

        result = summarizer.compress("вопрос", chunks)

        assert 0 in result.dropped_indices
        assert 1 not in result.dropped_indices
        assert result.final_token_count <= 70
        assert len(result.chunks) == 1
        assert result.chunks[0].source == "confluence:policy"

    def test_summarize_chunk_preserves_score_and_source(self):
        def _fake_summarize(prompt: str) -> str:
            return "сжато"

        summarizer = ContextSummarizer(summarize_fn=_fake_summarize)
        chunk = ContextChunk(text="исходный длинный текст", score=0.42, source="jira:SUPPORT-8842")

        summarized = summarizer.summarize_chunk("вопрос", chunk)

        assert summarized.score == 0.42
        assert summarized.source == "jira:SUPPORT-8842"
        assert summarized.text == "сжато"
        assert summarized.metadata.get("summarized") is True


@pytest.mark.unit
class TestEstimateTokens:
    def test_estimate_tokens_empty_string(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_rough_heuristic(self):
        assert estimate_tokens("abcd" * 10) == 10  # 40 символов // 4


@pytest.mark.integration
class TestContextSummarizerIntegration:
    """Требуют реального вызова LLM API — не выполняются по умолчанию."""

    def test_with_real_llm_summarization(self):
        pytest.skip("Требует реального вызова LLM API — запускать вручную")
