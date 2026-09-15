"""Tests for app/context/abstraction_ladder.py (Lesson 4.3, module M4).

select_level() is pure — score and token_budget in, LadderLevel out — so
it needs no fake LLM call at all. render_level()/pick() delegate the
actual text-shortening step to ContextSummarizer.summarize_chunk (4.1),
which is exercised here with a fake summarize_fn (a plain Python
callable), exactly as in test_context_summarizer.py. Every test in this
file is therefore a unit test — no external services are required.

Run:
    pytest tests/test_abstraction_ladder.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.abstraction_ladder import (
    AbstractionLadder,
    LadderLevel,
    LadderResult,
    LevelSpec,
    DEFAULT_LEVELS,
)
from app.context.context_summarizer import ContextSummarizer
from app.rag.context_builder import ContextChunk


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


def _counting_summarize_fn(calls: list):
    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "сжатая версия уровня"

    return _fn


@pytest.mark.unit
class TestSelectLevelPureLogic:
    """select_level() must never touch summarize_fn — pure score/budget math."""

    def test_high_score_and_full_budget_selects_full(self):
        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=lambda p: "x"))

        # full text is small (500 tokens) and comfortably fits the budget
        level = ladder.select_level(score=0.9, token_budget=10_000, full_token_count=500)

        assert level == LadderLevel.FULL

    def test_high_score_but_large_full_text_downgrades_below_full(self):
        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=lambda p: "x"))

        # score alone would earn FULL, but the full text (5000 tokens)
        # does not fit the budget (200) — high relevance does not waive
        # the size constraint
        level = ladder.select_level(score=0.95, token_budget=200, full_token_count=5_000)

        assert level == LadderLevel.PARAGRAPH

    def test_low_score_never_gets_full_even_with_huge_budget(self):
        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=lambda p: "x"))

        # marginally relevant chunk from the incident: plenty of budget
        # and a small full text, but low score must still cap it at the
        # coarsest level — budget alone does not earn detail
        level = ladder.select_level(score=0.1, token_budget=10_000, full_token_count=50)

        assert level == LadderLevel.GIST

    def test_mid_score_and_mid_budget_selects_sentence(self):
        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=lambda p: "x"))

        level = ladder.select_level(score=0.3, token_budget=45, full_token_count=300)

        assert level == LadderLevel.SENTENCE

    def test_extremely_tight_budget_falls_back_to_coarsest_level(self):
        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=lambda p: "x"))

        # not even GIST's 12-token ceiling fits a 1-token budget — fall
        # back to the coarsest defined level rather than raising
        level = ladder.select_level(score=0.9, token_budget=1, full_token_count=5_000)

        assert level == LadderLevel.GIST

    def test_select_level_never_calls_summarize_fn(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("select_level не должен вызывать summarize_fn")

        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=_fail_if_called))

        ladder.select_level(score=0.5, token_budget=500, full_token_count=800)
        ladder.select_level(score=0.0, token_budget=0, full_token_count=0)
        ladder.select_level(score=1.0, token_budget=1_000_000, full_token_count=1_000_000)


@pytest.mark.unit
class TestRenderLevelCascade:
    def test_full_level_returns_chunk_unchanged_without_any_call(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("FULL не должен вызывать summarize_fn")

        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=_fail_if_called))
        chunk = ContextChunk(text="оригинальный текст", score=0.9, source="confluence:policy")

        rendered = ladder.render_level("вопрос", chunk, LadderLevel.FULL)

        assert rendered is chunk

    def test_paragraph_level_calls_summarize_once(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        ladder = AbstractionLadder(summarizer=summarizer)
        chunk = ContextChunk(text="оригинальный текст", score=0.5, source="confluence:policy")

        ladder.render_level("вопрос", chunk, LadderLevel.PARAGRAPH)

        assert len(calls) == 1

    def test_gist_level_cascades_through_intermediate_levels(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        ladder = AbstractionLadder(summarizer=summarizer)
        chunk = ContextChunk(text="оригинальный текст", score=0.1, source="slack:thread-1")

        # DEFAULT_LEVELS = FULL, PARAGRAPH, SENTENCE, GIST -> 3 calls to reach GIST
        ladder.render_level("вопрос", chunk, LadderLevel.GIST)

        assert len(calls) == 3

    def test_precomputed_cache_hit_avoids_any_call(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("precomputed уровень не должен пересчитываться")

        ladder = AbstractionLadder(summarizer=ContextSummarizer(summarize_fn=_fail_if_called))
        chunk = ContextChunk(text="оригинальный текст", score=0.1, source="slack:thread-1")
        gist_chunk = ContextChunk(text="самая суть", score=0.1, source="slack:thread-1")

        rendered = ladder.render_level(
            "вопрос", chunk, LadderLevel.GIST, precomputed={LadderLevel.GIST: gist_chunk}
        )

        assert rendered is gist_chunk


@pytest.mark.unit
class TestGenerateFullLadder:
    def test_produces_one_entry_per_level(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        ladder = AbstractionLadder(summarizer=summarizer)
        chunk = ContextChunk(text="оригинальный текст", score=0.8, source="confluence:policy")

        full_ladder = ladder.generate_full_ladder("вопрос", chunk)

        assert set(full_ladder.keys()) == {
            LadderLevel.FULL,
            LadderLevel.PARAGRAPH,
            LadderLevel.SENTENCE,
            LadderLevel.GIST,
        }
        assert full_ladder[LadderLevel.FULL] is chunk
        assert len(calls) == 3  # PARAGRAPH, SENTENCE, GIST — one call each


@pytest.mark.unit
class TestPickIntegration:
    def test_pick_returns_level_consistent_with_select_level(self):
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn([]))
        ladder = AbstractionLadder(summarizer=summarizer)
        chunk = ContextChunk(text="оригинальный текст", score=0.95, source="confluence:policy")

        result = ladder.pick("вопрос", chunk, token_budget=10_000)

        assert isinstance(result, LadderResult)
        assert result.level == LadderLevel.FULL
        assert result.chunk is chunk
        assert result.score == 0.95
        assert result.token_budget == 10_000

    def test_pick_uses_precomputed_ladder_without_new_calls(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("pick() должен переиспользовать precomputed, а не считать заново")

        summarizer = ContextSummarizer(summarize_fn=_fail_if_called)
        ladder = AbstractionLadder(summarizer=summarizer)
        chunk = ContextChunk(text="оригинальный текст", score=0.1, source="slack:thread-1")
        precomputed = {LadderLevel.GIST: ContextChunk(text="суть", score=0.1, source="slack:thread-1")}

        result = ladder.pick("вопрос", chunk, token_budget=10_000, precomputed=precomputed)

        assert result.level == LadderLevel.GIST
        assert result.chunk.text == "суть"


@pytest.mark.unit
class TestDefaultLevelsOrdering:
    def test_default_levels_ordered_from_detailed_to_coarse(self):
        assert [spec.level for spec in DEFAULT_LEVELS] == [
            LadderLevel.FULL,
            LadderLevel.PARAGRAPH,
            LadderLevel.SENTENCE,
            LadderLevel.GIST,
        ]

    def test_default_levels_min_score_decreases_with_coarseness(self):
        scores = [spec.min_score for spec in DEFAULT_LEVELS]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
class TestAbstractionLadderIntegration:
    """Требуют реального вызова LLM API — не выполняются по умолчанию."""

    def test_full_ladder_with_real_llm(self):
        pytest.skip("Требует реального вызова LLM API — запускать вручную")
