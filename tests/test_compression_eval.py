"""Tests for app/context/compression_eval.py (Lesson 4.5, module M4, closing).

Every test in this file supplies a plain-Python fake CompressionStrategyFn —
never one of the real ContextSummarizer/ExtractiveCompressor/AbstractionLadder/
HierarchicalCompressor instances — so the whole harness is exercised without a
single LLM call, exactly as required by the lesson's "deterministic and
unit-testable by injecting fake compression functions" design.

Run:
    pytest tests/test_compression_eval.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.compression_eval import (
    CaseEvalResult,
    CompressionEvalHarness,
    CompressionEvalReport,
    CompressionTestCase,
    default_fact_match,
    select_best_strategy,
)
from app.rag.context_builder import ContextChunk


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


def _make_incident_case() -> CompressionTestCase:
    """The "Печора Телеком" scenario from lessons 4.4/4.5: three near-duplicate
    chunks about the standard penalty, one unique chunk about the special-
    client clause, which is exactly the fact worth protecting."""
    chunks = [
        ContextChunk(
            text="Стандартная неустойка при расторжении — 3% от остатка контракта.",
            score=0.82,
            source="confluence:tariff-policy",
        ),
        ContextChunk(
            text="Как мы обычно отвечаем: неустойка 3%, это стандартная ставка.",
            score=0.79,
            source="jira:TICKET-101",
        ),
        ContextChunk(
            text="в саппорте всегда говорим что 3 процента при досрочном расторжении",
            score=0.75,
            source="slack:support-thread",
        ),
        ContextChunk(
            text="Для программы спецклиент неустойка снижается до 1% по договору.",
            score=0.35,
            source="jira:TICKET-202",
        ),
    ]
    return CompressionTestCase(
        case_id="pechora-1",
        query="какая неустойка положена спецклиенту при расторжении?",
        ground_truth_fact="спецклиент неустойка снижается до 1%",
        chunks=chunks,
    )


def _identity_strategy(query: str, chunks: list[ContextChunk]) -> list[ContextChunk]:
    """Fake strategy: no compression at all — everything survives."""
    return list(chunks)


def _drop_lowest_score_strategy(query: str, chunks: list[ContextChunk]) -> list[ContextChunk]:
    """Fake strategy simulating the incident's truncation-by-score: drops the
    single chunk with the lowest score — the one carrying the rare fact."""
    if not chunks:
        return []
    survivors = sorted(chunks, key=lambda c: c.score, reverse=True)[:-1]
    return survivors


def _keep_only_unique_strategy(query: str, chunks: list[ContextChunk]) -> list[ContextChunk]:
    """Fake strategy simulating a safer HierarchicalCompressor-style result:
    duplicates collapse, but the unique rare-fact chunk survives untouched."""
    return [c for c in chunks if "спецклиент" in c.text or "1%" in c.text]


@pytest.mark.unit
class TestDefaultFactMatch:
    def test_empty_fact_always_matches(self):
        assert default_fact_match("", "любой текст") is True
        assert default_fact_match("   ", "любой текст") is True

    def test_exact_phrase_present_matches(self):
        fact = "спецклиент неустойка снижается до 1%"
        text = "Для программы спецклиент неустойка снижается до 1% по договору."
        assert default_fact_match(fact, text) is True

    def test_completely_absent_fact_does_not_match(self):
        fact = "спецклиент неустойка снижается до 1%"
        text = "Стандартная неустойка при расторжении — 3% от остатка контракта."
        assert default_fact_match(fact, text) is False

    def test_partial_overlap_below_threshold_does_not_match(self):
        fact = "спецклиент неустойка снижается до 1%"
        # Shares only "неустойка" out of five fact words -> well below 0.6
        text = "неустойка обсуждается в отдельном документе"
        assert default_fact_match(fact, text, threshold=0.6) is False

    def test_threshold_is_configurable(self):
        fact = "спецклиент неустойка снижается до 1%"
        text = "неустойка обсуждается в отдельном документе"
        # Same low overlap, but a lenient threshold should now accept it
        assert default_fact_match(fact, text, threshold=0.1) is True


@pytest.mark.unit
class TestRunCase:
    def test_identity_strategy_survives_fact(self):
        harness = CompressionEvalHarness(strategy_fn=_identity_strategy, token_counter=_char_len)
        case = _make_incident_case()

        result = harness.run_case(case)

        assert isinstance(result, CaseEvalResult)
        assert result.fact_survived is True
        assert result.case_id == "pechora-1"
        assert "jira:TICKET-202" in result.matched_chunk_sources

    def test_drop_lowest_score_strategy_loses_fact(self):
        harness = CompressionEvalHarness(strategy_fn=_drop_lowest_score_strategy, token_counter=_char_len)
        case = _make_incident_case()

        result = harness.run_case(case)

        assert result.fact_survived is False
        assert result.matched_chunk_sources == []

    def test_compression_ratio_reflects_token_reduction(self):
        harness = CompressionEvalHarness(strategy_fn=_keep_only_unique_strategy, token_counter=_char_len)
        case = _make_incident_case()

        result = harness.run_case(case)

        original = sum(_char_len(c.text) for c in case.chunks)
        expected_ratio = _char_len(case.chunks[3].text) / original
        assert result.compression_ratio == pytest.approx(expected_ratio)
        assert result.original_token_count == original

    def test_zero_original_tokens_gives_ratio_one(self):
        harness = CompressionEvalHarness(strategy_fn=_identity_strategy, token_counter=_char_len)
        empty_case = CompressionTestCase(
            case_id="empty", query="q", ground_truth_fact="", chunks=[]
        )

        result = harness.run_case(empty_case)

        assert result.compression_ratio == pytest.approx(1.0)


@pytest.mark.unit
class TestRun:
    def test_empty_cases_raises(self):
        harness = CompressionEvalHarness(strategy_fn=_identity_strategy, token_counter=_char_len)

        with pytest.raises(ValueError):
            harness.run([])

    def test_recall_is_fraction_of_surviving_cases(self):
        harness = CompressionEvalHarness(strategy_fn=_drop_lowest_score_strategy, token_counter=_char_len)
        surviving_case = CompressionTestCase(
            case_id="survives",
            query="q",
            ground_truth_fact="стандартная неустойка",
            chunks=[
                ContextChunk(text="Стандартная неустойка 3%.", score=0.9, source="a"),
                ContextChunk(text="Что-то совсем другое и неважное.", score=0.1, source="b"),
            ],
        )
        losing_case = _make_incident_case()

        report = harness.run([surviving_case, losing_case], strategy_name="drop-lowest")

        assert isinstance(report, CompressionEvalReport)
        assert report.strategy_name == "drop-lowest"
        assert report.recall == pytest.approx(0.5)
        assert len(report.results) == 2

    def test_full_recall_when_every_case_survives(self):
        harness = CompressionEvalHarness(strategy_fn=_identity_strategy, token_counter=_char_len)
        cases = [_make_incident_case(), _make_incident_case()]

        report = harness.run(cases, strategy_name="identity")

        assert report.recall == pytest.approx(1.0)


@pytest.mark.unit
class TestSelectBestStrategy:
    def _report(self, name: str, recall: float, ratio: float) -> CompressionEvalReport:
        return CompressionEvalReport(
            strategy_name=name, results=[], recall=recall, mean_compression_ratio=ratio
        )

    def test_picks_most_aggressive_among_safe_strategies(self):
        reports = [
            self._report("summarizer@2000", recall=1.0, ratio=0.8),
            self._report("hierarchical@1000", recall=1.0, ratio=0.4),
            self._report("hierarchical@600", recall=0.5, ratio=0.15),
        ]

        best = select_best_strategy(reports, min_recall=1.0)

        assert best is not None
        assert best.strategy_name == "hierarchical@1000"

    def test_returns_none_when_nothing_meets_recall_floor(self):
        reports = [
            self._report("hierarchical@600", recall=0.5, ratio=0.15),
            self._report("hierarchical@800", recall=0.75, ratio=0.2),
        ]

        best = select_best_strategy(reports, min_recall=1.0)

        assert best is None

    def test_tie_break_keeps_first_in_input_order(self):
        reports = [
            self._report("strategy-a", recall=1.0, ratio=0.5),
            self._report("strategy-b", recall=1.0, ratio=0.5),
        ]

        best = select_best_strategy(reports, min_recall=1.0)

        assert best is not None
        assert best.strategy_name == "strategy-a"


@pytest.mark.integration
class TestCompressionEvalIntegration:
    """Требуют реального вызова LLM API (реальной стратегии сжатия) — не
    выполняются по умолчанию."""

    def test_full_pipeline_with_real_llm_strategy(self):
        pytest.skip("Требует реального вызова LLM API — запускать вручную")
