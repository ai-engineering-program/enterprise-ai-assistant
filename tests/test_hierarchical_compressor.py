"""Tests for app/context/hierarchical_compressor.py (Lesson 4.4, module M4).

group() and default_group_key() are pure — they never touch summarize_fn,
so they are tested with no fake LLM at all. map_group()/reduce_groups()
delegate the actual text-shortening step to ContextSummarizer.summarize_chunk
(4.1), exercised here with a fake summarize_fn (a plain Python callable),
exactly as in test_context_summarizer.py and test_abstraction_ladder.py.
Every test in this file is therefore a unit test — no external services
are required.

Run:
    pytest tests/test_hierarchical_compressor.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.hierarchical_compressor import (
    ChunkGroup,
    HierarchicalCompressionResult,
    HierarchicalCompressor,
    default_group_key,
)
from app.context.context_summarizer import ContextSummarizer
from app.rag.context_builder import ContextChunk


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


def _counting_summarize_fn(calls: list):
    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "сведённая версия группы"

    return _fn


def _make_incident_chunks() -> list[ContextChunk]:
    """The three-near-duplicates-plus-one-unique-fact scenario from the
    lesson's incident: Confluence/Jira#1/Slack all restate the standard
    penalty (topic_key="неустойка"), Jira#2 is the only source of the
    special-client clause (topic_key="спецклиент")."""
    return [
        ContextChunk(
            text="Стандартная неустойка при расторжении — 3% от остатка контракта.",
            score=0.82,
            source="confluence:tariff-policy",
            metadata={"topic_key": "неустойка"},
        ),
        ContextChunk(
            text="Как мы обычно отвечаем: неустойка 3%, это стандартная ставка.",
            score=0.79,
            source="jira:TICKET-101",
            metadata={"topic_key": "неустойка"},
        ),
        ContextChunk(
            text="в саппорте всегда говорим что 3 процента при досрочном расторжении",
            score=0.75,
            source="slack:support-thread",
            metadata={"topic_key": "неустойка"},
        ),
        ContextChunk(
            text="Для программы «спецклиент» неустойка снижается до 1% по договору.",
            score=0.35,
            source="jira:TICKET-202",
            metadata={"topic_key": "спецклиент"},
        ),
    ]


@pytest.mark.unit
class TestDefaultGroupKey:
    def test_uses_topic_key_when_present(self):
        chunk = ContextChunk(
            text="x", score=0.5, source="confluence:policy", metadata={"topic_key": "неустойка"}
        )
        assert default_group_key(chunk) == "неустойка"

    def test_falls_back_to_source_when_topic_key_missing(self):
        chunk = ContextChunk(text="x", score=0.5, source="slack:thread-1", metadata={})

        assert default_group_key(chunk) == "slack:thread-1"

    def test_falls_back_to_source_when_topic_key_empty_string(self):
        chunk = ContextChunk(
            text="x", score=0.5, source="jira:TICKET-1", metadata={"topic_key": ""}
        )

        assert default_group_key(chunk) == "jira:TICKET-1"


@pytest.mark.unit
class TestGroupPureLogic:
    def test_cross_source_duplicates_land_in_one_group(self):
        summarizer = ContextSummarizer(summarize_fn=lambda p: "x")
        compressor = HierarchicalCompressor(summarizer=summarizer)
        chunks = _make_incident_chunks()

        groups = compressor.group(chunks)

        assert len(groups) == 2
        keys = {g.key for g in groups}
        assert keys == {"неустойка", "спецклиент"}

    def test_group_order_follows_first_appearance(self):
        summarizer = ContextSummarizer(summarize_fn=lambda p: "x")
        compressor = HierarchicalCompressor(summarizer=summarizer)
        chunks = _make_incident_chunks()

        groups = compressor.group(chunks)

        # "неустойка" appears first in the input list, "спецклиент" last
        assert [g.key for g in groups] == ["неустойка", "спецклиент"]

    def test_duplicate_group_contains_all_three_chunks_in_order(self):
        summarizer = ContextSummarizer(summarize_fn=lambda p: "x")
        compressor = HierarchicalCompressor(summarizer=summarizer)
        chunks = _make_incident_chunks()

        groups = compressor.group(chunks)
        duplicate_group = next(g for g in groups if g.key == "неустойка")

        assert [c.source for c in duplicate_group.chunks] == [
            "confluence:tariff-policy",
            "jira:TICKET-101",
            "slack:support-thread",
        ]

    def test_group_never_calls_summarize_fn(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("group() не должен вызывать summarize_fn")

        summarizer = ContextSummarizer(summarize_fn=_fail_if_called)
        compressor = HierarchicalCompressor(summarizer=summarizer)

        compressor.group(_make_incident_chunks())

    def test_empty_input_returns_empty_groups(self):
        summarizer = ContextSummarizer(summarize_fn=lambda p: "x")
        compressor = HierarchicalCompressor(summarizer=summarizer)

        assert compressor.group([]) == []


@pytest.mark.unit
class TestMapGroup:
    def test_single_chunk_group_returned_unchanged_without_call(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("одиночная группа не должна вызывать summarize_fn")

        summarizer = ContextSummarizer(summarize_fn=_fail_if_called)
        compressor = HierarchicalCompressor(summarizer=summarizer)
        unique_chunk = _make_incident_chunks()[3]
        group = ChunkGroup(key="спецклиент", chunks=[unique_chunk])

        result = compressor.map_group("вопрос", group)

        assert result is unique_chunk

    def test_multi_chunk_group_calls_summarize_exactly_once(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(summarizer=summarizer)
        duplicate_chunks = _make_incident_chunks()[:3]
        group = ChunkGroup(key="неустойка", chunks=duplicate_chunks)

        result = compressor.map_group("вопрос", group)

        assert len(calls) == 1
        assert result.source == "неустойка"

    def test_merged_chunk_score_is_max_of_group(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(summarizer=summarizer)
        duplicate_chunks = _make_incident_chunks()[:3]  # scores 0.82, 0.79, 0.75
        group = ChunkGroup(key="неустойка", chunks=duplicate_chunks)

        result = compressor.map_group("вопрос", group)

        assert result.score == pytest.approx(0.82)

    def test_prompt_contains_every_source_in_group(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(summarizer=summarizer)
        duplicate_chunks = _make_incident_chunks()[:3]
        group = ChunkGroup(key="неустойка", chunks=duplicate_chunks)

        compressor.map_group("вопрос", group)

        assert len(calls) == 1
        for chunk in duplicate_chunks:
            assert chunk.source in calls[0]


@pytest.mark.unit
class TestReduceGroups:
    def test_within_budget_returns_unchanged_without_new_calls(self):
        def _fail_if_called(prompt: str) -> str:
            raise AssertionError("reduce_groups не должен вызывать summarize_fn, если бюджет уже выполнен")

        summarizer = ContextSummarizer(summarize_fn=_fail_if_called)
        compressor = HierarchicalCompressor(
            summarizer=summarizer, token_counter=_char_len, max_context_tokens=1000
        )
        summaries = [
            ContextChunk(text="короткий summary A", score=0.8, source="группа A"),
            ContextChunk(text="короткий summary B", score=0.3, source="группа B"),
        ]

        result = compressor.reduce_groups("вопрос", summaries)

        assert result == summaries

    def test_over_budget_merges_lowest_score_pair(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(
            summarizer=summarizer, token_counter=_char_len, max_context_tokens=10
        )
        summaries = [
            ContextChunk(text="a" * 20, score=0.9, source="группа A"),
            ContextChunk(text="b" * 20, score=0.2, source="группа B"),
        ]

        result = compressor.reduce_groups("вопрос", summaries)

        assert len(calls) == 1
        assert len(result) == 1

    def test_does_not_merge_forever_when_only_one_group_left(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(
            summarizer=summarizer, token_counter=_char_len, max_context_tokens=1
        )
        summaries = [ContextChunk(text="x" * 50, score=0.5, source="группа A")]

        result = compressor.reduce_groups("вопрос", summaries)

        assert len(calls) == 0
        assert result == summaries


@pytest.mark.unit
class TestCompressIntegration:
    def test_incident_scenario_keeps_unique_fact_as_separate_chunk(self):
        calls: list = []
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn(calls))
        compressor = HierarchicalCompressor(
            summarizer=summarizer, token_counter=_char_len, max_context_tokens=10_000
        )
        chunks = _make_incident_chunks()

        result = compressor.compress("неустойка для спецклиента", chunks)

        assert isinstance(result, HierarchicalCompressionResult)
        # two groups survive: the merged duplicate-policy summary, and the
        # untouched unique special-client chunk — the incident's lost fact
        # is no longer competing 1-vs-3 for truncation.
        assert len(result.chunks) == 2
        sources = {c.source for c in result.chunks}
        assert "jira:TICKET-202" in sources  # unique chunk survived unchanged

    def test_result_records_original_groups_for_diagnostics(self):
        summarizer = ContextSummarizer(summarize_fn=_counting_summarize_fn([]))
        compressor = HierarchicalCompressor(
            summarizer=summarizer, token_counter=_char_len, max_context_tokens=10_000
        )
        chunks = _make_incident_chunks()

        result = compressor.compress("вопрос", chunks)

        assert len(result.groups) == 2
        assert all(isinstance(g, ChunkGroup) for g in result.groups)


@pytest.mark.integration
class TestHierarchicalCompressorIntegration:
    """Требуют реального вызова LLM API — не выполняются по умолчанию."""

    def test_full_pipeline_with_real_llm(self):
        pytest.skip("Требует реального вызова LLM API — запускать вручную")
