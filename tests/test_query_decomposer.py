"""Tests for app/context/query_decomposer.py (Lesson 3.1, opening M3).

QueryDecomposer is a pure rule-based component — no external services are
involved, so every test here is a unit test. The single integration-marked
test is a placeholder for the LLM-based decompose() implementation
mentioned in the class docstring (out of scope for this lesson).

Run:
    pytest tests/test_query_decomposer.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.query_decomposer import (
    DecompositionPattern,
    DecompositionResult,
    QueryDecomposer,
    SubQuery,
)


@pytest.mark.unit
class TestClassifyPattern:
    """classify_pattern() must apply the documented priority order:
    comparison > conjunctive > implicit_subquestion > atomic.
    """

    def test_atomic_query_has_no_markers(self):
        decomposer = QueryDecomposer()
        query = "Какой тариф у клиента ООО «Северный Мост»?"

        assert decomposer.classify_pattern(query) is DecompositionPattern.ATOMIC

    def test_conjunctive_wins_over_implicit_marker_in_same_query(self):
        # Contains BOTH a conjunctive marker (" и почему") AND an implicit
        # marker ("больше обычного") — conjunctive must win per priority.
        decomposer = QueryDecomposer()
        query = "Какой тариф у клиента и почему списание больше обычного?"

        assert decomposer.classify_pattern(query) is DecompositionPattern.CONJUNCTIVE

    def test_comparison_wins_over_conjunctive_marker_in_same_query(self):
        # Contains BOTH a comparison marker ("что лучше") AND a conjunctive
        # marker (" и почему") — comparison must win per priority.
        decomposer = QueryDecomposer()
        query = "Что лучше: тариф Гигабит или тариф Оптима и почему так дороже?"

        assert decomposer.classify_pattern(query) is DecompositionPattern.COMPARISON

    def test_implicit_subquestion_detected_without_other_markers(self):
        decomposer = QueryDecomposer()
        query = "Почему у клиента списание больше обычного в этом месяце?"

        assert (
            decomposer.classify_pattern(query)
            is DecompositionPattern.IMPLICIT_SUBQUESTION
        )


@pytest.mark.unit
class TestDecompose:
    def test_atomic_query_decomposes_into_single_subquery(self):
        decomposer = QueryDecomposer()
        query = "Какой тариф у клиента ООО «Северный Мост»?"

        result = decomposer.decompose(query)

        assert isinstance(result, DecompositionResult)
        assert result.original_query == query
        assert result.pattern is DecompositionPattern.ATOMIC
        assert result.sub_queries == [
            SubQuery(text=query, order=0, pattern=DecompositionPattern.ATOMIC)
        ]

    def test_conjunctive_query_splits_into_two_ordered_subqueries(self):
        decomposer = QueryDecomposer()
        query = "Какой тариф у клиента и почему списание больше обычного?"

        result = decomposer.decompose(query)

        assert result.pattern is DecompositionPattern.CONJUNCTIVE
        texts = [sq.text for sq in result.sub_queries]
        assert texts == [
            "Какой тариф у клиента",
            "почему списание больше обычного?",
        ]
        assert [sq.order for sq in result.sub_queries] == [0, 1]
        assert all(
            sq.pattern is DecompositionPattern.CONJUNCTIVE
            for sq in result.sub_queries
        )

    def test_comparison_query_extracts_two_entities_around_ili(self):
        decomposer = QueryDecomposer()
        query = "Что лучше: тариф Гигабит или тариф Оптима?"

        result = decomposer.decompose(query)

        assert result.pattern is DecompositionPattern.COMPARISON
        texts = [sq.text for sq in result.sub_queries]
        assert texts == [
            "Что лучше: тариф Гигабит",
            "тариф Оптима",
        ]

    def test_comparison_without_ili_marker_falls_back_to_single_query(self):
        # "чем отличается X от Y" phrasing has no literal " или " — this
        # lesson's rule-based splitter does not attempt to parse it, and
        # must fall back to returning the query unsplit rather than
        # guessing or raising.
        decomposer = QueryDecomposer()
        query = "Чем отличается тариф Гигабит от тарифа Оптима?"

        result = decomposer.decompose(query)

        assert result.pattern is DecompositionPattern.COMPARISON
        assert [sq.text for sq in result.sub_queries] == [query]

    def test_implicit_subquestion_prepends_baseline_subquery(self):
        decomposer = QueryDecomposer()
        query = "Почему у клиента списание больше обычного в этом месяце?"

        result = decomposer.decompose(query)

        assert result.pattern is DecompositionPattern.IMPLICIT_SUBQUESTION
        texts = [sq.text for sq in result.sub_queries]
        assert texts == [
            "текущее состояние: Почему у клиента списание",
            query,
        ]
        # Baseline subquery must come first — the explicit part cannot be
        # evaluated as "unusual" without it.
        assert result.sub_queries[0].order == 0
        assert result.sub_queries[1].order == 1

    def test_custom_markers_override_defaults(self):
        # A decomposer configured with custom markers must not fall back
        # to the built-in defaults — this is the same extensibility
        # contract as TruthAxisRouter (lesson 2.4).
        decomposer = QueryDecomposer(
            conjunctive_markers=(" и в дополнение",),
            comparison_markers=(),
            implicit_markers=(),
        )
        query = "Покажи баланс и в дополнение историю платежей"

        assert decomposer.classify_pattern(query) is DecompositionPattern.CONJUNCTIVE
        # Default marker " и почему" must NOT match anymore.
        default_style_query = "Покажи баланс и почему он изменился"
        assert (
            decomposer.classify_pattern(default_style_query)
            is DecompositionPattern.ATOMIC
        )


@pytest.mark.integration
class TestQueryDecomposerLLMBacked:
    """Placeholder for an LLM-backed QueryDecomposer implementation.

    Out of scope for this CORE-level lesson (see class docstring in
    app/context/query_decomposer.py) — kept here so the test suite already
    has a slot for it when an LLM-based decompose() is introduced later
    in the course.
    """

    def test_llm_backed_decomposition_not_covered_by_this_lesson(self):
        pytest.skip(
            "LLM-based decomposition is out of scope for lesson 3.1 — "
            "requires an LLM API key and is not exercised here."
        )
