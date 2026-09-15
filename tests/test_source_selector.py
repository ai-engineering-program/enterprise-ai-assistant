"""Tests for app/context/source_selector.py (Lesson 3.3, module M3).

SourceSelector is pure rule-based logic (regex + keyword lists + an
optionally injected TruthAxisRouter) — no vector store or API connector is
involved, so every test here is a unit test. The single integration-marked
test is a placeholder for wiring SourceSelector into real retrieval
connectors in a later course.

Run:
    pytest tests/test_source_selector.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.source_registry import SourceType
from app.context.source_selector import (
    DEFAULT_FALLBACK_SOURCES,
    SourceSelectionResult,
    SourceSelector,
)
from app.context.truth_axis_router import TruthAxis, TruthAxisRouter


# Ticket-number query from the lesson incident: Regina asking about
# SUPPORT-8842 for client "Северный Мороз".
TICKET_QUERY = "Какой сейчас статус тикета SUPPORT-8842 по миграции DNS?"
CONFLUENCE_KEYWORD_QUERY = "Что говорит регламент миграции DNS для новых клиентов?"
SLACK_KEYWORD_QUERY = "Мы это уже обсуждали в чате поддержки на прошлой неделе?"
NEUTRAL_QUERY = "Расскажи про миграцию DNS в компании."
CONFLICTING_QUERY = (
    "Какой сейчас статус SUPPORT-8842 согласно регламенту эскалации?"
)


@pytest.mark.unit
class TestFindEntityHint:
    def test_ticket_number_maps_to_jira(self):
        selector = SourceSelector()
        assert selector.find_entity_hint(TICKET_QUERY) is SourceType.JIRA

    def test_confluence_keyword_maps_to_confluence(self):
        selector = SourceSelector()
        assert selector.find_entity_hint(CONFLUENCE_KEYWORD_QUERY) is SourceType.CONFLUENCE

    def test_slack_keyword_maps_to_slack(self):
        selector = SourceSelector()
        assert selector.find_entity_hint(SLACK_KEYWORD_QUERY) is SourceType.SLACK

    def test_no_marker_returns_none(self):
        selector = SourceSelector()
        assert selector.find_entity_hint(NEUTRAL_QUERY) is None

    def test_ticket_number_wins_over_confluence_keyword(self):
        # Both a ticket number AND regulation wording are present — the
        # more specific signal (ticket number) must be checked first and
        # win, per the lesson's priority-order rule.
        selector = SourceSelector()
        assert selector.find_entity_hint(CONFLICTING_QUERY) is SourceType.JIRA

    def test_case_insensitive_keyword_match(self):
        selector = SourceSelector()
        assert selector.find_entity_hint("Есть ТРЕД про это в Slack?") is SourceType.SLACK


@pytest.mark.unit
class TestClassifyIntent:
    def test_without_router_returns_ambiguous(self):
        selector = SourceSelector()
        assert selector.classify_intent(NEUTRAL_QUERY) is TruthAxis.AMBIGUOUS

    def test_with_router_delegates_to_it(self):
        router = TruthAxisRouter()
        selector = SourceSelector(truth_axis_router=router)
        query = "Какой сейчас статус тикета миграции DNS?"
        assert selector.classify_intent(query) == router.classify_axis(query)


@pytest.mark.unit
class TestSelect:
    def test_entity_hint_produces_single_source(self):
        selector = SourceSelector()
        result = selector.select(TICKET_QUERY)

        assert isinstance(result, SourceSelectionResult)
        assert result.selected_sources == [SourceType.JIRA]
        assert result.entity_hint is SourceType.JIRA
        assert result.used_fallback is False

    def test_entity_hint_wins_even_with_mismatched_axis_signal(self):
        # CONFLICTING_QUERY carries both a ticket number and regulation
        # wording; TruthAxisRouter would call this ambiguous, but the
        # entity hint must still decide alone.
        router = TruthAxisRouter()
        selector = SourceSelector(truth_axis_router=router)

        result = selector.select(CONFLICTING_QUERY)

        assert result.selected_sources == [SourceType.JIRA]
        assert result.entity_hint is SourceType.JIRA
        assert result.used_fallback is False

    def test_design_intent_axis_selects_confluence_only(self):
        router = TruthAxisRouter()
        selector = SourceSelector(truth_axis_router=router)
        query = "Как должно быть организовано разрешение DNS по дизайну?"

        result = selector.select(query)

        assert result.axis is TruthAxis.DESIGN_INTENT
        assert result.selected_sources == [SourceType.CONFLUENCE]
        assert result.entity_hint is None
        assert result.used_fallback is False

    def test_operational_status_axis_selects_jira_only(self):
        router = TruthAxisRouter()
        selector = SourceSelector(truth_axis_router=router)
        query = "Какой сейчас статус миграции DNS на данный момент?"

        result = selector.select(query)

        assert result.axis is TruthAxis.OPERATIONAL_STATUS
        assert result.selected_sources == [SourceType.JIRA]
        assert result.used_fallback is False

    def test_no_signal_at_all_triggers_honest_fallback(self):
        selector = SourceSelector()  # no router injected either
        result = selector.select(NEUTRAL_QUERY)

        assert result.used_fallback is True
        assert result.entity_hint is None
        assert set(result.selected_sources) == set(DEFAULT_FALLBACK_SOURCES)

    def test_ambiguous_axis_with_router_also_triggers_fallback(self):
        router = TruthAxisRouter()
        selector = SourceSelector(truth_axis_router=router)
        # Mixes design-intent and status markers -> AMBIGUOUS per 2.4,
        # and carries no ticket-number or entity-hint keyword either
        # (deliberately avoids "регламент"/"тред"/etc. so find_entity_hint
        # stays None and the decision actually reaches axis classification).
        query = (
            "Целевая архитектура DNS должна быть в облаке, а какой "
            "сейчас статус миграции?"
        )

        result = selector.select(query)

        assert result.axis is TruthAxis.AMBIGUOUS
        assert result.used_fallback is True
        assert set(result.selected_sources) == set(DEFAULT_FALLBACK_SOURCES)

    def test_fallback_is_not_used_when_any_signal_fires(self):
        selector = SourceSelector()
        result = selector.select(TICKET_QUERY)
        assert result.used_fallback is False

    def test_result_carries_original_query(self):
        selector = SourceSelector()
        result = selector.select(TICKET_QUERY)
        assert result.query == TICKET_QUERY


@pytest.mark.unit
class TestCustomConfiguration:
    def test_custom_fallback_sources_are_respected(self):
        selector = SourceSelector(fallback_sources=[SourceType.OTHER])
        result = selector.select(NEUTRAL_QUERY)
        assert result.selected_sources == [SourceType.OTHER]
        assert result.used_fallback is True

    def test_custom_ticket_pattern_is_used(self):
        selector = SourceSelector(ticket_pattern=r"TASK-\d+")
        assert selector.find_entity_hint("см. TASK-42") is SourceType.JIRA
        # Default-shaped ticket id no longer matches the custom pattern.
        assert selector.find_entity_hint("см. SUPPORT-8842") is None


@pytest.mark.integration
class TestSourceSelectorIntegration:
    """Требует реальных коннекторов Confluence/Jira/Slack, чтобы измерить
    фактическую экономию вызовов и латентности — не выполняется в
    юнит-тестах курса."""

    def test_measures_real_call_savings_against_live_connectors(self):
        pytest.skip(
            "Требует настроенных коннекторов Confluence, Jira и Slack "
            "для измерения реальной экономии вызовов"
        )
