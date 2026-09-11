"""Tests for app/context/knowledge_hierarchy.py (Lesson 2.5, synthesis of M2).

structure_scorer, thread_filter, page_trust_scorer and truth_axis_router are
injected via the constructor as MagicMock objects — these tests check ONLY
the orchestration logic of KnowledgeHierarchyResolver (which dependency is
called for which candidate, how weights and the axis-mismatch penalty are
combined, tie-breaking, cross-check grouping), not the correctness of each
dependency (already covered by its own dedicated test file:
test_structure_scorer.py, test_thread_filter.py, test_page_trust_scorer.py,
test_truth_axis_router.py).

Run unit tests only:
    pytest tests/test_knowledge_hierarchy.py -v -m unit
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.context.knowledge_hierarchy import (
    AxisWeights,
    DEFAULT_NEUTRAL_TRUST,
    HierarchyResolution,
    KnowledgeCandidate,
    KnowledgeHierarchyResolver,
    PriorityBreakdown,
)
from app.context.message_quality import ChatMessage
from app.context.page_trust_scorer import ConfluencePage
from app.context.source_registry import SourceType
from app.context.thread_filter import Thread
from app.context.truth_axis_router import QueryTruthClassification, TruthAxis


TODAY = date(2026, 1, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_resolver(**kwargs):
    structure_scorer = MagicMock()
    thread_filter = MagicMock()
    page_trust_scorer = MagicMock()
    truth_axis_router = MagicMock()
    resolver = KnowledgeHierarchyResolver(
        structure_scorer=structure_scorer,
        thread_filter=thread_filter,
        page_trust_scorer=page_trust_scorer,
        truth_axis_router=truth_axis_router,
        **kwargs,
    )
    return resolver, structure_scorer, thread_filter, page_trust_scorer, truth_axis_router


def _candidate(fragment_id, source_type, text="текст", confluence_page=None, thread=None):
    return KnowledgeCandidate(
        fragment_id=fragment_id,
        source_type=source_type,
        text=text,
        confluence_page=confluence_page,
        thread=thread,
    )


def _confluence_page(topic_key="ont-activation"):
    return ConfluencePage(
        page_id="p-1", title="Страница", space_key="techdocs", topic_key=topic_key,
        owner="owner", last_reviewed_at=date(2025, 12, 1), last_edited_at=date(2025, 12, 1),
        view_count_90d=100, is_deprecated=False,
    )


def _thread():
    return Thread(
        thread_id="t-1", channel="#noc-incidents", topic_key="olt-reboot",
        last_active_at="2025-09-02T14:22:00",
        messages=[ChatMessage("Игорь", "делаем graceful restart", "14:22")],
    )


def _classification(axis, recommended_source_type):
    return QueryTruthClassification(
        query="запрос", axis=axis,
        matched_intent_keywords=[], matched_status_keywords=[],
        recommended_source_type=recommended_source_type,
    )


# ---------------------------------------------------------------------------
# weights_for / register_weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWeights:
    def test_known_source_type_returns_configured_weights(self):
        resolver, *_ = _build_resolver()
        weights = resolver.weights_for(SourceType.CONFLUENCE)
        assert isinstance(weights, AxisWeights)
        assert weights.trust == pytest.approx(0.45)

    def test_never_registered_type_falls_back_to_balanced_default(self):
        resolver, *_ = _build_resolver(weights={})
        weights = resolver.weights_for(SourceType.JIRA)
        assert weights == AxisWeights()

    def test_register_weights_overrides_without_touching_others(self):
        resolver, *_ = _build_resolver()
        custom = AxisWeights(structure=0.1, trust=0.8, axis_alignment=0.1)

        resolver.register_weights(SourceType.OTHER, custom)

        assert resolver.weights_for(SourceType.OTHER) == custom
        # Другой, уже сконфигурированный тип не пострадал от регистрации
        assert resolver.weights_for(SourceType.CONFLUENCE).trust == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# structure_component / trust_component
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStructureComponent:
    def test_delegates_to_structure_scorer(self):
        resolver, structure_scorer, *_ = _build_resolver()
        structure_scorer.score.return_value = 0.73
        candidate = _candidate("f1", SourceType.JIRA, text="Root Cause: ...")

        result = resolver.structure_component(candidate)

        assert result == 0.73
        structure_scorer.score.assert_called_once_with("Root Cause: ...")


@pytest.mark.unit
class TestTrustComponent:
    def test_confluence_page_uses_page_trust_scorer(self):
        resolver, _, thread_filter, page_trust_scorer, _ = _build_resolver()
        page = _confluence_page()
        page_trust_scorer.score.return_value = 0.2
        candidate = _candidate("f1", SourceType.CONFLUENCE, confluence_page=page)

        trust, note = resolver.trust_component(candidate, TODAY)

        assert trust == 0.2
        page_trust_scorer.score.assert_called_once_with(page, TODAY)
        thread_filter.is_worth_indexing.assert_not_called()

    def test_thread_not_worth_indexing_returns_zero_without_signal_ratio(self):
        resolver, _, thread_filter, _, _ = _build_resolver()
        thread_filter.is_worth_indexing.return_value = False
        candidate = _candidate("f1", SourceType.SLACK, thread=_thread())

        trust, note = resolver.trust_component(candidate, TODAY)

        assert trust == 0.0
        thread_filter.signal_ratio.assert_not_called()

    def test_thread_worth_indexing_uses_signal_ratio(self):
        resolver, _, thread_filter, _, _ = _build_resolver()
        thread_filter.is_worth_indexing.return_value = True
        thread_filter.signal_ratio.return_value = 0.67
        candidate = _candidate("f1", SourceType.SLACK, thread=_thread())

        trust, note = resolver.trust_component(candidate, TODAY)

        assert trust == 0.67

    def test_no_specialized_scorer_uses_neutral_default(self):
        resolver, *_ = _build_resolver()
        candidate = _candidate("f1", SourceType.JIRA)

        trust, note = resolver.trust_component(candidate, TODAY)

        assert trust == DEFAULT_NEUTRAL_TRUST

    def test_custom_neutral_default_is_respected(self):
        resolver, *_ = _build_resolver(neutral_trust_default=0.42)
        candidate = _candidate("f1", SourceType.OTHER)

        trust, _ = resolver.trust_component(candidate, TODAY)

        assert trust == 0.42


# ---------------------------------------------------------------------------
# axis_alignment_component
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAxisAlignmentComponent:
    def test_ambiguous_axis_is_neutral_for_any_source(self):
        resolver, *_ = _build_resolver()
        classification = _classification(TruthAxis.AMBIGUOUS, None)
        candidate = _candidate("f1", SourceType.CONFLUENCE)

        score, mismatch = resolver.axis_alignment_component(candidate, classification)

        assert score == 0.5
        assert mismatch is False

    def test_matching_recommended_type_scores_full(self):
        resolver, *_ = _build_resolver()
        classification = _classification(TruthAxis.OPERATIONAL_STATUS, SourceType.JIRA)
        candidate = _candidate("f1", SourceType.JIRA)

        score, mismatch = resolver.axis_alignment_component(candidate, classification)

        assert score == 1.0
        assert mismatch is False

    def test_opposite_confluence_vs_jira_is_hard_mismatch(self):
        resolver, *_ = _build_resolver()
        classification = _classification(TruthAxis.OPERATIONAL_STATUS, SourceType.JIRA)
        candidate = _candidate("f1", SourceType.CONFLUENCE)

        score, mismatch = resolver.axis_alignment_component(candidate, classification)

        assert score == 0.0
        assert mismatch is True

    def test_opposite_jira_vs_confluence_is_hard_mismatch(self):
        resolver, *_ = _build_resolver()
        classification = _classification(TruthAxis.DESIGN_INTENT, SourceType.CONFLUENCE)
        candidate = _candidate("f1", SourceType.JIRA)

        score, mismatch = resolver.axis_alignment_component(candidate, classification)

        assert score == 0.0
        assert mismatch is True

    def test_slack_candidate_is_neutral_regardless_of_recommended_type(self):
        resolver, *_ = _build_resolver()
        classification = _classification(TruthAxis.DESIGN_INTENT, SourceType.CONFLUENCE)
        candidate = _candidate("f1", SourceType.SLACK)

        score, mismatch = resolver.axis_alignment_component(candidate, classification)

        assert score == 0.5
        assert mismatch is False


# ---------------------------------------------------------------------------
# score_candidate — combination + mismatch penalty
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScoreCandidate:
    def test_combines_three_signals_with_configured_weights(self):
        resolver, structure_scorer, _, page_trust_scorer, _ = _build_resolver(
            weights={SourceType.CONFLUENCE: AxisWeights(structure=0.5, trust=0.5, axis_alignment=0.0)}
        )
        structure_scorer.score.return_value = 0.8
        page = _confluence_page()
        page_trust_scorer.score.return_value = 0.4
        candidate = _candidate("f1", SourceType.CONFLUENCE, confluence_page=page)
        classification = _classification(TruthAxis.AMBIGUOUS, None)

        breakdown = resolver.score_candidate(candidate, classification, TODAY)

        assert isinstance(breakdown, PriorityBreakdown)
        assert breakdown.final_score == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)
        assert breakdown.axis_mismatch is False

    def test_hard_mismatch_applies_penalty_multiplier(self):
        penalty = 0.5
        resolver, structure_scorer, _, page_trust_scorer, _ = _build_resolver(
            weights={SourceType.CONFLUENCE: AxisWeights(structure=0.0, trust=1.0, axis_alignment=0.0)},
            axis_mismatch_penalty=penalty,
        )
        structure_scorer.score.return_value = 0.0  # вес структуры и так 0.0, но мок должен быть числом
        page = _confluence_page()
        page_trust_scorer.score.return_value = 0.9  # высокое доверие
        candidate = _candidate("f1", SourceType.CONFLUENCE, confluence_page=page)
        # Запрос про операционный статус -> рекомендован Jira, кандидат Confluence -> mismatch
        classification = _classification(TruthAxis.OPERATIONAL_STATUS, SourceType.JIRA)

        breakdown = resolver.score_candidate(candidate, classification, TODAY)

        assert breakdown.axis_mismatch is True
        assert breakdown.final_score == pytest.approx(0.9 * penalty)
        assert any("2.4" in note for note in breakdown.notes)


# ---------------------------------------------------------------------------
# rank — ordering and determinism
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRank:
    def test_orders_by_descending_final_score(self):
        resolver, structure_scorer, thread_filter, page_trust_scorer, truth_axis_router = _build_resolver(
            weights={
                SourceType.CONFLUENCE: AxisWeights(structure=1.0, trust=0.0, axis_alignment=0.0),
                SourceType.JIRA: AxisWeights(structure=1.0, trust=0.0, axis_alignment=0.0),
            }
        )
        truth_axis_router.classify.return_value = _classification(TruthAxis.AMBIGUOUS, None)
        structure_scorer.score.side_effect = lambda text: {"low": 0.2, "high": 0.9}[text]
        low = _candidate("f-low", SourceType.CONFLUENCE, text="low")
        high = _candidate("f-high", SourceType.JIRA, text="high")

        ranked = resolver.rank([low, high], "запрос", TODAY)

        assert [b.fragment_id for b in ranked] == ["f-high", "f-low"]

    def test_ties_broken_by_fragment_id(self):
        resolver, structure_scorer, thread_filter, page_trust_scorer, truth_axis_router = _build_resolver()
        truth_axis_router.classify.return_value = _classification(TruthAxis.AMBIGUOUS, None)
        structure_scorer.score.return_value = 0.5
        a = _candidate("z-fragment", SourceType.JIRA)
        b = _candidate("a-fragment", SourceType.JIRA)

        ranked = resolver.rank([a, b], "запрос", TODAY)

        assert [item.fragment_id for item in ranked] == ["a-fragment", "z-fragment"]


# ---------------------------------------------------------------------------
# resolve — cross-check grouping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolve:
    def test_ambiguous_axis_recommends_cross_check_with_top_per_type(self):
        resolver, structure_scorer, thread_filter, page_trust_scorer, truth_axis_router = _build_resolver()
        classification = _classification(TruthAxis.AMBIGUOUS, None)
        truth_axis_router.classify.return_value = classification
        truth_axis_router.should_cross_check.return_value = True
        structure_scorer.score.return_value = 0.5
        confluence_candidate = _candidate("c1", SourceType.CONFLUENCE)
        jira_candidate = _candidate("j1", SourceType.JIRA)

        result = resolver.resolve([confluence_candidate, jira_candidate], "запрос", TODAY)

        assert isinstance(result, HierarchyResolution)
        assert result.cross_check_recommended is True
        assert result.top_by_source_type is not None
        assert set(result.top_by_source_type.keys()) == {SourceType.CONFLUENCE, SourceType.JIRA}

    def test_unambiguous_axis_has_no_cross_check_grouping(self):
        resolver, structure_scorer, thread_filter, page_trust_scorer, truth_axis_router = _build_resolver()
        classification = _classification(TruthAxis.OPERATIONAL_STATUS, SourceType.JIRA)
        truth_axis_router.classify.return_value = classification
        truth_axis_router.should_cross_check.return_value = False
        structure_scorer.score.return_value = 0.5
        candidate = _candidate("j1", SourceType.JIRA)

        result = resolver.resolve([candidate], "запрос", TODAY)

        assert result.cross_check_recommended is False
        assert result.top_by_source_type is None
        assert result.axis is TruthAxis.OPERATIONAL_STATUS


# ---------------------------------------------------------------------------
# Integration tests — require real TextStructureScorer/ThreadNoiseFilter/
# PageTrustScorer/TruthAxisRouter from lessons 2.1-2.4 fully implemented
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKnowledgeHierarchyResolverIntegration:
    """Собирает реальные (не mock) TextStructureScorer, ThreadNoiseFilter,
    PageTrustScorer и TruthAxisRouter из уроков 2.1-2.4. Помечено как
    integration, потому что предполагает, что ВСЕ четыре предыдущих задания
    уже решены — запускайте вручную после завершения уроков 2.1-2.4."""

    def test_reproduces_pechora_telecom_dns_neva_resolution(self):
        pytest.skip(
            "Требует реализованных TextStructureScorer, ThreadNoiseFilter, "
            "PageTrustScorer и TruthAxisRouter из уроков 2.1-2.4 — "
            "запускайте вручную:\n"
            "  pytest tests/test_knowledge_hierarchy.py -m integration -v"
        )
