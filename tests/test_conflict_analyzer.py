"""Tests for app/context/conflict_analyzer.py (Lesson 1.4, synthesis of M1).

registry, quality_scorer and structure_classifier are injected via the
constructor as MagicMock objects — these tests check ONLY the orchestration
logic of ContextConflictAnalyzer (classification order, which dependency is
called and how its result is interpreted), not the correctness of each
dependency (already covered by its own dedicated test file: test_source_registry.py,
test_message_quality.py, test_structure_classifier.py).

Run unit tests only:
    pytest tests/test_conflict_analyzer.py -v -m unit
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.context.conflict_analyzer import (
    ConflictType,
    ContextConflictAnalyzer,
    ContextFragment,
)
from app.context.message_quality import ChatMessage
from app.context.source_registry import SourceType
from app.context.structure_classifier import StructureClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analyzer():
    registry = MagicMock()
    quality_scorer = MagicMock()
    structure_classifier = MagicMock()
    analyzer = ContextConflictAnalyzer(
        registry=registry,
        quality_scorer=quality_scorer,
        structure_classifier=structure_classifier,
    )
    return analyzer, registry, quality_scorer, structure_classifier


def _fragment(source_name, source_type, text="текст", chat_message=None):
    return ContextFragment(
        source_name=source_name,
        source_type=source_type,
        text=text,
        chat_message=chat_message,
    )


def _structure_by_type(mapping):
    """
    Возвращает side_effect-функцию для structure_classifier.classify_structure,
    определяющую результат по source_type фрагмента, а НЕ по порядку вызова.
    Реализация может вызывать classify_structure любое число раз (один раз
    при классификации типа конфликта, ещё раз при поиске неформального
    фрагмента) — тест не должен зависеть от того, сколько именно раз.
    """
    def _side_effect(source_type):
        return mapping[source_type]

    return _side_effect


# ---------------------------------------------------------------------------
# classify_conflict_type
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyConflictType:
    def test_same_source_name_is_contradictory_within_source(self):
        analyzer, _, _, structure_classifier = _build_analyzer()
        a = _fragment("slack_platform_migrations", SourceType.SLACK, text="A")
        b = _fragment("slack_platform_migrations", SourceType.SLACK, text="B")

        result = analyzer.classify_conflict_type(a, b)

        assert result == ConflictType.CONTRADICTORY_WITHIN_SOURCE

    def test_same_structure_class_is_stale_vs_fresh(self):
        analyzer, _, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        a = _fragment("confluence_plan_v1", SourceType.CONFLUENCE)
        b = _fragment("confluence_plan_v2", SourceType.CONFLUENCE)

        result = analyzer.classify_conflict_type(a, b)

        assert result == ConflictType.STALE_VS_FRESH

    def test_conversation_vs_document_is_authoritative_vs_informal(self):
        analyzer, _, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.side_effect = _structure_by_type({
            SourceType.CONFLUENCE: StructureClass.DOCUMENT,
            SourceType.SLACK: StructureClass.CONVERSATION,
        })
        a = _fragment("confluence_pdn_plan", SourceType.CONFLUENCE)
        b = _fragment("slack_platform_migrations", SourceType.SLACK)

        result = analyzer.classify_conflict_type(a, b)

        assert result == ConflictType.AUTHORITATIVE_VS_INFORMAL

    def test_document_vs_semi_structured_is_structured_vs_unstructured(self):
        analyzer, _, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.side_effect = _structure_by_type({
            SourceType.CONFLUENCE: StructureClass.DOCUMENT,
            SourceType.JIRA: StructureClass.SEMI_STRUCTURED,
        })
        a = _fragment("confluence_pdn_plan", SourceType.CONFLUENCE)
        b = _fragment("jira_pdn_142", SourceType.JIRA)

        result = analyzer.classify_conflict_type(a, b)

        assert result == ConflictType.STRUCTURED_VS_UNSTRUCTURED


# ---------------------------------------------------------------------------
# analyze_pair: CONTRADICTORY_WITHIN_SOURCE
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzePairContradictoryWithinSource:
    def test_recommends_higher_scoring_message(self):
        analyzer, registry, quality_scorer, _ = _build_analyzer()
        msg_a = ChatMessage("Игорь", "предлагаю AES-128-CBC", "10:03")
        msg_b = ChatMessage("Вячеслав", "фиксируем: GCM, 256", "14:40")
        a = _fragment(
            "slack_platform_migrations", SourceType.SLACK,
            text="предлагаю AES-128-CBC", chat_message=msg_a,
        )
        b = _fragment(
            "slack_platform_migrations", SourceType.SLACK,
            text="фиксируем: GCM, 256", chat_message=msg_b,
        )
        quality_scorer.score.side_effect = lambda m: 0.4 if m is msg_a else 0.8

        finding = analyzer.analyze_pair(a, b)

        assert finding.conflict_type == ConflictType.CONTRADICTORY_WITHIN_SOURCE
        assert finding.recommended_source_name == "slack_platform_migrations"
        registry.resolve_conflict.assert_not_called()

    def test_equal_scores_escalate_without_recommendation(self):
        analyzer, registry, quality_scorer, _ = _build_analyzer()
        msg_a = ChatMessage("Игорь", "предложение A", "10:03")
        msg_b = ChatMessage("Марина", "предложение B", "10:07")
        a = _fragment("slack_ch", SourceType.SLACK, chat_message=msg_a)
        b = _fragment("slack_ch", SourceType.SLACK, chat_message=msg_b)
        quality_scorer.score.return_value = 0.6

        finding = analyzer.analyze_pair(a, b)

        assert finding.recommended_source_name is None

    def test_missing_chat_message_cannot_be_resolved(self):
        analyzer, registry, quality_scorer, _ = _build_analyzer()
        a = _fragment("slack_ch", SourceType.SLACK, chat_message=None)
        b = _fragment(
            "slack_ch", SourceType.SLACK,
            chat_message=ChatMessage("Игорь", "текст", "10:03"),
        )

        finding = analyzer.analyze_pair(a, b)

        assert finding.recommended_source_name is None
        quality_scorer.score.assert_not_called()


# ---------------------------------------------------------------------------
# analyze_pair: AUTHORITATIVE_VS_INFORMAL
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzePairAuthoritativeVsInformal:
    def test_low_quality_informal_fragment_loses_without_registry(self):
        analyzer, registry, quality_scorer, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.side_effect = _structure_by_type({
            SourceType.CONFLUENCE: StructureClass.DOCUMENT,
            SourceType.SLACK: StructureClass.CONVERSATION,
        })
        quality_scorer.score.return_value = 0.2
        formal = _fragment("confluence_pdn_plan", SourceType.CONFLUENCE)
        informal_msg = ChatMessage("Вячеслав", "го", "10:09")
        informal = _fragment(
            "slack_platform_migrations", SourceType.SLACK, chat_message=informal_msg
        )

        finding = analyzer.analyze_pair(formal, informal)

        assert finding.conflict_type == ConflictType.AUTHORITATIVE_VS_INFORMAL
        assert finding.recommended_source_name == "confluence_pdn_plan"
        registry.resolve_conflict.assert_not_called()

    def test_high_quality_informal_fragment_falls_through_to_registry(self):
        analyzer, registry, quality_scorer, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.side_effect = _structure_by_type({
            SourceType.CONFLUENCE: StructureClass.DOCUMENT,
            SourceType.SLACK: StructureClass.CONVERSATION,
        })
        quality_scorer.score.return_value = 0.9
        registry.resolve_conflict.return_value = "slack_platform_migrations"
        formal = _fragment("confluence_pdn_plan", SourceType.CONFLUENCE)
        informal_msg = ChatMessage(
            "Вячеслав", "из-за задержки поставки от вендора дата съезжает на 20 июля", "14:00"
        )
        informal = _fragment(
            "slack_platform_migrations", SourceType.SLACK, chat_message=informal_msg
        )

        finding = analyzer.analyze_pair(formal, informal)

        assert finding.recommended_source_name == "slack_platform_migrations"
        registry.resolve_conflict.assert_called_once_with(
            "confluence_pdn_plan", "slack_platform_migrations"
        )


# ---------------------------------------------------------------------------
# analyze_pair: STALE_VS_FRESH / STRUCTURED_VS_UNSTRUCTURED delegate to registry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzePairDelegatesToRegistry:
    def test_stale_vs_fresh_uses_registry_result(self):
        analyzer, registry, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        registry.resolve_conflict.return_value = "confluence_plan_v2"
        a = _fragment("confluence_plan_v1", SourceType.CONFLUENCE)
        b = _fragment("confluence_plan_v2", SourceType.CONFLUENCE)

        finding = analyzer.analyze_pair(a, b)

        assert finding.conflict_type == ConflictType.STALE_VS_FRESH
        assert finding.recommended_source_name == "confluence_plan_v2"

    def test_structured_vs_unstructured_uses_registry_result(self):
        analyzer, registry, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.side_effect = _structure_by_type({
            SourceType.CONFLUENCE: StructureClass.DOCUMENT,
            SourceType.JIRA: StructureClass.SEMI_STRUCTURED,
        })
        registry.resolve_conflict.return_value = "jira_pdn_142"
        a = _fragment("confluence_pdn_plan", SourceType.CONFLUENCE)
        b = _fragment("jira_pdn_142", SourceType.JIRA)

        finding = analyzer.analyze_pair(a, b)

        assert finding.conflict_type == ConflictType.STRUCTURED_VS_UNSTRUCTURED
        assert finding.recommended_source_name == "jira_pdn_142"

    def test_registry_value_error_escalates(self):
        analyzer, registry, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        registry.resolve_conflict.side_effect = ValueError("equal weights")
        a = _fragment("confluence_a", SourceType.CONFLUENCE)
        b = _fragment("confluence_b", SourceType.CONFLUENCE)

        finding = analyzer.analyze_pair(a, b)

        assert finding.recommended_source_name is None

    def test_registry_key_error_escalates(self):
        analyzer, registry, _, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        registry.resolve_conflict.side_effect = KeyError("unregistered_source")
        a = _fragment("confluence_a", SourceType.CONFLUENCE)
        b = _fragment("unregistered_source", SourceType.CONFLUENCE)

        finding = analyzer.analyze_pair(a, b)

        assert finding.recommended_source_name is None


# ---------------------------------------------------------------------------
# analyze: batch orchestration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyze:
    def test_skips_identical_text_pairs(self):
        analyzer, registry, quality_scorer, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        registry.resolve_conflict.return_value = "confluence_a"
        a = _fragment("confluence_a", SourceType.CONFLUENCE, text="одинаковый текст")
        b = _fragment("confluence_b", SourceType.CONFLUENCE, text="одинаковый текст")

        findings = analyzer.analyze([a, b])

        assert findings == []

    def test_analyzes_all_distinct_pairs(self):
        analyzer, registry, quality_scorer, structure_classifier = _build_analyzer()
        structure_classifier.classify_structure.return_value = StructureClass.DOCUMENT
        registry.resolve_conflict.return_value = "confluence_a"
        a = _fragment("confluence_a", SourceType.CONFLUENCE, text="текст A")
        b = _fragment("confluence_b", SourceType.CONFLUENCE, text="текст B")
        c = _fragment("confluence_c", SourceType.CONFLUENCE, text="текст C")

        findings = analyzer.analyze([a, b, c])

        # 3 фрагмента -> 3 уникальные пары (a,b) (a,c) (b,c)
        assert len(findings) == 3

    def test_empty_and_single_fragment_produce_no_findings(self):
        analyzer, *_ = _build_analyzer()

        assert analyzer.analyze([]) == []
        assert analyzer.analyze([_fragment("only_one", SourceType.CONFLUENCE)]) == []


# ---------------------------------------------------------------------------
# Integration tests — require real SourceRegistry/MessageQualityScorer/
# SourceStructureClassifier from lessons 1.1-1.3 fully implemented
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestContextConflictAnalyzerIntegration:
    """Собирает реальные (не mock) SourceRegistry, MessageQualityScorer и
    SourceStructureClassifier из уроков 1.1-1.3. Помечено как integration,
    потому что предполагает, что ВСЕ три предыдущих задания уже решены —
    запускайте вручную после завершения уроков 1.1-1.3."""

    def test_reproduces_pechora_telecom_incident_1_resolution(self):
        pytest.skip(
            "Требует реализованных SourceRegistry, MessageQualityScorer и "
            "SourceStructureClassifier из уроков 1.1-1.3 — запускайте вручную:\n"
            "  pytest tests/test_conflict_analyzer.py -m integration -v"
        )
