import pytest

from app.context.source_registry import SourceType
from app.context.structure_classifier import (
    RetrievalTreatment,
    SourceStructureClassifier,
    StructuralProfile,
    StructureClass,
)


@pytest.mark.unit
class TestClassifyStructure:
    def test_confluence_is_document(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.classify_structure(SourceType.CONFLUENCE)
            == StructureClass.DOCUMENT
        )

    def test_jira_is_semi_structured(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.classify_structure(SourceType.JIRA)
            == StructureClass.SEMI_STRUCTURED
        )

    def test_slack_is_conversation(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.classify_structure(SourceType.SLACK)
            == StructureClass.CONVERSATION
        )

    def test_other_raises_value_error(self):
        classifier = SourceStructureClassifier()
        with pytest.raises(ValueError):
            classifier.classify_structure(SourceType.OTHER)


@pytest.mark.unit
class TestRecommendTreatment:
    def test_document_gets_section_chunk(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.recommend_treatment(StructureClass.DOCUMENT)
            == RetrievalTreatment.SECTION_CHUNK
        )

    def test_semi_structured_gets_field_level_extract(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.recommend_treatment(StructureClass.SEMI_STRUCTURED)
            == RetrievalTreatment.FIELD_LEVEL_EXTRACT
        )

    def test_conversation_gets_thread_aware_chunk(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.recommend_treatment(StructureClass.CONVERSATION)
            == RetrievalTreatment.THREAD_AWARE_CHUNK
        )


@pytest.mark.unit
class TestExpectedHalfLifeDays:
    def test_document_half_life(self):
        classifier = SourceStructureClassifier()
        assert classifier.expected_half_life_days(StructureClass.DOCUMENT) == 90

    def test_semi_structured_half_life(self):
        classifier = SourceStructureClassifier()
        assert (
            classifier.expected_half_life_days(StructureClass.SEMI_STRUCTURED) == 14
        )

    def test_conversation_half_life(self):
        classifier = SourceStructureClassifier()
        assert classifier.expected_half_life_days(StructureClass.CONVERSATION) == 1

    def test_document_half_life_greater_than_conversation(self):
        classifier = SourceStructureClassifier()
        doc_half_life = classifier.expected_half_life_days(StructureClass.DOCUMENT)
        chat_half_life = classifier.expected_half_life_days(
            StructureClass.CONVERSATION
        )
        assert doc_half_life > chat_half_life


@pytest.mark.unit
class TestProfile:
    def test_confluence_profile(self):
        classifier = SourceStructureClassifier()
        profile = classifier.profile(SourceType.CONFLUENCE)
        assert profile == StructuralProfile(
            source_type=SourceType.CONFLUENCE,
            structure_class=StructureClass.DOCUMENT,
            treatment=RetrievalTreatment.SECTION_CHUNK,
            expected_half_life_days=90,
        )

    def test_jira_profile(self):
        classifier = SourceStructureClassifier()
        profile = classifier.profile(SourceType.JIRA)
        assert profile.structure_class == StructureClass.SEMI_STRUCTURED
        assert profile.treatment == RetrievalTreatment.FIELD_LEVEL_EXTRACT
        assert profile.expected_half_life_days == 14

    def test_slack_profile(self):
        classifier = SourceStructureClassifier()
        profile = classifier.profile(SourceType.SLACK)
        assert profile.structure_class == StructureClass.CONVERSATION
        assert profile.treatment == RetrievalTreatment.THREAD_AWARE_CHUNK
        assert profile.expected_half_life_days == 1

    def test_other_raises_value_error(self):
        classifier = SourceStructureClassifier()
        with pytest.raises(ValueError):
            classifier.profile(SourceType.OTHER)


@pytest.mark.integration
class TestStructureClassifierIntegration:
    """Требует реальных коннекторов Confluence/Jira/Slack, которые сами
    сообщают свой контент-тип (например, вложения, кастомные поля Jira
    неизвестных типов). В этом уроке не используется — заготовка для
    более поздних курсов, где классификация обогащается метаданными
    из живых источников, а не только из SourceType."""

    def test_classification_from_live_connector_metadata(self):
        pytest.skip("Требует настроенных коннекторов Confluence/Jira/Slack")
