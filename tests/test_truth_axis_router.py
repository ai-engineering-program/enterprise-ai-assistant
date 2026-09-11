import pytest

from app.context.source_registry import SourceType
from app.context.truth_axis_router import (
    QueryTruthClassification,
    TruthAxis,
    TruthAxisRouter,
)


# Запрос из инцидента урока 2.4: "Целевая архитектура DNS" (Confluence)
# описывает design intent, тикет миграции сегмента "Северный" (Jira)
# описывает operational status.
DESIGN_QUERY = (
    "Как должно быть организовано разрешение DNS согласно регламенту "
    "проекта Нева?"
)
STATUS_QUERY = (
    "Какой сейчас статус тикета миграции DNS и что используется "
    "на данный момент?"
)
AMBIGUOUS_QUERY = (
    "Согласно регламенту DNS должно быть на облаке, а какой сейчас "
    "статус тикета миграции?"
)
NEUTRAL_QUERY = "Расскажи про DNS в Печора Телеком."


@pytest.mark.unit
class TestFindMatches:
    def test_finds_matching_keyword_case_insensitive(self):
        router = TruthAxisRouter()
        matches = router.find_matches(
            "Какой СЕЙЧАС статус подключения?", ["сейчас", "должно быть"]
        )
        assert matches == ["сейчас"]

    def test_preserves_keyword_order_not_text_order(self):
        router = TruthAxisRouter()
        query = "статус тикета и целевая архитектура в одном вопросе"
        matches = router.find_matches(
            query, ["целевая архитектура", "статус тикета"]
        )
        assert matches == ["целевая архитектура", "статус тикета"]

    def test_no_matches_returns_empty_list(self):
        router = TruthAxisRouter()
        assert router.find_matches("произвольный текст без маркеров", ["сейчас"]) == []

    def test_empty_keywords_returns_empty_list(self):
        router = TruthAxisRouter()
        assert router.find_matches(STATUS_QUERY, []) == []


@pytest.mark.unit
class TestClassify:
    def test_design_only_query_classified_as_design_intent(self):
        router = TruthAxisRouter()
        result = router.classify(DESIGN_QUERY)
        assert isinstance(result, QueryTruthClassification)
        assert result.axis is TruthAxis.DESIGN_INTENT
        assert result.matched_intent_keywords
        assert result.matched_status_keywords == []
        assert result.recommended_source_type is SourceType.CONFLUENCE

    def test_status_only_query_classified_as_operational_status(self):
        router = TruthAxisRouter()
        result = router.classify(STATUS_QUERY)
        assert result.axis is TruthAxis.OPERATIONAL_STATUS
        assert result.matched_status_keywords
        assert result.matched_intent_keywords == []
        assert result.recommended_source_type is SourceType.JIRA

    def test_mixed_markers_classified_as_ambiguous(self):
        router = TruthAxisRouter()
        result = router.classify(AMBIGUOUS_QUERY)
        assert result.axis is TruthAxis.AMBIGUOUS
        assert result.matched_intent_keywords
        assert result.matched_status_keywords
        assert result.recommended_source_type is None

    def test_no_markers_defaults_to_ambiguous(self):
        router = TruthAxisRouter()
        result = router.classify(NEUTRAL_QUERY)
        assert result.axis is TruthAxis.AMBIGUOUS
        assert result.matched_intent_keywords == []
        assert result.matched_status_keywords == []
        assert result.recommended_source_type is None

    def test_classification_carries_original_query(self):
        router = TruthAxisRouter()
        result = router.classify(DESIGN_QUERY)
        assert result.query == DESIGN_QUERY


@pytest.mark.unit
class TestClassifyAxis:
    def test_matches_classify_result(self):
        router = TruthAxisRouter()
        assert router.classify_axis(STATUS_QUERY) == router.classify(STATUS_QUERY).axis
        assert router.classify_axis(STATUS_QUERY) is TruthAxis.OPERATIONAL_STATUS


@pytest.mark.unit
class TestRecommendSourceType:
    def test_design_intent_recommends_confluence(self):
        router = TruthAxisRouter()
        assert router.recommend_source_type(TruthAxis.DESIGN_INTENT) is SourceType.CONFLUENCE

    def test_operational_status_recommends_jira(self):
        router = TruthAxisRouter()
        assert router.recommend_source_type(TruthAxis.OPERATIONAL_STATUS) is SourceType.JIRA

    def test_ambiguous_recommends_nothing(self):
        router = TruthAxisRouter()
        assert router.recommend_source_type(TruthAxis.AMBIGUOUS) is None


@pytest.mark.unit
class TestShouldCrossCheck:
    def test_ambiguous_classification_requires_cross_check(self):
        router = TruthAxisRouter()
        classification = router.classify(AMBIGUOUS_QUERY)
        assert router.should_cross_check(classification) is True

    def test_unambiguous_classification_does_not_require_cross_check(self):
        router = TruthAxisRouter()
        design = router.classify(DESIGN_QUERY)
        status = router.classify(STATUS_QUERY)
        assert router.should_cross_check(design) is False
        assert router.should_cross_check(status) is False


@pytest.mark.integration
class TestTruthAxisRouterIntegration:
    """Требует реального retrieval-слоя с подключёнными Confluence и Jira
    коннекторами — не выполняется в юнит-тестах курса."""

    def test_routes_real_query_against_live_sources(self):
        pytest.skip("Требует настроенных коннекторов Confluence и Jira")
