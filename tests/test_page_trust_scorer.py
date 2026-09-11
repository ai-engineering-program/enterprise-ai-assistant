import pytest
from datetime import date

from app.context.page_trust_scorer import ConfluencePage, PageTrustScorer


TODAY = date(2026, 1, 15)


def _official_page() -> ConfluencePage:
    """Официальная, но осиротевшая страница из инцидента урока 2.3:
    владелец уволен (owner=None), ревью не пересматривалось 14+ месяцев."""
    return ConfluencePage(
        page_id="p-official",
        title="Активация абонентского оборудования GPON",
        space_key="techdocs",
        topic_key="ont-activation",
        owner=None,
        last_reviewed_at=date(2024, 10, 1),
        last_edited_at=date(2024, 10, 1),
        view_count_90d=340,
        is_deprecated=False,
    )


def _field_notes_page() -> ConfluencePage:
    """Актуальная страница полевых заметок: владелец активен, ревью недавнее."""
    return ConfluencePage(
        page_id="p-field",
        title="Заметки НОЦ: активация ONT Гигабит про",
        space_key="noc-notes",
        topic_key="ont-activation",
        owner="active.engineer",
        last_reviewed_at=date(2025, 12, 25),
        last_edited_at=date(2025, 12, 25),
        view_count_90d=12,
        is_deprecated=False,
    )


def _deprecated_page() -> ConfluencePage:
    """Страница с идеальными метаданными, но явно помеченная устаревшей."""
    return ConfluencePage(
        page_id="p-deprecated",
        title="Старый регламент (архив)",
        space_key="techdocs",
        topic_key="ont-activation",
        owner="active.engineer",
        last_reviewed_at=date(2026, 1, 1),
        last_edited_at=date(2026, 1, 1),
        view_count_90d=500,
        is_deprecated=True,
    )


def _never_reviewed_page() -> ConfluencePage:
    """Страница с владельцем, но ни разу не прошедшая ревью."""
    return ConfluencePage(
        page_id="p-never-reviewed",
        title="Регламент без ревью",
        space_key="techdocs",
        topic_key="unrelated-topic",
        owner="someone",
        last_reviewed_at=None,
        last_edited_at=date(2023, 1, 1),
        view_count_90d=50,
        is_deprecated=False,
    )


def _single_topic_page() -> ConfluencePage:
    """Единственная страница по своей теме — не может конфликтовать."""
    return ConfluencePage(
        page_id="p-solo",
        title="Уникальная тема",
        space_key="techdocs",
        topic_key="solo-topic",
        owner="active.engineer",
        last_reviewed_at=date(2025, 12, 1),
        last_edited_at=date(2025, 12, 1),
        view_count_90d=100,
        is_deprecated=False,
    )


@pytest.mark.unit
class TestDaysSinceReview:
    def test_none_when_never_reviewed(self):
        scorer = PageTrustScorer()
        assert scorer.days_since_review(_never_reviewed_page(), TODAY) is None

    def test_computes_days_delta(self):
        scorer = PageTrustScorer()
        page = _field_notes_page()
        assert scorer.days_since_review(page, TODAY) == (TODAY - page.last_reviewed_at).days


@pytest.mark.unit
class TestHasOwner:
    def test_missing_owner_is_false(self):
        scorer = PageTrustScorer()
        assert scorer.has_owner(_official_page()) is False

    def test_present_owner_is_true(self):
        scorer = PageTrustScorer()
        assert scorer.has_owner(_field_notes_page()) is True

    def test_blank_owner_is_false(self):
        scorer = PageTrustScorer()
        page = _field_notes_page()
        page.owner = "   "
        assert scorer.has_owner(page) is False


@pytest.mark.unit
class TestIsStaleByViews:
    def test_low_views_is_stale(self):
        scorer = PageTrustScorer(min_views_90d=5)
        page = _field_notes_page()
        page.view_count_90d = 2
        assert scorer.is_stale_by_views(page) is True

    def test_high_views_is_not_stale(self):
        scorer = PageTrustScorer(min_views_90d=5)
        assert scorer.is_stale_by_views(_official_page()) is False


@pytest.mark.unit
class TestAnalyze:
    def test_collects_all_signals(self):
        scorer = PageTrustScorer(min_views_90d=5)
        signals = scorer.analyze(_official_page(), TODAY)
        assert signals.has_owner is False
        assert signals.days_since_review is not None
        assert signals.is_stale_by_views is False
        assert signals.is_deprecated is False


@pytest.mark.unit
class TestScore:
    def test_deprecated_page_scores_zero_regardless_of_other_signals(self):
        scorer = PageTrustScorer()
        assert scorer.score(_deprecated_page(), TODAY) == 0.0

    def test_orphaned_stale_review_page_scores_low(self):
        scorer = PageTrustScorer(review_staleness_days=180)
        score = scorer.score(_official_page(), TODAY)
        # нет владельца (-0.3) и ревью просрочено (-0.2) -> не выше 0.5
        assert score <= 0.5

    def test_owned_recently_reviewed_page_scores_high(self):
        scorer = PageTrustScorer(review_staleness_days=180, min_views_90d=5)
        score = scorer.score(_field_notes_page(), TODAY)
        assert score >= 0.8

    def test_never_reviewed_page_is_penalized_like_orphan(self):
        scorer = PageTrustScorer()
        page = _never_reviewed_page()
        score = scorer.score(page, TODAY)
        # владелец есть, но ревью не было ни разу -> -0.3
        assert score == pytest.approx(0.7)

    def test_score_bounded_between_zero_and_one(self):
        scorer = PageTrustScorer()
        for page in (_official_page(), _field_notes_page(), _deprecated_page()):
            assert 0.0 <= scorer.score(page, TODAY) <= 1.0


@pytest.mark.unit
class TestRankByTrust:
    def test_higher_trust_page_ranked_first(self):
        scorer = PageTrustScorer()
        ranked = scorer.rank_by_trust(
            [_official_page(), _field_notes_page()], TODAY
        )
        assert [p.page_id for p in ranked] == ["p-field", "p-official"]

    def test_deterministic_tiebreak_by_page_id(self):
        scorer = PageTrustScorer()
        a = _field_notes_page()
        a.page_id = "p-b"
        b = _field_notes_page()
        b.page_id = "p-a"
        ranked = scorer.rank_by_trust([a, b], TODAY)
        assert [p.page_id for p in ranked] == ["p-a", "p-b"]


@pytest.mark.unit
class TestFindConflictingTopics:
    def test_recommends_field_notes_over_official_space(self):
        scorer = PageTrustScorer()
        result = scorer.find_conflicting_topics(
            [_official_page(), _field_notes_page()], TODAY
        )
        # Ключевая проверка урока 2.3: победитель определяется по
        # реальному доверию к странице, а не по "официальности" пространства
        assert result == {"ont-activation": "p-field"}

    def test_single_page_topic_not_in_result(self):
        scorer = PageTrustScorer()
        result = scorer.find_conflicting_topics([_single_topic_page()], TODAY)
        assert result == {}

    def test_empty_input_returns_empty_dict(self):
        scorer = PageTrustScorer()
        assert scorer.find_conflicting_topics([], TODAY) == {}


@pytest.mark.integration
class TestPageTrustScorerIntegration:
    """Требует реального коннектора Confluence (REST API, права доступа,
    реальные метаданные ревью) — не выполняется в юнит-тестах курса."""

    def test_score_real_confluence_space(self):
        pytest.skip("Требует настроенного коннектора Confluence Cloud/Server")
