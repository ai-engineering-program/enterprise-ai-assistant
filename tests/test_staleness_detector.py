import pytest
from datetime import datetime, timedelta

from app.rag.staleness_detector import StalenessDetector, StaleDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector() -> StalenessDetector:
    """Detector with a 1-hour staleness threshold."""
    return StalenessDetector(threshold_seconds=3600)


@pytest.fixture()
def index_updated() -> datetime:
    return datetime(2024, 10, 15, 12, 0, 0)


# ---------------------------------------------------------------------------
# Unit tests — run without any external services
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStalenessDetectorUnit:
    """All tests run without external services (no Qdrant, no Redis, no DB)."""

    def test_fresh_document_not_stale(self, detector: StalenessDetector, index_updated: datetime):
        """Document modified before index update is not stale."""
        doc_modified = index_updated - timedelta(hours=1)
        assert detector.detect_stale(doc_modified, index_updated) is False

    def test_stale_document_detected(self, detector: StalenessDetector, index_updated: datetime):
        """Document modified 2 hours after index update exceeds 1-hour threshold."""
        doc_modified = index_updated + timedelta(hours=2)
        assert detector.detect_stale(doc_modified, index_updated) is True

    def test_boundary_value_not_stale(self, detector: StalenessDetector, index_updated: datetime):
        """Exactly at the threshold (lag == threshold_seconds) is NOT stale."""
        doc_modified = index_updated + timedelta(seconds=3600)
        assert detector.detect_stale(doc_modified, index_updated) is False

    def test_one_second_over_threshold_is_stale(self, detector: StalenessDetector, index_updated: datetime):
        """One second over threshold IS stale."""
        doc_modified = index_updated + timedelta(seconds=3601)
        assert detector.detect_stale(doc_modified, index_updated) is True

    def test_get_staleness_seconds_positive(self, detector: StalenessDetector, index_updated: datetime):
        """Returns positive value when document is modified after index update."""
        doc_modified = index_updated + timedelta(seconds=7200)
        result = detector.get_staleness_seconds(doc_modified, index_updated)
        assert result == pytest.approx(7200.0)

    def test_get_staleness_seconds_negative(self, detector: StalenessDetector, index_updated: datetime):
        """Returns negative value when document was modified before index update."""
        doc_modified = index_updated - timedelta(seconds=1800)
        result = detector.get_staleness_seconds(doc_modified, index_updated)
        assert result == pytest.approx(-1800.0)

    def test_get_stale_documents_filters_correctly(
        self, detector: StalenessDetector, index_updated: datetime
    ):
        """Only documents exceeding the threshold are returned."""
        fresh_time = index_updated - timedelta(hours=1)
        stale_time = index_updated + timedelta(hours=2)

        documents = [
            {"id": "doc_fresh", "last_modified": fresh_time},
            {"id": "doc_stale", "last_modified": stale_time},
        ]
        result = detector.get_stale_documents(documents, index_updated)

        assert len(result) == 1
        assert result[0].doc_id == "doc_stale"

    def test_get_stale_documents_returns_stale_document_type(
        self, detector: StalenessDetector, index_updated: datetime
    ):
        """Returned items are StaleDocument instances with correct fields."""
        stale_time = index_updated + timedelta(hours=3)
        documents = [{"id": "doc_1", "last_modified": stale_time}]

        result = detector.get_stale_documents(documents, index_updated)

        assert len(result) == 1
        stale = result[0]
        assert isinstance(stale, StaleDocument)
        assert stale.doc_id == "doc_1"
        assert stale.last_modified == stale_time
        assert stale.staleness_seconds == pytest.approx(10800.0)

    def test_empty_document_list_returns_empty(
        self, detector: StalenessDetector, index_updated: datetime
    ):
        """Empty input produces empty output."""
        result = detector.get_stale_documents([], index_updated)
        assert result == []

    def test_all_fresh_documents_returns_empty(
        self, detector: StalenessDetector, index_updated: datetime
    ):
        """When all documents are fresh, returns empty list."""
        documents = [
            {"id": f"doc_{i}", "last_modified": index_updated - timedelta(hours=i + 1)}
            for i in range(5)
        ]
        result = detector.get_stale_documents(documents, index_updated)
        assert result == []

    def test_custom_threshold_respected(self, index_updated: datetime):
        """Detector with zero threshold considers any post-update modification stale."""
        strict_detector = StalenessDetector(threshold_seconds=0)
        doc_modified = index_updated + timedelta(seconds=1)
        assert strict_detector.detect_stale(doc_modified, index_updated) is True

    def test_large_threshold_allows_old_documents(self, index_updated: datetime):
        """Detector with very large threshold accepts documents modified well after index."""
        lenient_detector = StalenessDetector(threshold_seconds=86400 * 30)  # 30 days
        doc_modified = index_updated + timedelta(days=29)
        assert lenient_detector.detect_stale(doc_modified, index_updated) is False


# ---------------------------------------------------------------------------
# Integration tests — require external services
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStalenessDetectorIntegration:
    """Tests that require running services. Skip with: pytest -m 'not integration'"""

    def test_placeholder(self):
        pytest.skip(
            "Integration tests for StalenessDetector require a document source "
            "(e.g. Confluence mock or database). Run manually after setting up services."
        )
