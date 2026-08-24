"""Tests for app/ingestion/data_quality_report.py (Lesson 7.1).

DataQualityReport only aggregates results already produced by other
ingestion components (IngestionReport, DocumentDeduplicator matches,
OCR diagnostics flags, freshness lag samples) — these unit tests build
those inputs directly as plain dataclass instances, without running
any real parsing, OCR or embedding pipeline.

Run unit tests only:
    pytest tests/test_data_quality_report.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.data_quality_report import (
    DataQualityMetrics,
    DataQualityReport,
    DataQualityThresholds,
)
from app.ingestion.document_deduplicator import DuplicateMatch
from app.ingestion.pipeline import IngestionReport


# ---------------------------------------------------------------------------
# compute_completeness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeCompleteness:
    def test_all_processed_no_dead_letter(self):
        report = DataQualityReport()
        ingestion_report = IngestionReport(
            files_processed=["a.txt", "b.txt"], files_skipped=[], chunks_indexed=10
        )
        assert report.compute_completeness(ingestion_report) == 1.0

    def test_matches_ingestion_report_and_dead_letter_split(self):
        report = DataQualityReport()
        ingestion_report = IngestionReport(
            files_processed=["a.txt"] * 39800, files_skipped=[], chunks_indexed=1000
        )
        completeness = report.compute_completeness(ingestion_report, dead_letter_count=200)
        assert completeness == pytest.approx(39800 / 40000, abs=1e-6)

    def test_files_skipped_and_dead_letter_both_reduce_completeness(self):
        report = DataQualityReport()
        ingestion_report = IngestionReport(
            files_processed=["a.txt"] * 90,
            files_skipped=["b.txt"] * 5,
            chunks_indexed=500,
        )
        completeness = report.compute_completeness(ingestion_report, dead_letter_count=5)
        assert completeness == pytest.approx(90 / 100, abs=1e-6)

    def test_no_attempts_returns_one(self):
        report = DataQualityReport()
        ingestion_report = IngestionReport(files_processed=[], files_skipped=[], chunks_indexed=0)
        assert report.compute_completeness(ingestion_report, dead_letter_count=0) == 1.0


# ---------------------------------------------------------------------------
# compute_validity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeValidity:
    def test_no_invalid_documents(self):
        report = DataQualityReport()
        assert report.compute_validity(total_documents=100, invalid_document_ids=set()) == 1.0

    def test_some_invalid_documents(self):
        report = DataQualityReport()
        validity = report.compute_validity(
            total_documents=100, invalid_document_ids={"doc1", "doc2", "doc3"}
        )
        assert validity == pytest.approx(0.97, abs=1e-6)

    def test_zero_total_documents_returns_one(self):
        report = DataQualityReport()
        assert report.compute_validity(total_documents=0, invalid_document_ids=set()) == 1.0

    def test_clamped_to_zero_when_invalid_exceeds_total(self):
        report = DataQualityReport()
        validity = report.compute_validity(
            total_documents=2, invalid_document_ids={"a", "b", "c", "d"}
        )
        assert validity == 0.0


# ---------------------------------------------------------------------------
# compute_freshness / _percentile
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeFreshness:
    def test_empty_list_returns_zero_zero(self):
        report = DataQualityReport()
        assert report.compute_freshness([]) == (0.0, 0.0)

    def test_percentiles_on_known_distribution(self):
        report = DataQualityReport()
        lags = [10.0, 20.0, 30.0, 40.0, 100.0]
        p50, p95 = report.compute_freshness(lags)
        assert p50 == pytest.approx(30.0)
        assert p95 == pytest.approx(100.0)

    def test_p95_reflects_slow_tail_not_masked_by_average(self):
        report = DataQualityReport()
        # 19 документов синхронизируются за минуту, один — за две недели.
        fast = [60.0] * 19
        slow = [14 * 24 * 3600.0]
        lags = fast + slow
        p50, p95 = report.compute_freshness(lags)
        mean_lag = sum(lags) / len(lags)
        assert p50 == pytest.approx(60.0)
        # p95 должен явно показать медленный хвост, который среднее прячет.
        assert p95 > mean_lag
        assert p95 == pytest.approx(14 * 24 * 3600.0)


# ---------------------------------------------------------------------------
# compute_duplication_rate
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeDuplicationRate:
    def test_no_matches_returns_zero(self):
        report = DataQualityReport()
        assert report.compute_duplication_rate([]) == 0.0

    def test_all_unique_returns_zero(self):
        report = DataQualityReport()
        matches = [DuplicateMatch(match_type="unique", similarity=0.0) for _ in range(5)]
        assert report.compute_duplication_rate(matches) == 0.0

    def test_mixed_matches(self):
        report = DataQualityReport()
        matches = [
            DuplicateMatch(match_type="unique", similarity=0.0),
            DuplicateMatch(match_type="exact", similarity=1.0, matched_doc_id="d1"),
            DuplicateMatch(match_type="near_duplicate", similarity=0.9, matched_doc_id="d2"),
            DuplicateMatch(match_type="unique", similarity=0.0),
        ]
        assert report.compute_duplication_rate(matches) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildReport:
    def test_aggregates_all_four_metrics(self):
        report = DataQualityReport()
        ingestion_report = IngestionReport(
            files_processed=["a.txt"] * 398, files_skipped=[], chunks_indexed=2000
        )
        duplicate_matches = [
            DuplicateMatch(match_type="unique", similarity=0.0),
            DuplicateMatch(match_type="near_duplicate", similarity=0.9, matched_doc_id="d1"),
        ]

        metrics = report.build_report(
            ingestion_report=ingestion_report,
            total_documents=400,
            invalid_document_ids={"docA"},
            freshness_lag_seconds=[60.0, 120.0, 900.0],
            duplicate_matches=duplicate_matches,
            dead_letter_count=2,
        )

        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.total_documents == 400
        assert metrics.completeness == pytest.approx(398 / 400)
        assert metrics.validity == pytest.approx(1 - 1 / 400)
        assert metrics.duplication_rate == pytest.approx(0.5)
        assert metrics.freshness_p50_seconds > 0
        assert metrics.freshness_p95_seconds >= metrics.freshness_p50_seconds


# ---------------------------------------------------------------------------
# check_thresholds
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckThresholds:
    def test_all_metrics_within_thresholds_returns_empty_list(self):
        report = DataQualityReport(DataQualityThresholds(
            min_completeness=0.9,
            min_validity=0.9,
            max_freshness_p95_seconds=1000.0,
            max_duplication_rate=0.5,
        ))
        metrics = DataQualityMetrics(
            total_documents=100,
            completeness=0.99,
            validity=0.97,
            freshness_p50_seconds=10.0,
            freshness_p95_seconds=100.0,
            duplication_rate=0.02,
        )
        assert report.check_thresholds(metrics) == []

    def test_completeness_violation_detected(self):
        report = DataQualityReport(DataQualityThresholds(min_completeness=0.98))
        metrics = DataQualityMetrics(
            total_documents=100,
            completeness=0.96,
            validity=1.0,
            freshness_p50_seconds=10.0,
            freshness_p95_seconds=10.0,
            duplication_rate=0.0,
        )
        violations = report.check_thresholds(metrics)
        assert len(violations) == 1
        assert violations[0].metric == "completeness"

    def test_freshness_violation_uses_p95_not_p50(self):
        report = DataQualityReport(DataQualityThresholds(max_freshness_p95_seconds=3600.0))
        metrics = DataQualityMetrics(
            total_documents=100,
            completeness=1.0,
            validity=1.0,
            freshness_p50_seconds=60.0,
            freshness_p95_seconds=14 * 24 * 3600.0,
            duplication_rate=0.0,
        )
        violations = report.check_thresholds(metrics)
        assert any(v.metric == "freshness_p95_seconds" for v in violations)

    def test_multiple_violations_all_reported(self):
        report = DataQualityReport(DataQualityThresholds(
            min_completeness=0.98,
            min_validity=0.95,
            max_freshness_p95_seconds=3600.0,
            max_duplication_rate=0.05,
        ))
        metrics = DataQualityMetrics(
            total_documents=100,
            completeness=0.80,
            validity=0.70,
            freshness_p50_seconds=100.0,
            freshness_p95_seconds=100000.0,
            duplication_rate=0.30,
        )
        violations = report.check_thresholds(metrics)
        flagged_metrics = {v.metric for v in violations}
        assert flagged_metrics == {
            "completeness", "validity", "freshness_p95_seconds", "duplication_rate"
        }


# ---------------------------------------------------------------------------
# Integration — requires a real ingestion run against Qdrant / OCR
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDataQualityReportIntegration:
    """Требует реального прогона AsyncIngestionPipeline с настоящими
    OCR-диагностикой, дедупликацией и Qdrant. Скип по умолчанию:
    pytest -m 'not integration'."""

    def test_build_report_against_real_pipeline_run(self):
        pytest.skip(
            "Требует запущенный Qdrant и реальный прогон пайплайна "
            "поглощения — запускайте вручную:\n"
            "  pytest tests/test_data_quality_report.py -m integration -v"
        )
