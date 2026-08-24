"""Tests for app/ingestion/production_pipeline.py (Lesson 7.4, final project).

All sub-components (chunker, vector_store, normalizer, deduplicator,
pii_sanitizer, quality_report, alert_monitor, delivery_guard,
contract_validator) are injected via the constructor as MagicMock objects —
these tests check ONLY the orchestration logic of ProductionIngestionPipeline
(the order of calls and how results are interpreted), not the correctness of
each component (already covered by its own dedicated test file from earlier
lessons).

Run unit tests only:
    pytest tests/test_production_pipeline.py -v -m unit

Run all tests (requires Qdrant):
    pytest tests/test_production_pipeline.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.data_contract import ContractViolation
from app.ingestion.document_deduplicator import DuplicateMatch
from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument
from app.ingestion.production_pipeline import (
    DocumentOutcome,
    PipelineConfig,
    ProductionIngestionPipeline,
    ProductionRunReport,
)
from app.ingestion.retry_dead_letter import PermanentIngestionError, TransientIngestionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parsed_document(text: str = "текст документа", warnings: list[str] | None = None):
    return ParsedDocument(
        source_format=DocumentFormat.TEXT,
        title=None,
        blocks=[ParsedBlock(text=text, block_type="text")],
        warnings=warnings or [],
    )


def _build_pipeline(contract_validator=None):
    chunker = MagicMock()
    chunker.split.side_effect = lambda text: [text] if text else []

    vector_store = MagicMock()

    normalizer = MagicMock()
    normalizer.normalize.side_effect = lambda text: text

    deduplicator = MagicMock()
    deduplicator.register_document.return_value = DuplicateMatch(
        match_type="unique", similarity=0.0
    )

    pii_sanitizer = MagicMock()
    pii_sanitizer.sanitize.side_effect = lambda text: (text, [])

    quality_report = MagicMock()
    alert_monitor = MagicMock()
    alert_monitor.from_quality_violations.return_value = []
    alert_monitor.from_contract_violations.return_value = []
    alert_monitor.dispatch.side_effect = lambda alerts: alerts

    delivery_guard = MagicMock()

    config = PipelineConfig()

    pipeline = ProductionIngestionPipeline(
        chunker=chunker,
        vector_store=vector_store,
        normalizer=normalizer,
        deduplicator=deduplicator,
        pii_sanitizer=pii_sanitizer,
        quality_report=quality_report,
        alert_monitor=alert_monitor,
        delivery_guard=delivery_guard,
        config=config,
        contract_validator=contract_validator,
    )
    components = {
        "chunker": chunker,
        "vector_store": vector_store,
        "normalizer": normalizer,
        "deduplicator": deduplicator,
        "pii_sanitizer": pii_sanitizer,
        "quality_report": quality_report,
        "alert_monitor": alert_monitor,
        "delivery_guard": delivery_guard,
    }
    return pipeline, components


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.collection_name == "granit_invest_docs"
        assert config.max_retry_attempts == 5
        assert config.contract is None

    def test_custom_values(self):
        config = PipelineConfig(collection_name="siberline_docs", max_retry_attempts=3)
        assert config.collection_name == "siberline_docs"
        assert config.max_retry_attempts == 3


# ---------------------------------------------------------------------------
# process_document: data contract short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessDocumentContract:
    def test_contract_violation_short_circuits_before_parsing(self):
        contract_validator = MagicMock()
        violation = ContractViolation(
            document_id="doc_1",
            field="published_at",
            violation_type="unparseable_datetime",
            message="дата не распознана",
        )
        contract_validator.validate_document.return_value = [violation]
        pipeline, components = _build_pipeline(contract_validator=contract_validator)

        with patch("app.ingestion.production_pipeline.get_parser") as mock_get_parser:
            outcome = pipeline.process_document(
                "doc_1", Path("cert.txt"), {"published_at": "not-a-date"}
            )

        assert outcome.status == "quarantined_contract"
        assert outcome.contract_violations == [violation]
        mock_get_parser.assert_not_called()
        components["normalizer"].normalize.assert_not_called()

    def test_no_violation_proceeds_to_parsing(self):
        contract_validator = MagicMock()
        contract_validator.validate_document.return_value = []
        pipeline, components = _build_pipeline(contract_validator=contract_validator)

        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("немного текста")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.TEXT,
            ),
            patch("app.ingestion.production_pipeline.get_embedding", return_value=[0.1]),
        ):
            outcome = pipeline.process_document("doc_1", Path("cert.txt"), {"published_at": "2024-01-01T00:00:00"})

        assert outcome.status == "indexed"

    def test_no_contract_validator_skips_step(self):
        pipeline, components = _build_pipeline(contract_validator=None)
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("немного текста")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.TEXT,
            ),
            patch("app.ingestion.production_pipeline.get_embedding", return_value=[0.1]),
        ):
            outcome = pipeline.process_document("doc_1", Path("cert.txt"), {})

        assert outcome.status == "indexed"


# ---------------------------------------------------------------------------
# process_document: unsupported format
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessDocumentUnsupportedFormat:
    def test_unregistered_format_raises_permanent_error(self):
        pipeline, _ = _build_pipeline()

        with (
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.DOCX,
            ),
            patch(
                "app.ingestion.production_pipeline.get_parser",
                side_effect=ValueError("нет парсера для docx"),
            ),
        ):
            with pytest.raises(PermanentIngestionError):
                pipeline.process_document("doc_1", Path("act.docx"), {})


# ---------------------------------------------------------------------------
# process_document: OCR routing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessDocumentOCRRouting:
    def test_scanned_pdf_flagged_for_ocr(self):
        pipeline, components = _build_pipeline()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document(
            "", warnings=["страница 1: нет текстового слоя (похоже на скан)"]
        )

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.PDF,
            ),
        ):
            outcome = pipeline.process_document("doc_1", Path("scan.pdf"), {})

        assert outcome.status == "flagged_for_ocr"
        components["normalizer"].normalize.assert_not_called()

    def test_pdf_with_text_layer_proceeds_to_indexing(self):
        pipeline, components = _build_pipeline()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("нормальный текстовый слой")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.PDF,
            ),
            patch("app.ingestion.production_pipeline.get_embedding", return_value=[0.1]),
        ):
            outcome = pipeline.process_document("doc_1", Path("report.pdf"), {})

        assert outcome.status == "indexed"


# ---------------------------------------------------------------------------
# process_document: deduplication
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessDocumentDeduplication:
    def test_duplicate_short_circuits_before_pii_and_chunking(self):
        pipeline, components = _build_pipeline()
        components["deduplicator"].register_document.return_value = DuplicateMatch(
            match_type="exact", similarity=1.0, matched_doc_id="doc_0"
        )
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("повторяющийся текст")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.TEXT,
            ),
        ):
            outcome = pipeline.process_document("doc_1", Path("copy.txt"), {})

        assert outcome.status == "duplicate"
        assert outcome.duplicate_match.match_type == "exact"
        components["pii_sanitizer"].sanitize.assert_not_called()
        components["chunker"].split.assert_not_called()


# ---------------------------------------------------------------------------
# process_document: успешная индексация
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessDocumentIndexing:
    def test_indexed_document_calls_upsert_with_sanitized_text(self):
        pipeline, components = _build_pipeline()
        components["pii_sanitizer"].sanitize.side_effect = lambda text: (
            "текст без email",
            [MagicMock()],
        )
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("текст с email a@b.com")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.TEXT,
            ),
            patch("app.ingestion.production_pipeline.get_embedding", return_value=[0.1, 0.2]),
        ):
            outcome = pipeline.process_document("doc_1", Path("doc.txt"), {})

        assert outcome.status == "indexed"
        assert outcome.chunks_indexed == 1
        assert outcome.pii_masked_count == 1
        components["vector_store"].upsert_documents.assert_called_once()
        docs = components["vector_store"].upsert_documents.call_args.args[0]
        assert docs[0]["metadata"]["text"] == "текст без email"

    def test_empty_chunks_returns_indexed_with_zero_chunks(self):
        pipeline, components = _build_pipeline()
        components["chunker"].split.side_effect = lambda text: []
        mock_parser = MagicMock()
        mock_parser.parse.return_value = _parsed_document("текст")

        with (
            patch("app.ingestion.production_pipeline.get_parser", return_value=mock_parser),
            patch(
                "app.ingestion.production_pipeline.detect_format",
                return_value=DocumentFormat.TEXT,
            ),
        ):
            outcome = pipeline.process_document("doc_1", Path("doc.txt"), {})

        assert outcome.status == "indexed"
        assert outcome.chunks_indexed == 0
        components["vector_store"].upsert_documents.assert_not_called()


# ---------------------------------------------------------------------------
# process_event
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessEvent:
    def test_duplicate_delivery_returns_none_without_processing(self):
        pipeline, components = _build_pipeline()
        components["delivery_guard"].handle_delivery.return_value = "duplicate"

        with patch.object(pipeline, "process_document") as mock_process:
            result = pipeline.process_event({"doc_id": "doc_1"}, Path("doc.txt"))

        assert result is None
        mock_process.assert_not_called()

    def test_applied_delivery_calls_process_document(self):
        pipeline, components = _build_pipeline()
        components["delivery_guard"].handle_delivery.return_value = "applied"
        expected = DocumentOutcome(doc_id="doc_1", status="indexed", chunks_indexed=2)

        with patch.object(pipeline, "process_document", return_value=expected) as mock_process:
            result = pipeline.process_event(
                {"doc_id": "doc_1", "raw_metadata": {"a": 1}}, Path("doc.txt")
            )

        assert result is expected
        mock_process.assert_called_once_with("doc_1", Path("doc.txt"), {"a": 1})


# ---------------------------------------------------------------------------
# run: агрегация батча
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRun:
    def test_aggregates_indexed_documents(self):
        pipeline, components = _build_pipeline()
        components["quality_report"].build_report.return_value = MagicMock()
        components["quality_report"].check_thresholds.return_value = []

        outcomes = [
            DocumentOutcome(doc_id="doc_1", status="indexed", chunks_indexed=3),
            DocumentOutcome(doc_id="doc_2", status="indexed", chunks_indexed=2),
        ]
        with patch.object(pipeline, "process_document", side_effect=outcomes):
            report = pipeline.run(
                [
                    {"doc_id": "doc_1", "path": Path("a.txt"), "raw_metadata": {}},
                    {"doc_id": "doc_2", "path": Path("b.txt"), "raw_metadata": {}},
                ]
            )

        assert isinstance(report, ProductionRunReport)
        assert len(report.ingestion_report.files_processed) == 2
        assert report.ingestion_report.chunks_indexed == 5
        assert report.dead_letter_count == 0
        assert report.contract_violation_count == 0
        assert report.duplicate_count == 0
        assert report.ocr_flagged_count == 0

    def test_permanent_error_moves_document_to_dead_letter(self):
        pipeline, components = _build_pipeline()
        components["quality_report"].build_report.return_value = MagicMock()
        components["quality_report"].check_thresholds.return_value = []

        with patch.object(
            pipeline,
            "process_document",
            side_effect=PermanentIngestionError("формат не поддерживается"),
        ):
            report = pipeline.run(
                [{"doc_id": "doc_bad", "path": Path("bad.xyz"), "raw_metadata": {}}]
            )

        assert report.dead_letter_count == 1
        assert "doc_bad" in report.ingestion_report.files_skipped

    def test_quarantined_and_duplicate_counts(self):
        pipeline, components = _build_pipeline()
        components["quality_report"].build_report.return_value = MagicMock()
        components["quality_report"].check_thresholds.return_value = []

        violation = ContractViolation(
            document_id="doc_1", field="x", violation_type="missing_required", message="нет поля"
        )
        outcomes = [
            DocumentOutcome(
                doc_id="doc_1", status="quarantined_contract", contract_violations=[violation]
            ),
            DocumentOutcome(
                doc_id="doc_2",
                status="duplicate",
                duplicate_match=DuplicateMatch(match_type="exact", similarity=1.0),
            ),
            DocumentOutcome(doc_id="doc_3", status="flagged_for_ocr"),
        ]
        with patch.object(pipeline, "process_document", side_effect=outcomes):
            report = pipeline.run(
                [
                    {"doc_id": "doc_1", "path": Path("a.txt"), "raw_metadata": {}},
                    {"doc_id": "doc_2", "path": Path("b.txt"), "raw_metadata": {}},
                    {"doc_id": "doc_3", "path": Path("c.pdf"), "raw_metadata": {}},
                ]
            )

        assert report.contract_violation_count == 1
        assert report.duplicate_count == 1
        assert report.ocr_flagged_count == 1
        components["alert_monitor"].from_contract_violations.assert_called_once()
        components["alert_monitor"].dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildSignature:
    def test_build_is_classmethod(self):
        assert isinstance(
            ProductionIngestionPipeline.__dict__.get("build"),
            classmethod,
        )

    def test_build_accepts_config(self):
        import inspect

        sig = inspect.signature(ProductionIngestionPipeline.build)
        assert "config" in sig.parameters


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProductionIngestionPipelineIntegration:
    """Требует запущенный Qdrant. Скип по умолчанию: pytest -m 'not integration'."""

    def test_build_and_run_against_real_qdrant(self):
        pytest.skip(
            "Требует запущенный Qdrant (docker run -p 6333:6333 qdrant/qdrant) — "
            "запускайте вручную:\n"
            "  pytest tests/test_production_pipeline.py -m integration -v"
        )
