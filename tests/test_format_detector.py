"""Tests for app/ingestion/format_detector.py (Lesson 2.1).

Everything here operates on raw bytes and local temporary files — no
external services are involved, so the whole module is covered by
unit tests only.

Run:
    pytest tests/test_format_detector.py -v -m unit
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.format_detector import (
    DocumentFormat,
    ParsingStrategy,
    STRATEGY_REGISTRY,
    detect_format_from_bytes,
    detect_format,
    select_parsing_strategy,
    looks_like_scanned_pdf,
)


# ---------------------------------------------------------------------------
# detect_format_from_bytes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectFormatFromBytes:
    def test_pdf_magic(self):
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj"
        assert detect_format_from_bytes(header) == DocumentFormat.PDF

    def test_docx_zip_magic(self):
        header = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"
        assert detect_format_from_bytes(header) == DocumentFormat.DOCX

    def test_html_doctype_lowercase(self):
        header = b"<!doctype html>\n<html><head></head></html>"
        assert detect_format_from_bytes(header) == DocumentFormat.HTML

    def test_html_doctype_uppercase(self):
        header = b"<!DOCTYPE HTML>\n<HTML></HTML>"
        assert detect_format_from_bytes(header) == DocumentFormat.HTML

    def test_html_leading_whitespace(self):
        header = b"   \n<html><body>текст</body></html>"
        assert detect_format_from_bytes(header) == DocumentFormat.HTML

    def test_plain_text_has_no_signature(self):
        header = "Обычный текстовый файл без магической сигнатуры".encode("utf-8")
        assert detect_format_from_bytes(header) == DocumentFormat.UNKNOWN

    def test_empty_header(self):
        assert detect_format_from_bytes(b"") == DocumentFormat.UNKNOWN


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectFormat:
    def test_txt_file_by_extension(self, tmp_path: Path):
        path = tmp_path / "policy.txt"
        path.write_text("обычный регламент", encoding="utf-8")
        assert detect_format(path) == DocumentFormat.TEXT

    def test_md_file_by_extension(self, tmp_path: Path):
        path = tmp_path / "readme.md"
        path.write_text("# Заголовок\n\nТекст", encoding="utf-8")
        assert detect_format(path) == DocumentFormat.MARKDOWN

    def test_pdf_signature_wins_over_wrong_extension(self, tmp_path: Path):
        """Файл с содержимым PDF, но расширением .txt — сигнатура важнее
        расширения (см. ловушку 'расширение лжёт' из урока 2.1)."""
        path = tmp_path / "mislabeled.txt"
        path.write_bytes(b"%PDF-1.4\n%fake pdf content for test\n")
        assert detect_format(path) == DocumentFormat.PDF

    def test_docx_signature_detected_regardless_of_extension(self, tmp_path: Path):
        path = tmp_path / "report.docx"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 20)
        assert detect_format(path) == DocumentFormat.DOCX

    def test_html_exported_with_txt_extension(self, tmp_path: Path):
        path = tmp_path / "exported_page.txt"
        path.write_bytes(b"<!doctype html><html><body>content</body></html>")
        assert detect_format(path) == DocumentFormat.HTML

    def test_unknown_extension_and_no_signature(self, tmp_path: Path):
        path = tmp_path / "notes.bin"
        path.write_bytes(b"\x01\x02\x03random binary garbage\x04\x05")
        assert detect_format(path) == DocumentFormat.UNKNOWN


# ---------------------------------------------------------------------------
# select_parsing_strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSelectParsingStrategy:
    @pytest.mark.parametrize(
        "fmt",
        [DocumentFormat.TEXT, DocumentFormat.MARKDOWN],
    )
    def test_implemented_formats(self, fmt):
        strategy = select_parsing_strategy(fmt)
        assert isinstance(strategy, ParsingStrategy)
        assert strategy.implemented is True

    @pytest.mark.parametrize(
        "fmt",
        [DocumentFormat.PDF, DocumentFormat.DOCX, DocumentFormat.HTML],
    )
    def test_not_yet_implemented_formats(self, fmt):
        strategy = select_parsing_strategy(fmt)
        assert strategy.implemented is False
        assert strategy.notes  # обязательно объясняет, когда появится

    def test_unknown_format_has_no_strategy(self):
        strategy = select_parsing_strategy(DocumentFormat.UNKNOWN)
        assert strategy.implemented is False

    def test_registry_covers_every_format(self):
        for fmt in DocumentFormat:
            assert fmt in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# looks_like_scanned_pdf
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLooksLikeScannedPdf:
    def test_pdf_without_text_operators_looks_scanned(self):
        raw = b"%PDF-1.4\n1 0 obj << /Type /XObject /Subtype /Image >>\nstream\n\xff\xd8\xff\nendstream"
        assert looks_like_scanned_pdf(raw) is True

    def test_pdf_with_tj_operator_is_not_scanned(self):
        raw = b"%PDF-1.4\nBT /F1 12 Tf (Hello world) Tj ET\n"
        assert looks_like_scanned_pdf(raw) is False

    def test_pdf_with_tj_array_operator_is_not_scanned(self):
        raw = b"%PDF-1.4\nBT /F1 12 Tf [(Hello) -250 (world)] TJ ET\n"
        assert looks_like_scanned_pdf(raw) is False

    def test_raises_on_non_pdf_input(self):
        with pytest.raises(ValueError):
            looks_like_scanned_pdf(b"PK\x03\x04not a pdf at all")
