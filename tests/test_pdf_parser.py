"""Tests for app/ingestion/pdf_parser.py (Lesson 2.2).

Fixture PDFs are generated on the fly with PyMuPDF (fitz) — real,
well-formed PDF binaries with genuine content-stream text and vector-drawn
table gridlines, not hand-crafted byte fragments. Fixtures use ASCII-only
text on purpose: this keeps the test suite portable across operating
systems without bundling a Cyrillic-capable font file into the repo.

Run:
    pytest tests/test_pdf_parser.py -v -m unit
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.ingestion.pdf_parser import PDFElement, PDFParseResult, PDFParser


def _build_report_pdf() -> bytes:
    """Собрать тестовый PDF из трёх страниц:

    1. таблица показателей (сетка из линий) + пояснительный абзац вне таблицы
    2. только связный текст, без единой таблицы на странице
    3. страница без текстового слоя (закрашенный прямоугольник — имитация
       отсканированной страницы без операторов Tj/TJ)
    """
    doc = fitz.open()

    page1 = doc.new_page(width=400, height=220)
    xs = [50, 200, 350]
    ys = [50, 80, 110, 140]
    for y in ys:
        page1.draw_line((xs[0], y), (xs[-1], y))
    for x in xs:
        page1.draw_line((x, ys[0]), (x, ys[-1]))
    cells = [
        ("Year", 60, 70), ("Net_profit", 210, 70),
        ("2023", 60, 100), ("1200000", 210, 100),
        ("2024", 60, 130), ("1450000", 210, 130),
    ]
    for text, x, y in cells:
        page1.insert_text((x, y), text, fontsize=10)
    page1.insert_text((50, 170), "Notes: figures in thousands of RUB.", fontsize=10)

    page2 = doc.new_page(width=400, height=220)
    page2.insert_text((50, 60), "This page has plain narrative text only.", fontsize=10)
    page2.insert_text((50, 80), "No tables appear on this page at all.", fontsize=10)

    page3 = doc.new_page(width=400, height=220)
    page3.draw_rect(fitz.Rect(0, 0, 400, 220), fill=(0.8, 0.8, 0.8))

    return doc.tobytes()


@pytest.fixture(scope="module")
def report_pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf_parser") / "report.pdf"
    path.write_bytes(_build_report_pdf())
    return path


@pytest.mark.unit
class TestPDFParserTables:
    def test_table_extracted_as_structured_rows_not_flat_text(self, report_pdf_path):
        result = PDFParser().parse(report_pdf_path)
        tables = [e for e in result.elements if e.element_type == "table"]
        assert len(tables) == 1
        rows = tables[0].rows
        assert rows[0] == ["Year", "Net_profit"]
        assert rows[1] == ["2023", "1200000"]
        assert rows[2] == ["2024", "1450000"]

    def test_table_element_has_page_number_and_bbox(self, report_pdf_path):
        result = PDFParser().parse(report_pdf_path)
        table = next(e for e in result.elements if e.element_type == "table")
        assert table.page_number == 1
        assert table.bbox is not None

    def test_surrounding_text_does_not_duplicate_table_values(self, report_pdf_path):
        """Регрессия для инцидента из урока: значения ячеек таблицы не
        должны просочиться в текстовый блок той же страницы."""
        result = PDFParser().parse(report_pdf_path)
        text_blocks_page1 = [
            e for e in result.elements
            if e.element_type == "text_block" and e.page_number == 1
        ]
        assert len(text_blocks_page1) == 1
        assert "1200000" not in text_blocks_page1[0].content
        assert "1450000" not in text_blocks_page1[0].content
        assert "Notes" in text_blocks_page1[0].content


@pytest.mark.unit
class TestPDFParserTextOnlyPages:
    def test_page_without_table_yields_single_text_block(self, report_pdf_path):
        result = PDFParser().parse(report_pdf_path)
        text_blocks_page2 = [
            e for e in result.elements
            if e.element_type == "text_block" and e.page_number == 2
        ]
        assert len(text_blocks_page2) == 1
        assert "plain narrative text" in text_blocks_page2[0].content

        tables_page2 = [
            e for e in result.elements
            if e.element_type == "table" and e.page_number == 2
        ]
        assert tables_page2 == []


@pytest.mark.unit
class TestPDFParserScannedPages:
    def test_page_without_text_layer_is_flagged_not_silently_dropped(self, report_pdf_path):
        result = PDFParser().parse(report_pdf_path)
        assert 3 in result.pages_without_text_layer
        assert all(e.page_number != 3 for e in result.elements)


@pytest.mark.unit
class TestTableToMarkdown:
    def test_serializes_rows_with_header_and_separator(self):
        element = PDFElement(
            element_type="table",
            page_number=1,
            rows=[["Year", "Net_profit"], ["2023", "1200000"], ["2024", "1450000"]],
        )
        markdown = PDFParser().table_to_markdown(element)
        lines = markdown.split("\n")
        assert lines[0] == "| Year | Net_profit |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 2023 | 1200000 |"
        assert lines[3] == "| 2024 | 1450000 |"

    def test_none_cells_become_empty_strings(self):
        element = PDFElement(
            element_type="table",
            page_number=1,
            rows=[["Year", "Note"], ["2023", None]],
        )
        markdown = PDFParser().table_to_markdown(element)
        assert "| 2023 |  |" in markdown

    def test_raises_on_non_table_element(self):
        element = PDFElement(element_type="text_block", page_number=1, content="x")
        with pytest.raises(ValueError):
            PDFParser().table_to_markdown(element)

    def test_raises_on_empty_rows(self):
        element = PDFElement(element_type="table", page_number=1, rows=[])
        with pytest.raises(ValueError):
            PDFParser().table_to_markdown(element)


@pytest.mark.integration
class TestPDFParserIntegration:
    """Требует реального PDF из production-корпуса (вне репозитория) —
    например, отчёт с многоколоночной вёрсткой или сложной вложенной
    таблицей, где нужно вручную проверить качество извлечения.

    Запуск: pytest tests/test_pdf_parser.py -v -m integration
    """

    def test_real_financial_report_table_matches_expected_values(self):
        pytest.skip("Требуется реальный файл из корпуса — запустить вручную")
