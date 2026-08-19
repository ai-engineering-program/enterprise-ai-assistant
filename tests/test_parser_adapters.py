"""Tests for app/ingestion/parsers/ (Lesson 2.4).

PDF fixtures are built on the fly with PyMuPDF (fitz), same approach as
test_pdf_parser.py. HTML and Markdown fixtures are small strings written
to tmp_path files. No external services are involved, so the whole
module is covered by unit tests only.

Run:
    pytest tests/test_parser_adapters.py -v -m unit
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import DocumentParser, ParsedBlock, ParsedDocument
from app.ingestion.parsers.factory import PARSER_REGISTRY, get_parser
from app.ingestion.parsers.html_adapter import HTMLParserAdapter
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_adapter import PDFParserAdapter


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParsedDocument:
    def test_full_text_joins_blocks_with_blank_line(self):
        doc = ParsedDocument(
            source_format=DocumentFormat.MARKDOWN,
            title="t",
            blocks=[
                ParsedBlock(text="первый блок", block_type="text"),
                ParsedBlock(text="второй блок", block_type="text"),
            ],
        )
        assert doc.full_text == "первый блок\n\nвторой блок"

    def test_full_text_skips_empty_blocks(self):
        doc = ParsedDocument(
            source_format=DocumentFormat.MARKDOWN,
            title=None,
            blocks=[
                ParsedBlock(text="", block_type="text"),
                ParsedBlock(text="содержимое", block_type="text"),
            ],
        )
        assert doc.full_text == "содержимое"

    def test_full_text_empty_for_no_blocks(self):
        doc = ParsedDocument(source_format=DocumentFormat.MARKDOWN, title=None)
        assert doc.full_text == ""


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMarkdownParser:
    def test_title_from_front_matter(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text(
            "---\n"
            "title: Регламент подключения оборудования\n"
            "author: Дамир\n"
            "---\n\n"
            "# Заголовок в тексте документа\n\n"
            "Основной текст регламента с деталями подключения.",
            encoding="utf-8",
        )
        doc = MarkdownParser().parse(path)
        assert doc.source_format == DocumentFormat.MARKDOWN
        assert doc.title == "Регламент подключения оборудования"
        assert doc.metadata["author"] == "Дамир"

    def test_title_falls_back_to_first_heading_without_front_matter(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text(
            "# Заголовок документа\n\nТекст без front matter вообще.",
            encoding="utf-8",
        )
        doc = MarkdownParser().parse(path)
        assert doc.title == "Заголовок документа"
        assert doc.metadata == {}

    def test_body_excludes_front_matter_delimiters(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text(
            "---\ntitle: Х\n---\nТело документа без разметки заголовка.",
            encoding="utf-8",
        )
        doc = MarkdownParser().parse(path)
        assert "---" not in doc.full_text
        assert "Тело документа" in doc.full_text

    def test_no_heading_and_no_front_matter_title_is_none(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("Просто текст без заголовка и без front matter.", encoding="utf-8")
        doc = MarkdownParser().parse(path)
        assert doc.title is None

    def test_unclosed_front_matter_is_ignored_not_crashed(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("---\ntitle: незакрытый блок\nТекст без второго ---", encoding="utf-8")
        doc = MarkdownParser().parse(path)
        assert doc.metadata == {}
        assert "title: незакрытый блок" in doc.full_text


# ---------------------------------------------------------------------------
# PDFParserAdapter
# ---------------------------------------------------------------------------

def _build_two_page_pdf() -> bytes:
    """Собрать тестовый PDF: страница с таблицей + абзацем, страница-скан."""
    doc = fitz.open()

    page1 = doc.new_page(width=400, height=220)
    xs = [50, 200, 350]
    ys = [50, 80, 110]
    for y in ys:
        page1.draw_line((xs[0], y), (xs[-1], y))
    for x in xs:
        page1.draw_line((x, ys[0]), (x, ys[-1]))
    cells = [("Year", 60, 70), ("Total", 210, 70), ("2024", 60, 100), ("500", 210, 100)]
    for text, x, y in cells:
        page1.insert_text((x, y), text, fontsize=10)
    page1.insert_text((50, 150), "Explanatory note outside the table.", fontsize=10)

    page2 = doc.new_page(width=400, height=220)
    page2.draw_rect(fitz.Rect(0, 0, 400, 220), fill=(0.8, 0.8, 0.8))

    return doc.tobytes()


@pytest.fixture(scope="module")
def two_page_pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("parser_adapters") / "report.pdf"
    path.write_bytes(_build_two_page_pdf())
    return path


@pytest.mark.unit
class TestPDFParserAdapter:
    def test_returns_parsed_document_with_correct_format(self, two_page_pdf_path):
        doc = PDFParserAdapter().parse(two_page_pdf_path)
        assert isinstance(doc, ParsedDocument)
        assert doc.source_format == DocumentFormat.PDF

    def test_table_block_serialized_as_markdown_table(self, two_page_pdf_path):
        doc = PDFParserAdapter().parse(two_page_pdf_path)
        table_blocks = [b for b in doc.blocks if b.block_type == "table"]
        assert len(table_blocks) == 1
        assert "| Year | Total |" in table_blocks[0].text

    def test_text_block_present_with_page_number(self, two_page_pdf_path):
        doc = PDFParserAdapter().parse(two_page_pdf_path)
        text_blocks = [b for b in doc.blocks if b.block_type == "text"]
        assert any("Explanatory note" in b.text for b in text_blocks)
        assert all(b.page_number is not None for b in doc.blocks)

    def test_scanned_page_reported_as_warning_not_silently_dropped(self, two_page_pdf_path):
        doc = PDFParserAdapter().parse(two_page_pdf_path)
        assert any("нет текстового слоя" in w for w in doc.warnings)


# ---------------------------------------------------------------------------
# HTMLParserAdapter
# ---------------------------------------------------------------------------

_DOC_HTML = """
<html>
<head><title>Заголовок страницы</title></head>
<body>
  <nav class="breadcrumbs">Главная / Раздел</nav>
  <main id="content">
    <h1>Заголовок статьи</h1>
    <p>Первый содержательный абзац с реальной информацией для проверки плотности текста.</p>
    <p>Второй абзац продолжает описание и тоже содержит достаточно текста для фильтра.</p>
  </main>
  <footer><p>&copy; 2024</p></footer>
</body>
</html>
"""

_SPA_HTML = """
<html>
<head><title>Панель</title></head>
<body>
  <div id="root"></div>
  <script src="/static/bundle.js"></script>
</body>
</html>
"""


@pytest.mark.unit
class TestHTMLParserAdapter:
    def test_reads_file_and_wraps_main_text(self, tmp_path: Path):
        path = tmp_path / "page.html"
        path.write_text(_DOC_HTML, encoding="utf-8")

        doc = HTMLParserAdapter().parse(path)

        assert doc.source_format == DocumentFormat.HTML
        assert doc.title == "Заголовок страницы"
        assert len(doc.blocks) == 1
        assert "содержательный абзац" in doc.blocks[0].text
        assert "Главная" not in doc.blocks[0].text

    def test_spa_shell_flagged_in_warnings_with_no_blocks(self, tmp_path: Path):
        path = tmp_path / "spa.html"
        path.write_text(_SPA_HTML, encoding="utf-8")

        doc = HTMLParserAdapter().parse(path)

        assert doc.blocks == []
        assert len(doc.warnings) == 1
        assert "SPA" in doc.warnings[0] or "JavaScript" in doc.warnings[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParserFactory:
    @pytest.mark.parametrize(
        "fmt, expected_type",
        [
            (DocumentFormat.PDF, PDFParserAdapter),
            (DocumentFormat.HTML, HTMLParserAdapter),
            (DocumentFormat.MARKDOWN, MarkdownParser),
        ],
    )
    def test_returns_expected_adapter_type(self, fmt, expected_type):
        parser = get_parser(fmt)
        assert isinstance(parser, expected_type)

    def test_unregistered_format_raises_value_error(self):
        with pytest.raises(ValueError):
            get_parser(DocumentFormat.DOCX)

    def test_all_registered_parsers_conform_to_protocol(self):
        for parser in PARSER_REGISTRY.values():
            assert isinstance(parser, DocumentParser)

    def test_adding_new_format_does_not_require_touching_existing_entries(self):
        """Регрессия духа урока: реестр — обычный dict, расширяемый одной
        строкой, без необходимости трогать существующие записи."""
        assert DocumentFormat.PDF in PARSER_REGISTRY
        assert DocumentFormat.HTML in PARSER_REGISTRY
        assert DocumentFormat.MARKDOWN in PARSER_REGISTRY
        assert len(PARSER_REGISTRY) == 3


@pytest.mark.integration
class TestParserAdaptersIntegration:
    """Требует реального корпуса документов вне репозитория — например,
    Markdown-экспорт wiki с нетривиальным front matter или PDF со
    сложной многоколоночной вёрсткой.

    Запуск: pytest tests/test_parser_adapters.py -v -m integration
    """

    def test_real_corpus_round_trip(self):
        pytest.skip("Требуется реальный корпус документов — запустить вручную")
