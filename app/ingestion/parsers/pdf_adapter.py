from __future__ import annotations

from pathlib import Path

from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument
from app.ingestion.pdf_parser import PDFParser


__all__ = ["PDFParserAdapter"]


class PDFParserAdapter:
    """Адаптер, приводящий PDFParser (урок 2.2) к единому интерфейсу DocumentParser.

    PDFParser.parse() возвращает PDFParseResult — список PDFElement с
    element_type "text_block"/"table" и отдельный список номеров страниц
    без текстового слоя. Здесь эта структура переводится в список
    ParsedBlock и warnings. PDFParser не меняется ни на одну строку —
    адаптер оборачивает существующий код, а не переписывает его.
    """

    def __init__(self, pdf_parser: PDFParser | None = None) -> None:
        # TODO: сохранить self._pdf_parser = pdf_parser or PDFParser()
        ...

    def parse(self, path: Path) -> ParsedDocument:
        """Разобрать PDF-файл и вернуть его как ParsedDocument.

        TODO:
        1. result = self._pdf_parser.parse(path)
        2. Собрать blocks: list[ParsedBlock] — пройти по result.elements:
           - если element.element_type == "table":
             text = self._pdf_parser.table_to_markdown(element)
             добавить ParsedBlock(text=text, block_type="table",
                                    page_number=element.page_number)
           - иначе (element_type == "text_block"):
             добавить ParsedBlock(text=element.content or "",
                                    block_type="text",
                                    page_number=element.page_number)
        3. Собрать warnings: list[str] — для каждого номера страницы p в
           result.pages_without_text_layer добавить строку вида
           f"страница {p}: нет текстового слоя (похоже на скан)"
        4. Вернуть ParsedDocument(source_format=DocumentFormat.PDF,
                                    title=None, blocks=blocks,
                                    warnings=warnings)
           (у PDF в рамках этого курса нет понятия заголовка документа
           на уровне метаданных файла — title остаётся None)
        """
        ...
