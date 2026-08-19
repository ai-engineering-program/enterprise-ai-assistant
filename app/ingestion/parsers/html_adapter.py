from __future__ import annotations

from pathlib import Path

from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument
from app.ingestion.html_parser import HTMLParser


__all__ = ["HTMLParserAdapter"]


class HTMLParserAdapter:
    """Адаптер, приводящий HTMLParser (урок 2.3) к единому интерфейсу DocumentParser.

    Ключевое отличие от PDFParserAdapter: HTMLParser.parse() принимает
    уже прочитанную строку с HTML, а не путь к файлу — именно это
    несовпадение контрактов вызова между PDFParser и HTMLParser стало
    причиной инцидента из урока 2.4. Адаптер прячет разницу внутри
    себя: наружу торчит один и тот же parse(path).
    """

    def __init__(self, html_parser: HTMLParser | None = None) -> None:
        # TODO: сохранить self._html_parser = html_parser or HTMLParser()
        ...

    def parse(self, path: Path) -> ParsedDocument:
        """Прочитать HTML-файл и вернуть его как ParsedDocument.

        TODO:
        1. html = path.read_text(encoding="utf-8")
        2. result = self._html_parser.parse(html)
        3. blocks: если result.main_text непустой — список из одного
           ParsedBlock(text=result.main_text, block_type="text");
           иначе — пустой список
        4. warnings: если result.likely_js_rendered is True — добавить
           строку вида "страница похожа на SPA-шелл без содержимого до
           выполнения JavaScript — нужен рендеринг headless-браузером"
        5. Вернуть ParsedDocument(source_format=DocumentFormat.HTML,
                                    title=result.title, blocks=blocks,
                                    warnings=warnings)
        """
        ...
