from __future__ import annotations

from app.ingestion.parsers.base import DocumentParser, ParsedBlock, ParsedDocument
from app.ingestion.parsers.factory import PARSER_REGISTRY, get_parser
from app.ingestion.parsers.html_adapter import HTMLParserAdapter
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_adapter import PDFParserAdapter


__all__ = [
    "DocumentParser",
    "ParsedBlock",
    "ParsedDocument",
    "PARSER_REGISTRY",
    "get_parser",
    "HTMLParserAdapter",
    "MarkdownParser",
    "PDFParserAdapter",
]
