from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.ingestion.format_detector import DocumentFormat


__all__ = ["ParsedBlock", "ParsedDocument", "DocumentParser"]


@dataclass
class ParsedBlock:
    """Один нормализованный фрагмент содержимого документа.

    block_type — общий словарь типов поверх разных форматов ("text",
    "table", ...). page_number заполняется только там, где у формата
    есть понятие страницы (PDF); для HTML и Markdown остаётся None.
    """

    text: str
    block_type: str
    page_number: int | None = None


@dataclass
class ParsedDocument:
    """Единый результат разбора документа, общий для всех форматов.

    Родные парсеры (PDFParser, HTMLParser, ...) возвращают богатые,
    специфичные для формата структуры (PDFParseResult, HTMLParseResult).
    Адаптеры в этом пакете переводят их в этот общий вид — ту часть,
    которая нужна пайплайну поглощения дальше по цепочке (чанкинг,
    эмбеддинг, индексация). Детали формата, не нужные дальше по
    конвейеру напрямую, оседают в warnings/metadata, а не теряются
    молча — см. урок 2.4.
    """

    source_format: DocumentFormat
    title: str | None
    blocks: list[ParsedBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Склеить текст непустых блоков в одну строку.

        TODO: вернуть "\\n\\n".join(block.text for block in self.blocks
        if block.text) — пустые блоки (block.text == "") пропускаются.
        """
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Единый интерфейс парсинга: один метод, один контракт вызова.

    Любая реализация принимает путь к файлу и возвращает ParsedDocument
    — независимо от того, что происходит внутри (pdfplumber,
    BeautifulSoup, ручной разбор текста). Это структурная типизация:
    класс не обязан наследоваться от DocumentParser явно, достаточно
    иметь метод с такой сигнатурой (см. PDFParserAdapter, HTMLParserAdapter,
    MarkdownParser — ни один из них не пишет class X(DocumentParser)).
    """

    def parse(self, path: Path) -> ParsedDocument: ...
