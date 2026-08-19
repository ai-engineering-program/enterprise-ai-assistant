from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


__all__ = [
    "DocumentFormat",
    "ParsingStrategy",
    "STRATEGY_REGISTRY",
    "HEADER_READ_SIZE",
    "detect_format_from_bytes",
    "detect_format",
    "select_parsing_strategy",
    "looks_like_scanned_pdf",
]


class DocumentFormat(str, Enum):
    """Формат документа, определённый по содержимому файла, а не по расширению."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    UNKNOWN = "unknown"


# Магические байты — сигнатуры начала файла, по которым формат определяется
# независимо от расширения. Расширению нельзя доверять как единственному
# источнику истины о формате — см. урок 2.1.
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"  # DOCX/XLSX/PPTX — все это ZIP-архивы (OOXML)
_HTML_MARKERS = (b"<!doctype html", b"<html")

# Операторы вывода текста в PDF content stream. Если ни один из них не
# встречается в файле, страница почти наверняка не содержит текстового
# слоя (упрощённая эвристика — полноценный разбор content stream придёт
# в уроке 2.2 вместе с pdfplumber).
_PDF_TEXT_OPERATORS = (b" Tj", b" TJ")

HEADER_READ_SIZE = 2048
"""Сколько байт достаточно прочитать с начала файла, чтобы определить
формат по сигнатуре — читать весь файл целиком для этого не нужно."""


@dataclass(frozen=True)
class ParsingStrategy:
    """Описывает, как пайплайн поглощения должен обрабатывать формат."""

    format: DocumentFormat
    implemented: bool
    parser_module: str
    notes: str


# Реестр стратегий парсинга по формату. Часть стратегий пока не реализована —
# implemented=False честно фиксирует это в самом реестре, а не молчаливым
# пропуском файла, как было в SUPPORTED_EXTENSIONS пайплайна v1 (урок 1.4).
# Уроки 2.2-2.4 заполняют оставшиеся стратегии реальными парсерами.
STRATEGY_REGISTRY: dict[DocumentFormat, ParsingStrategy] = {
    DocumentFormat.TEXT: ParsingStrategy(
        format=DocumentFormat.TEXT,
        implemented=True,
        parser_module="app.ingestion.pipeline",
        notes="Простое чтение UTF-8 — реализовано в уроке 1.4.",
    ),
    DocumentFormat.MARKDOWN: ParsingStrategy(
        format=DocumentFormat.MARKDOWN,
        implemented=True,
        parser_module="app.ingestion.document_chunker",
        notes="Структурный разбор Markdown — DocumentAwareChunker.",
    ),
    DocumentFormat.PDF: ParsingStrategy(
        format=DocumentFormat.PDF,
        implemented=False,
        parser_module="app.ingestion.pdf_parser",
        notes="Появится в уроке 2.2 — текстовые слои, таблицы, layout.",
    ),
    DocumentFormat.DOCX: ParsingStrategy(
        format=DocumentFormat.DOCX,
        implemented=False,
        parser_module="app.ingestion.docx_parser",
        notes="Появится в уроке 2.4 в составе единого адаптера парсинга.",
    ),
    DocumentFormat.HTML: ParsingStrategy(
        format=DocumentFormat.HTML,
        implemented=False,
        parser_module="app.ingestion.html_parser",
        notes="Появится в уроке 2.3 — борьба с шумом разметки.",
    ),
    DocumentFormat.UNKNOWN: ParsingStrategy(
        format=DocumentFormat.UNKNOWN,
        implemented=False,
        parser_module="",
        notes="Нет стратегии — файл должен попасть в явный карантин, а не быть молча пропущен.",
    ),
}


def detect_format_from_bytes(header: bytes) -> DocumentFormat:
    """Определить формат по первым байтам файла (магические числа/сигнатуры).

    TODO:
    1. Если header начинается с _PDF_MAGIC -> DocumentFormat.PDF
    2. Если header начинается с _ZIP_MAGIC -> DocumentFormat.DOCX
       (упрощение курса: любой OOXML zip считаем DOCX; различать
       xlsx/pptx по внутреннему манифесту — вне рамок урока)
    3. Привести header к нижнему регистру после lstrip() и проверить,
       начинается ли он с одного из _HTML_MARKERS -> DocumentFormat.HTML
    4. Если ничего не подошло — вернуть DocumentFormat.UNKNOWN
       (у текста и Markdown нет магической сигнатуры — их уточняет
       detect_format() через расширение)
    """
    ...


def detect_format(path: Path) -> DocumentFormat:
    """Определить формат документа по содержимому, с расширением как fallback.

    TODO:
    1. Прочитать первые HEADER_READ_SIZE байт файла в бинарном режиме
    2. fmt = detect_format_from_bytes(header)
    3. Если fmt != DocumentFormat.UNKNOWN -> вернуть fmt
       (сигнатура важнее расширения — файл может быть переименован
       или иметь неверное расширение, см. урок 2.1)
    4. Если fmt == UNKNOWN: посмотреть на path.suffix.lower():
       - ".md"  -> DocumentFormat.MARKDOWN
       - ".txt" -> DocumentFormat.TEXT
       - иначе  -> DocumentFormat.UNKNOWN
    5. Вернуть результат.
    """
    ...


def select_parsing_strategy(fmt: DocumentFormat) -> ParsingStrategy:
    """Вернуть стратегию парсинга для формата.

    TODO: return STRATEGY_REGISTRY.get(fmt, STRATEGY_REGISTRY[DocumentFormat.UNKNOWN])
    """
    ...


def looks_like_scanned_pdf(raw_bytes: bytes) -> bool:
    """Эвристически определить, что PDF не содержит текстового слоя.

    Настоящий разбор текстовых объектов PDF — тема урока 2.2 (pdfplumber).
    Здесь — дешёвая эвристика на уровне сырых байтов, которая уже помогает
    поймать самый частый случай ловушки «PDF без текстового слоя»: скан,
    обёрнутый в PDF без единого оператора вывода текста.

    TODO:
    1. Если raw_bytes не начинается с _PDF_MAGIC -> поднять
       ValueError("не похоже на PDF")
    2. Проверить наличие в raw_bytes хотя бы одного из _PDF_TEXT_OPERATORS
    3. Если ни один оператор не встретился -> вернуть True
       (похоже на скан без текстового слоя)
    4. Иначе -> вернуть False
    """
    ...
