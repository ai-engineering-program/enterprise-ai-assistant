from __future__ import annotations

from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.html_adapter import HTMLParserAdapter
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_adapter import PDFParserAdapter


__all__ = ["PARSER_REGISTRY", "get_parser"]


# Единая точка регистрации: формат -> готовый экземпляр парсера,
# реализующий DocumentParser. Добавление нового формата — одна новая
# строка здесь и один новый класс-адаптер, а не новая ветка в теле
# пайплайна поглощения (см. урок 2.4, историю про пятый формат).
PARSER_REGISTRY: dict[DocumentFormat, DocumentParser] = {
    DocumentFormat.PDF: PDFParserAdapter(),
    DocumentFormat.HTML: HTMLParserAdapter(),
    DocumentFormat.MARKDOWN: MarkdownParser(),
}


def get_parser(fmt: DocumentFormat) -> DocumentParser:
    """Вернуть парсер для формата или явно поднять ошибку, если его нет.

    TODO:
    1. Если fmt есть в PARSER_REGISTRY — вернуть PARSER_REGISTRY[fmt].
    2. Иначе — поднять ValueError(
           f"Нет зарегистрированного парсера для формата {fmt.value} — "
           "см. STRATEGY_REGISTRY в app/ingestion/format_detector.py"
       )
       (явная ошибка вместо молчаливого None или пропуска файла — тот
       же принцип "явный карантин", что и в format_detector.py, урок 2.1)
    """
    ...
