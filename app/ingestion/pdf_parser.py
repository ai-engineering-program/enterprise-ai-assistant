from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


__all__ = ["PDFElement", "PDFParseResult", "PDFParser"]


@dataclass
class PDFElement:
    """Один структурный элемент, извлечённый со страницы PDF.

    element_type == "text_block": content заполнен, rows == None.
    element_type == "table": rows заполнен списком строк (каждая строка —
    список значений ячеек, включая возможные None для пустых ячеек),
    content == None. Таблица НЕ превращается в плоский текст на этом
    уровне — см. PDFParser.table_to_markdown().
    """

    element_type: str
    page_number: int
    content: str | None = None
    rows: list[list[str | None]] | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class PDFParseResult:
    """Результат разбора одного PDF-документа."""

    elements: list[PDFElement] = field(default_factory=list)
    pages_without_text_layer: list[int] = field(default_factory=list)


class PDFParser:
    """Парсер PDF, сохраняющий текстовые слои и таблицы как раздельную структуру.

    Использует pdfplumber: extract_text() для текста, find_tables() —
    для обнаружения таблиц по линиям разметки страницы (а не по
    эвристике «числа через пробел»). Таблица никогда не сливается с
    окружающим текстом: перед извлечением оставшегося текста страницы
    область каждой найденной таблицы вырезается через
    page.outside_bbox(), чтобы значения ячеек не задваивались между
    table-элементом и text_block-элементом одной и той же страницы —
    именно это задваивание/перепутывание разобрано в истории урока.

    Страницы без текстового слоя (сканы без операторов Tj/TJ — см.
    looks_like_scanned_pdf() в app/ingestion/format_detector.py, урок
    2.1) не обрабатываются здесь как ошибка: они явно фиксируются в
    PDFParseResult.pages_without_text_layer и передаются дальше в
    OCR-конвейер раздела 3 курса, а не тихо пропускаются.
    """

    def __init__(self, table_settings: dict | None = None) -> None:
        # TODO: сохранить self.table_settings = table_settings or {}
        # (table_settings передаётся напрямую в page.find_tables(); пустой
        # словарь означает "использовать настройки pdfplumber по умолчанию" —
        # детектор линий разметки таблицы)
        ...

    def parse(self, path: str | Path) -> PDFParseResult:
        """Разобрать PDF-файл в PDFParseResult.

        TODO:
        1. Создать result = PDFParseResult().
        2. Открыть файл: with pdfplumber.open(str(path)) as pdf: ...
        3. Для каждой страницы (нумерация с 1:
           enumerate(pdf.pages, start=1)):
           a. full_text = page.extract_text() or ""
           b. Если full_text.strip() == "":
              - добавить номер страницы в result.pages_without_text_layer
              - перейти к следующей странице (continue) — текстового слоя
                нет, извлекать из этой страницы больше ничего не нужно
           c. Иначе: tables = page.find_tables(table_settings=self.table_settings)
           d. Для каждого найденного table в tables:
              - rows = table.extract()
              - добавить в result.elements
                PDFElement(element_type="table", page_number=<номер>,
                           rows=rows, bbox=table.bbox)
           e. Вырезать область всех найденных таблиц перед извлечением
              оставшегося текста страницы:
                  text_page = page
                  for table in tables:
                      text_page = text_page.outside_bbox(table.bbox)
              (если tables пуст — text_page остаётся исходной page)
           f. remaining_text = (text_page.extract_text() or "").strip()
           g. Если remaining_text не пуст — добавить в result.elements
              PDFElement(element_type="text_block", page_number=<номер>,
                         content=remaining_text)
        4. Вернуть result.

        Порядок важен только внутри одной страницы: таблицы должны быть
        вырезаны из области страницы ДО извлечения текста, иначе значения
        ячеек попадут одновременно в rows и в content.
        """
        ...

    def table_to_markdown(self, element: PDFElement) -> str:
        """Сериализовать табличный элемент в текст markdown-таблицы.

        Это единственное место в модуле, где строки таблицы превращаются
        в плоский текст — и делается это явно, с сохранением связи
        "значение — заголовок столбца" через синтаксис markdown-таблицы,
        а не порядком чтения слева-справа-сверху-вниз.

        TODO:
        1. Если element.element_type != "table" или не element.rows —
           поднять ValueError("элемент не является таблицей").
        2. Заменить все None-ячейки на "" (pdfplumber возвращает None для
           визуально пустых ячеек таблицы):
           rows = [[cell if cell is not None else "" for cell in row]
                   for row in element.rows]
        3. header, *body = rows — первая строка считается заголовком.
        4. Собрать строку заголовка: "| " + " | ".join(header) + " |"
        5. Собрать строку-разделитель:
           "| " + " | ".join(["---"] * len(header)) + " |"
        6. Для каждой строки body собрать аналогичную строку "| ... | ... |".
        7. Вернуть все строки, объединённые через "\\n".
        """
        ...
