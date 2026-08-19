from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


__all__ = [
    "OCRTrigger",
    "PageTextStats",
    "OCRNecessityReport",
    "OCRNecessityDetector",
]


class OCRTrigger(str, Enum):
    """Тип решения о необходимости OCR для документа."""

    NONE = "none"                  # текстовый слой в порядке, OCR не нужен
    LOW_DENSITY = "low_density"    # ни одна страница не пуста, но текста в целом мало
    PARTIAL_SCAN = "partial_scan"  # часть страниц ниже порога — обычно скан+текст вперемешку
    FULL_SCAN = "full_scan"        # почти весь документ ниже порога


@dataclass
class PageTextStats:
    """Статистика извлечённого текстового слоя одной страницы документа.

    char_count — число символов, которые извлёк обычный текстовый парсер
    (PDFParser.parse() из урока 2.2), ПОСЛЕ .strip(). Страница из
    PDFParseResult.pages_without_text_layer имеет char_count == 0, но
    здесь допускаются и небольшие положительные значения — например,
    несколько символов, которые слабый встроенный OCR старого сканера
    успел разобрать в оттиске печати.
    """

    page_number: int
    char_count: int


@dataclass
class OCRNecessityReport:
    """Итоговое решение детектора по документу."""

    needs_ocr: bool
    trigger: OCRTrigger
    empty_page_ratio: float
    avg_chars_per_page: float
    pages_for_ocr: list[int] = field(default_factory=list)


class OCRNecessityDetector:
    """
    Решает, нужен ли документу OCR-конвейер (урок 3.2), на основе
    постраничной статистики извлечённого текстового слоя — а не на
    основе содержимого самих страниц.

    Работает поверх уже посчитанной PDFParser.parse() информации
    (урок 2.2): страницы без текстового слоя уже известны через
    pages_without_text_layer, число извлечённых символов на странице
    можно получить как len(remaining_text) при разборе. Этот класс не
    парсит документ заново — он принимает готовую статистику и решает,
    нужно ли отправить документ (или отдельные его страницы) в
    OCR-конвейер урока 3.2, прежде чем результат уйдёт в чанкер (см.
    диаграмму размещения в тексте урока 3.1).
    """

    def __init__(
        self,
        min_chars_per_page: int = 40,
        full_scan_ratio_threshold: float = 0.9,
        low_density_multiplier: float = 2.0,
    ) -> None:
        # TODO: сохранить все три параметра как атрибуты экземпляра:
        # self.min_chars_per_page, self.full_scan_ratio_threshold,
        # self.low_density_multiplier
        ...

    def assess(self, pages: list[PageTextStats]) -> OCRNecessityReport:
        """
        Принять решение по документу на основе статистики его страниц.

        TODO:
        1. Если pages пуст -> вернуть OCRNecessityReport(
               needs_ocr=False, trigger=OCRTrigger.NONE,
               empty_page_ratio=0.0, avg_chars_per_page=0.0, pages_for_ocr=[])
           (документу без страниц нечего отправлять в OCR)
        2. total = len(pages)
        3. low_pages = [p for p in pages if p.char_count < self.min_chars_per_page]
           (сюда попадают страницы с char_count == 0 — те самые
           pages_without_text_layer из PDFParseResult, — и страницы с
           несколькими символами: колонтитул или штамп, отсканированный
           слабым встроенным OCR сканера)
        4. empty_ratio = len(low_pages) / total
        5. avg_chars = sum(p.char_count for p in pages) / total
        6. Определить trigger:
           - если low_pages непуст:
               - empty_ratio >= self.full_scan_ratio_threshold -> OCRTrigger.FULL_SCAN
               - иначе -> OCRTrigger.PARTIAL_SCAN
           - иначе (ни одна страница по отдельности не ниже порога):
               - avg_chars < self.min_chars_per_page * self.low_density_multiplier
                 -> OCRTrigger.LOW_DENSITY
               - иначе -> OCRTrigger.NONE
        7. needs_ocr = (trigger != OCRTrigger.NONE)
        8. Определить pages_for_ocr:
           - если trigger == OCRTrigger.LOW_DENSITY -> номера ВСЕХ страниц
             документа (нет точечного сигнала "какая именно страница" —
             разумнее переобработать документ целиком, чем угадывать)
           - иначе -> [p.page_number for p in low_pages], в исходном
             порядке появления в pages
        9. Вернуть OCRNecessityReport(needs_ocr, trigger, empty_ratio,
                                        avg_chars, pages_for_ocr)
        """
        ...

    def from_pdf_parse_result(
        self,
        total_pages: int,
        pages_without_text_layer: list[int],
        char_counts_by_page: dict[int, int],
    ) -> OCRNecessityReport:
        """
        Собрать список PageTextStats напрямую из данных, эквивалентных
        PDFParseResult (урок 2.2, app/ingestion/pdf_parser.py), и вызвать
        assess() — без повторного разбора файла.

        TODO:
        1. Собрать pages: list[PageTextStats] для номеров страниц от 1 до
           total_pages включительно (range(1, total_pages + 1)):
           - если номер страницы есть в pages_without_text_layer ->
             char_count = 0
           - иначе -> char_count = char_counts_by_page.get(номер, 0)
        2. Вернуть self.assess(pages)
        """
        ...
