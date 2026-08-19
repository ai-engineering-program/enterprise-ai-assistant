from __future__ import annotations

from dataclasses import dataclass, field


__all__ = [
    "CYRILLIC_LATIN_HOMOGLYPHS",
    "ReviewFlag",
    "DiagnosticsReport",
    "OCRQualityDiagnostics",
]


# Буквы, которые в большинстве шрифтов имеют практически идентичное
# начертание в кириллице и латинице ("омоглифы") — OCR-движок в режиме
# rus+eng выбирает алфавит для каждого символа независимо и может
# перепутать раскладку, даже будучи полностью уверенным в форме символа.
# Ключ — кириллическая буква, значение — визуально совпадающая латинская.
CYRILLIC_LATIN_HOMOGLYPHS: dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
}


@dataclass
class ReviewFlag:
    """Решение диагностики по одному блоку распознанного текста."""

    block_id: str
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    dictionary_ratio: float = 1.0
    homoglyph_tokens: list[str] = field(default_factory=list)


@dataclass
class DiagnosticsReport:
    """Итог диагностики по всей партии блоков."""

    total_blocks: int
    flagged_blocks: list[ReviewFlag] = field(default_factory=list)


class OCRQualityDiagnostics:
    """
    Диагностика качества OCR сверх confidence score (см. OCRConfidenceGate,
    урок 1.2, app/ingestion/ocr_confidence.py).

    Использует два независимых сигнала, ни один из которых не покрывается
    confidence gate:

    1. Доля словарных слов — ловит деградацию связного текста, где движок
       массово путает буквы внутри обычных слов языка.
    2. Кросс-скриптовые токены — ловит смешение кириллицы и латиницы
       внутри одного "слова", типичное следствие путаницы омоглифов в
       идентификаторах и кодах (трек-номера, артикулы, реестровые коды).

    Оба сигнала независимы от confidence: движок может быть уверен в
    форме каждого символа и всё равно ошибиться в выборе алфавита или
    массово деградировать целое слово при плохом качестве печати.
    """

    def __init__(self, dictionary: set[str], min_dictionary_ratio: float = 0.7) -> None:
        # TODO: сохранить dictionary и min_dictionary_ratio как атрибуты
        # экземпляра (self.dictionary, self.min_dictionary_ratio)
        ...

    def dictionary_word_ratio(self, text: str) -> float:
        """
        Доля буквенных токенов текста, входящих в словарь языка.

        TODO:
        1. Разбить text на токены по пробелам (text.split()).
        2. Отобрать alpha_tokens — токены, состоящие только из букв
           (token.isalpha()) и длиной >= 2 символов. Токены с цифрами,
           дефисами или другой пунктуацией (коды, номера, даты) в
           alpha_tokens не входят — у них по определению не может быть
           совпадения со словарём естественного языка.
        3. Если alpha_tokens пуст — вернуть 1.0 (блок состоит из кодов
           и чисел, а не из связного текста: нечего оценивать по этому
           сигналу, и это не повод его подозревать).
        4. Иначе вернуть долю токенов (в нижнем регистре), которые есть
           в self.dictionary, от общего числа alpha_tokens.
        """
        ...

    def find_homoglyph_tokens(self, text: str) -> list[str]:
        """
        Найти токены, где кириллические и латинские буквы смешаны в
        пределах одного "слова" — типичный след путаницы омоглифов.

        TODO:
        1. Разбить text на токены по пробелам (text.split()).
        2. Токен считается подозрительным, если в нём одновременно
           встречается хотя бы одна буква из диапазона кириллицы
           (а-яА-ЯёЁ, используйте re.search(r"[а-яА-ЯёЁ]", token)) И
           хотя бы одна буква из диапазона латиницы (a-zA-Z,
           re.search(r"[a-zA-Z]", token)).
        3. Вернуть список таких токенов в порядке появления в тексте,
           без повторов (сохраняя первое появление каждого).
        """
        ...

    def diagnose_block(self, block_id: str, text: str) -> ReviewFlag:
        """
        Принять решение по одному блоку на основе обоих сигналов.

        TODO:
        1. ratio = self.dictionary_word_ratio(text)
        2. homoglyphs = self.find_homoglyph_tokens(text)
        3. reasons = []
           - если ratio < self.min_dictionary_ratio:
             reasons.append("low_dictionary_ratio")
           - если homoglyphs не пуст:
             reasons.append("homoglyph_tokens")
        4. flagged = bool(reasons)
        5. вернуть ReviewFlag(block_id, flagged, reasons, ratio, homoglyphs)
        """
        ...

    def diagnose_batch(self, blocks: list[tuple[str, str]]) -> DiagnosticsReport:
        """
        Применить diagnose_block к партии блоков.

        Args:
            blocks: список пар (block_id, text).

        TODO:
        1. Вызвать self.diagnose_block(block_id, text) для каждой пары.
        2. Собрать в flagged_blocks только те ReviewFlag, где
           flagged == True, в исходном порядке.
        3. Вернуть DiagnosticsReport(total_blocks=len(blocks),
           flagged_blocks=flagged_blocks).
        """
        ...
