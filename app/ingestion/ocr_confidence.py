from __future__ import annotations

from dataclasses import dataclass, field


__all__ = ["OCRBlock", "QualityDecision", "OCRConfidenceGate", "AMBIGUOUS_CHAR_PAIRS"]


# Пары символов, которые OCR-движки систематически путают в кириллических
# документах: цифра слева, визуально похожая буква справа.
AMBIGUOUS_CHAR_PAIRS: dict[str, str] = {
    "0": "О",
    "3": "З",
    "6": "б",
    "1": "l",
    "5": "S",
}


@dataclass
class OCRBlock:
    """Один распознанный текстовый блок (абзац/строка/ячейка) с результатом OCR."""

    block_id: str
    text: str
    word_confidences: list[float]  # confidence 0.0-1.0 для каждого слова блока


@dataclass
class QualityDecision:
    """Решение confidence gate по одному блоку."""

    block_id: str
    average_confidence: float
    suspicious_tokens: list[str] = field(default_factory=list)
    action: str = "index"  # "index" | "quarantine"


class OCRConfidenceGate:
    """
    Шлюз качества распознавания между OCR и чанкингом.

    Блок текста попадает в индекс только если:
    1. Средняя confidence по словам блока не ниже порога.
    2. В тексте блока нет "подозрительных" токенов — фрагментов, где цифры
       соседствуют с буквами из AMBIGUOUS_CHAR_PAIRS (например номер пункта,
       где OCR мог перепутать цифру и визуально похожую букву).

    Блоки, не прошедшие проверку, не индексируются как есть — они
    помечаются action="quarantine" для ручной проверки или повторной
    обработки.
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self.confidence_threshold = confidence_threshold

    def average_confidence(self, block: OCRBlock) -> float:
        """
        Средняя confidence по всем словам блока.

        TODO:
        - если block.word_confidences пуст — вернуть 0.0
        - иначе вернуть среднее арифметическое значений списка
        """
        ...

    def find_suspicious_tokens(self, text: str) -> list[str]:
        """
        Найти в тексте токены, где цифры соседствуют с похожими буквами.

        TODO:
        1. Разбить text на токены по пробелам (text.split()).
        2. Собрать множество "подозрительных" символов — объединение ключей
           и значений AMBIGUOUS_CHAR_PAIRS (цифры и похожие на них буквы).
        3. Токен считается подозрительным, если в нём одновременно
           встречается хотя бы одна цифра (0-9) И хотя бы одна буква из
           значений AMBIGUOUS_CHAR_PAIRS ("О", "З", "б", "l", "S")
           — то есть смешение цифр и визуально похожих букв в одном токене.
        4. Вернуть список подозрительных токенов (в порядке появления,
           без дублей).
        """
        ...

    def evaluate(self, block: OCRBlock) -> QualityDecision:
        """
        Принять решение по блоку: index или quarantine.

        TODO:
        1. avg = self.average_confidence(block)
        2. suspicious = self.find_suspicious_tokens(block.text)
        3. action = "index", если avg >= self.confidence_threshold И
           suspicious пуст; иначе action = "quarantine"
        4. вернуть QualityDecision(block.block_id, avg, suspicious, action)
        """
        ...

    def filter_blocks(
        self, blocks: list[OCRBlock]
    ) -> tuple[list[OCRBlock], list[QualityDecision]]:
        """
        Применить evaluate() к списку блоков.

        TODO:
        - вызвать self.evaluate(block) для каждого блока
        - собрать blocks_to_index — исходные блоки, для которых
          decision.action == "index", в исходном порядке
        - собрать all_decisions — решения для ВСЕХ блоков, включая quarantine
        - вернуть (blocks_to_index, all_decisions)
        """
        ...
