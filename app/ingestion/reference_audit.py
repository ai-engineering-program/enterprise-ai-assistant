from __future__ import annotations

from dataclasses import dataclass, field


__all__ = ["AuditReport", "ReferenceSampleAuditor"]


@dataclass
class AuditReport:
    """Итог калибровочного аудита по выборке пар (OCR-текст, эталон)."""

    mean_cer: float
    per_block_cer: list[float] = field(default_factory=list)
    worst_block_indices: list[int] = field(default_factory=list)


class ReferenceSampleAuditor:
    """
    Калибровочный аудит: сравнение OCR-вывода с эталонным (вручную
    вычитанным) текстом на небольшой размеченной выборке — аудит, который
    команда должна проводить ДО выбора порога confidence_threshold в
    OCRConfidenceGate (урок 1.2), а не постфактум после инцидента.

    Метрика — character error rate (CER): отношение расстояния
    Левенштейна между OCR-текстом и эталоном к длине эталона. Реализация
    расстояния Левенштейна — учебная (Wagner-Fischer, O(n*m) по времени и
    памяти), без внешних зависимостей.
    """

    def levenshtein_distance(self, a: str, b: str) -> int:
        """
        Классическое расстояние редактирования между строками a и b:
        минимальное число вставок, удалений и замен символов, чтобы
        превратить a в b.

        TODO: алгоритм Вагнера-Фишера.
        1. Построить матрицу dp размера (len(a)+1) x (len(b)+1).
        2. dp[i][0] = i для всех i (удалить все символы a[:i]).
           dp[0][j] = j для всех j (вставить все символы b[:j]).
        3. Для i от 1 до len(a), j от 1 до len(b):
           если a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
           иначе: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        4. Вернуть dp[len(a)][len(b)].
        """
        ...

    def character_error_rate(self, ocr_text: str, reference_text: str) -> float:
        """
        Доля символов эталона, которые нужно исправить, чтобы получить
        OCR-текст (или наоборот — расстояние симметрично).

        TODO:
        - если reference_text == "":
          вернуть 0.0, если ocr_text тоже "" (оба пусты — нет ошибки),
          иначе вернуть 1.0 (эталон пуст, а OCR что-то "нашёл" — 100% шум)
        - иначе вернуть self.levenshtein_distance(ocr_text, reference_text)
          / len(reference_text)
        """
        ...

    def audit_sample(self, pairs: list[tuple[str, str]]) -> AuditReport:
        """
        Посчитать CER по калибровочной выборке и найти самые проблемные
        блоки.

        Args:
            pairs: список (ocr_text, reference_text) — один элемент на
                каждый блок выборки, в том порядке, в котором блоки идут
                в исходном документе.

        TODO:
        1. per_block_cer = [self.character_error_rate(ocr, ref)
                             for ocr, ref in pairs]
        2. mean_cer = среднее значений per_block_cer;
           0.0, если pairs пуст (не делить на ноль).
        3. worst_block_indices = индексы (в pairs) до 3 блоков с
           наибольшим CER, отсортированные по убыванию CER. Если в
           выборке меньше 3 блоков — вернуть индексы всех блоков,
           тоже по убыванию CER.
        4. Вернуть AuditReport(mean_cer, per_block_cer, worst_block_indices).
        """
        ...
