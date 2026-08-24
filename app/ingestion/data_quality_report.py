from __future__ import annotations

import math
from dataclasses import dataclass

from app.ingestion.document_deduplicator import DuplicateMatch
from app.ingestion.pipeline import IngestionReport


__all__ = [
    "DataQualityThresholds",
    "DataQualityMetrics",
    "QualityViolation",
    "DataQualityReport",
]


@dataclass
class DataQualityThresholds:
    """
    Пороговые значения метрик качества данных, сигнализирующие о проблеме
    (см. таблицу в тексте урока 7.1).

    Значения по умолчанию — отправная точка для первого прогона нового
    источника, а не универсальная константа: после нескольких недель
    эксплуатации разумнее калибровать пороги по исторической базовой
    линии конкретного источника.
    """

    min_completeness: float = 0.98
    min_validity: float = 0.95
    max_freshness_p95_seconds: float = 3600.0  # 1 час
    max_duplication_rate: float = 0.05


@dataclass
class DataQualityMetrics:
    """Итог одного прогона метрик качества данных пайплайна поглощения."""

    total_documents: int
    completeness: float
    validity: float
    freshness_p50_seconds: float
    freshness_p95_seconds: float
    duplication_rate: float


@dataclass
class QualityViolation:
    """Одно нарушение порогового значения — источник данных для алертов
    дежурному инженеру (см. урок 7.2)."""

    metric: str
    value: float
    threshold: float
    message: str


class DataQualityReport:
    """
    Агрегирует сигналы, уже посчитанные другими компонентами пайплайна
    поглощения, в четыре метрики качества данных верхнего уровня —
    completeness, validity, freshness, duplication rate (см. текст урока
    7.1). Этот класс НЕ парсит документы, НЕ считает character error
    rate и НЕ строит MinHash-сигнатуры заново — он только читает уже
    готовые результаты других компонентов:

    - IngestionReport (app/ingestion/pipeline.py, урок 1.4) плюс число
      документов в DeadLetterQueue (app/ingestion/retry_dead_letter.py,
      урок 6.3) -> completeness.
    - Множество doc_id, помеченных как invalid любым инструментом
      диагностики содержимого (например, DiagnosticsReport.flagged_blocks
      из app/ingestion/ocr_quality_diagnostics.py, урок 3.3, схлопнутый
      с уровня "блок" до уровня "документ") -> validity.
    - Список задержек в секундах между обновлением источника и
      обновлением индекса -> freshness (p50, p95).
    - Список DuplicateMatch из DocumentDeduplicator.register_document()
      (app/ingestion/document_deduplicator.py, урок 4.3) -> duplication
      rate.
    """

    def __init__(self, thresholds: DataQualityThresholds | None = None) -> None:
        # TODO: self.thresholds = thresholds or DataQualityThresholds()
        ...

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------
    def compute_completeness(
        self, ingestion_report: IngestionReport, dead_letter_count: int = 0
    ) -> float:
        """
        Доля документов, успешно дошедших до индекса, от числа всех
        документов, которые пайплайн попытался обработать в этом
        прогоне.

        dead_letter_count — число документов, окончательно осевших в
        DeadLetterQueue в ЭТОМ ЖЕ прогоне: они являются попыткой
        обработки, закончившейся неудачей, но не попадают в
        ingestion_report.files_skipped — это разные подсистемы
        отчётности (см. текст урока 7.1, "Знаменатель — не одно число").

        TODO:
        1. total_attempted = (len(ingestion_report.files_processed)
           + len(ingestion_report.files_skipped) + dead_letter_count)
        2. Если total_attempted == 0 — вернуть 1.0 (не было ни одной
           попытки обработки; пустой прогон не значит "провал").
        3. Иначе вернуть
           len(ingestion_report.files_processed) / total_attempted.
        """
        ...

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------
    def compute_validity(
        self, total_documents: int, invalid_document_ids: set[str]
    ) -> float:
        """
        Доля документов, прошедших парсинг и распознавание БЕЗ искажений
        содержимого — ось, ортогональная completeness (документ может
        успешно дойти до индекса и всё равно быть invalid).

        invalid_document_ids — множество doc_id документов, для которых
        хотя бы один инструмент диагностики содержимого нашёл проблему.
        Вызывающий код обязан схлопнуть несколько flagged-сигналов
        одного документа в один doc_id ДО вызова этого метода — сам
        метод этой схлопкой не занимается.

        TODO:
        1. Если total_documents == 0 — вернуть 1.0.
        2. invalid_count = len(invalid_document_ids)
        3. Вернуть max(0.0, 1.0 - invalid_count / total_documents)
           (защита от invalid_document_ids, по ошибке вызывающего кода
           содержащего больше элементов, чем total_documents).
        """
        ...

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------
    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        """
        Перцентиль pct (0-100) по values методом nearest-rank — без
        внешних зависимостей (numpy и т.п.).

        TODO:
        1. Если values пуст — вернуть 0.0.
        2. ordered = sorted(values)
        3. rank = math.ceil(pct / 100 * len(ordered))
        4. index = max(0, min(len(ordered) - 1, rank - 1))
        5. Вернуть ordered[index].
        """
        ...

    def compute_freshness(self, lag_seconds: list[float]) -> tuple[float, float]:
        """
        Медиана (p50) и p95 задержки между изменением документа в
        источнике и обновлением соответствующей точки в индексе.

        Почему не среднее арифметическое — см. текст урока 7.1: у
        freshness тяжёлый правый хвост, среднее по всей выборке
        маскирует именно тот хвост, который важнее всего увидеть.

        TODO:
        1. p50 = self._percentile(lag_seconds, 50)
        2. p95 = self._percentile(lag_seconds, 95)
        3. Вернуть (p50, p95).
        """
        ...

    # ------------------------------------------------------------------
    # Duplication rate
    # ------------------------------------------------------------------
    def compute_duplication_rate(
        self, duplicate_matches: list[DuplicateMatch]
    ) -> float:
        """
        Доля документов прогона, классифицированных
        DocumentDeduplicator.register_document() как "exact" или
        "near_duplicate" (то есть НЕ "unique"), от общего числа
        зарегистрированных документов.

        TODO:
        1. Если duplicate_matches пуст — вернуть 0.0.
        2. duplicates = [m for m in duplicate_matches
                          if m.match_type in ("exact", "near_duplicate")]
        3. Вернуть len(duplicates) / len(duplicate_matches).
        """
        ...

    # ------------------------------------------------------------------
    # Сборка полного отчёта
    # ------------------------------------------------------------------
    def build_report(
        self,
        *,
        ingestion_report: IngestionReport,
        total_documents: int,
        invalid_document_ids: set[str],
        freshness_lag_seconds: list[float],
        duplicate_matches: list[DuplicateMatch],
        dead_letter_count: int = 0,
    ) -> DataQualityMetrics:
        """
        Посчитать все четыре метрики и собрать их в единый
        DataQualityMetrics — итоговый отчёт по результатам одного
        прогона пайплайна поглощения.

        TODO:
        1. completeness = self.compute_completeness(ingestion_report, dead_letter_count)
        2. validity = self.compute_validity(total_documents, invalid_document_ids)
        3. p50, p95 = self.compute_freshness(freshness_lag_seconds)
        4. duplication_rate = self.compute_duplication_rate(duplicate_matches)
        5. Вернуть DataQualityMetrics(
               total_documents=total_documents,
               completeness=completeness,
               validity=validity,
               freshness_p50_seconds=p50,
               freshness_p95_seconds=p95,
               duplication_rate=duplication_rate,
           )
        """
        ...

    # ------------------------------------------------------------------
    # Проверка пороговых значений
    # ------------------------------------------------------------------
    def check_thresholds(self, metrics: DataQualityMetrics) -> list[QualityViolation]:
        """
        Сравнить metrics с self.thresholds и вернуть список нарушений —
        источник данных для алертов дежурному инженеру в уроке 7.2.

        TODO: добавить в violations (список, изначально пустой) по
        одному QualityViolation за каждое выполненное условие:
        1. metrics.completeness < self.thresholds.min_completeness ->
           QualityViolation(metric="completeness", value=metrics.completeness,
           threshold=self.thresholds.min_completeness, message=...)
        2. metrics.validity < self.thresholds.min_validity ->
           QualityViolation(metric="validity", ...)
        3. metrics.freshness_p95_seconds > self.thresholds.max_freshness_p95_seconds ->
           QualityViolation(metric="freshness_p95_seconds", ...)
        4. metrics.duplication_rate > self.thresholds.max_duplication_rate ->
           QualityViolation(metric="duplication_rate", ...)

        message — понятная строка на русском, например:
        f"completeness {metrics.completeness:.3f} ниже порога "
        f"{self.thresholds.min_completeness:.3f}"

        Вернуть violations (может быть пустым списком, если все метрики
        в норме).
        """
        ...
