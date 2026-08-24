from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.ingestion.data_contract import ContractViolation
from app.ingestion.data_quality_report import QualityViolation


__all__ = ["Alert", "OperationalThresholds", "PipelineAlertMonitor"]


@dataclass
class Alert:
    """Одно уведомление о нарушении порогового значения — единый формат
    для четырёх разных категорий сигналов этого урока (error rate по
    стадии, latency стадии, утилизация очереди, дрейф freshness,
    нарушения DataQualityReport из урока 7.1)."""

    signal: str
    value: float
    threshold: float
    severity: str  # "warning" | "critical"
    message: str
    fired_at: float = field(default_factory=time.time)


@dataclass
class OperationalThresholds:
    """
    Пороговые значения ОПЕРАЦИОННЫХ сигналов пайплайна поглощения —
    в отличие от DataQualityThresholds (урок 7.1), которые описывают
    качество СОДЕРЖИМОГО документов, эти пороги описывают поведение
    самого пайплайна как процесса.

    max_stage_latency_ms — порог задаётся ТОЛЬКО для стадий, явно
    перечисленных в этом словаре (например {"embed": 800.0, "upsert":
    500.0}); для остальных стадий latency не проверяется — нет
    обоснованного порога -> нет ложного алерта.

    critical_multiplier — во сколько раз значение должно превысить
    порог, чтобы severity стал "critical" вместо "warning" (см. текст
    урока 7.2, "Severity: не всякое нарушение порога одинаково важно").
    """

    max_error_rate: float = 0.02
    max_stage_latency_ms: dict[str, float] = field(default_factory=dict)
    max_queue_utilization: float = 0.9
    max_freshness_drift_ratio: float = 2.0
    critical_multiplier: float = 2.0


class PipelineAlertMonitor:
    """
    Сравнивает уже вычисленные операционные сигналы пайплайна
    поглощения с пороговыми значениями и превращает нарушения в Alert
    с защитой от повторной отправки (cooldown). Этот класс НЕ запускает
    пайплайн, НЕ считает error rate или latency заново и НЕ хранит
    расписание прогонов — он принимает уже готовые числа через отдельные
    check_*() методы (см. текст урока 7.2, "Сборка: как эти сигналы
    стыкуются в один цикл мониторинга").

    sink — вызываемый объект, реально отправляющий Alert во внешний
    канал (Slack webhook, PagerDuty, logging.warning и т.п.). По
    умолчанию — no-op: тесты и разработка без настроенного канала не
    должны требовать реальной интеграции.
    """

    def __init__(
        self,
        operational_thresholds: OperationalThresholds | None = None,
        sink: Optional[Callable[[Alert], None]] = None,
        cooldown_seconds: float = 900.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        # TODO: self._thresholds = operational_thresholds or OperationalThresholds()
        # TODO: self._sink = sink or (lambda alert: None)
        # TODO: self._cooldown_seconds = cooldown_seconds
        # TODO: self._clock = clock or time.time
        # TODO: self._last_fired: dict[str, float] = {}
        # TODO: self._history: list[Alert] = []
        ...

    # ------------------------------------------------------------------
    # Общая логика определения severity
    # ------------------------------------------------------------------
    @staticmethod
    def _severity_for(value: float, threshold: float, critical_multiplier: float) -> str:
        """
        Определить severity нарушения по относительному отклонению
        value от threshold — работает независимо от того, растёт ли
        "плохое" значение выше порога (error rate, latency, утилизация
        очереди, drift ratio) или падает ниже порога (completeness,
        validity из QualityViolation): в обоих случаях знак разности
        известен заранее (вызывающий код уже проверил, что порог
        нарушен), важна только величина отклонения.

        TODO:
        1. Если threshold == 0 — вернуть "critical" (защита от деления
           на ноль; нулевой порог означает "любое отклонение критично").
        2. relative_deviation = abs(value - threshold) / abs(threshold)
        3. Если relative_deviation >= (critical_multiplier - 1.0) —
           вернуть "critical".
        4. Иначе — вернуть "warning".
        """
        ...

    # ------------------------------------------------------------------
    # Error rate по стадиям
    # ------------------------------------------------------------------
    def check_stage_error_rates(
        self, stage_attempts: dict[str, int], stage_errors: dict[str, int]
    ) -> list[Alert]:
        """
        Сигнал: доля неуспешных попыток обработки документа отдельно по
        каждой стадии пайплайна (parse, ocr, embed, upsert...) — не по
        документу и не по прогону в целом (см. текст урока 7.2,
        "Важный нюанс: error rate — не одно число на весь пайплайн").

        stage_attempts / stage_errors — словари {имя_стадии: число},
        которые вызывающий код собирает из уже существующих счётчиков
        пайплайна. Этот метод НЕ ходит в пайплайн сам и не считает
        попытки заново.

        TODO:
        1. alerts = []
        2. Для каждой пары (stage, attempts) в stage_attempts.items():
           a. Если attempts == 0 — пропустить стадию (нет попыток ->
              нет сигнала).
           b. errors = stage_errors.get(stage, 0)
           c. rate = errors / attempts
           d. Если rate > self._thresholds.max_error_rate:
              severity = self._severity_for(rate,
                  self._thresholds.max_error_rate,
                  self._thresholds.critical_multiplier)
              message = (f"error rate стадии '{stage}' {rate:.3f} "
                  f"превышает порог {self._thresholds.max_error_rate:.3f} "
                  f"({errors}/{attempts} попыток)")
              alerts.append(Alert(signal=f"error_rate:{stage}", value=rate,
                  threshold=self._thresholds.max_error_rate,
                  severity=severity, message=message))
        3. Вернуть alerts.
        """
        ...

    # ------------------------------------------------------------------
    # Latency по стадиям
    # ------------------------------------------------------------------
    def check_stage_latency(self, stage_breakdown: dict[str, dict]) -> list[Alert]:
        """
        Сигнал: средняя длительность стадии (мс) — тот же формат, что
        возвращает IngestionStageProfiler.stage_breakdown() (урок 5.1):
        {stage: {"count": int, "total_ms": float, "mean_ms": float,
        "share_pct": float}}.

        TODO:
        1. alerts = []
        2. Для каждой пары (stage, threshold_ms) в
           self._thresholds.max_stage_latency_ms.items():
           a. Если stage not in stage_breakdown — пропустить (стадия не
              встретилась в этом прогоне).
           b. mean_ms = stage_breakdown[stage]["mean_ms"]
           c. Если mean_ms > threshold_ms:
              severity = self._severity_for(mean_ms, threshold_ms,
                  self._thresholds.critical_multiplier)
              message = (f"средняя латентность стадии '{stage}' "
                  f"{mean_ms:.1f} мс превышает порог {threshold_ms:.1f} мс")
              alerts.append(Alert(signal=f"latency:{stage}", value=mean_ms,
                  threshold=threshold_ms, severity=severity, message=message))
        3. Вернуть alerts.
        """
        ...

    # ------------------------------------------------------------------
    # Backpressure очереди
    # ------------------------------------------------------------------
    def check_queue_backpressure(
        self, queue_depth: int, queue_capacity: int
    ) -> list[Alert]:
        """
        Сигнал: утилизация очереди поглощения (BoundedIngestionQueue,
        урок 5.4) — глубина сама по себе ничего не говорит без ёмкости:
        глубина 950 из 1000 критична, глубина 950 из 100 000 — норма.

        TODO:
        1. Если queue_capacity <= 0 — вернуть [] (нет корректной ёмкости
           для расчёта утилизации).
        2. utilization = queue_depth / queue_capacity
        3. Если utilization > self._thresholds.max_queue_utilization:
           severity = self._severity_for(utilization,
               self._thresholds.max_queue_utilization,
               self._thresholds.critical_multiplier)
           message = (f"утилизация очереди поглощения {utilization:.2f} "
               f"превышает порог {self._thresholds.max_queue_utilization:.2f} "
               f"(глубина {queue_depth} из {queue_capacity})")
           вернуть [Alert(signal="queue_backpressure", value=utilization,
               threshold=self._thresholds.max_queue_utilization,
               severity=severity, message=message)]
        4. Иначе вернуть [].
        """
        ...

    # ------------------------------------------------------------------
    # Дрейф freshness относительно базовой линии источника
    # ------------------------------------------------------------------
    def check_freshness_drift(
        self, current_p95_seconds: float, baseline_p95_seconds: float
    ) -> list[Alert]:
        """
        Сигнал: относительный дрейф freshness p95 (урок 7.1) от
        ИСТОРИЧЕСКОЙ базовой линии конкретного источника, а не
        абсолютный порог. Абсолютный порог DataQualityThresholds.
        max_freshness_p95_seconds уже проверяется в
        DataQualityReport.check_thresholds() (урок 7.1) и легко
        проходит для источника, который синхронизируется медленно
        БЕЗ инцидента (см. таблицу калибровки порогов в тексте урока
        7.1 и текст урока 7.2, "Freshness: абсолютный порог и дрейф").
        Этот сигнал ловит другое: freshness ВНЕЗАПНО выросла в N раз
        относительно своей же недавней нормы.

        TODO:
        1. Если baseline_p95_seconds <= 0 — вернуть [] (нет базовой
           линии для сравнения).
        2. drift_ratio = current_p95_seconds / baseline_p95_seconds
        3. Если drift_ratio > self._thresholds.max_freshness_drift_ratio:
           severity = self._severity_for(drift_ratio,
               self._thresholds.max_freshness_drift_ratio,
               self._thresholds.critical_multiplier)
           message = (f"freshness p95 выросла в {drift_ratio:.1f}x "
               f"относительно базовой линии ({baseline_p95_seconds:.0f} с "
               f"-> {current_p95_seconds:.0f} с)")
           вернуть [Alert(signal="freshness_drift", value=drift_ratio,
               threshold=self._thresholds.max_freshness_drift_ratio,
               severity=severity, message=message)]
        4. Иначе вернуть [].
        """
        ...

    # ------------------------------------------------------------------
    # Обёртка над нарушениями качества данных из урока 7.1
    # ------------------------------------------------------------------
    def from_quality_violations(self, violations: list[QualityViolation]) -> list[Alert]:
        """
        Обернуть QualityViolation (DataQualityReport.check_thresholds(),
        урок 7.1) в Alert. Этот метод НЕ пересчитывает метрики качества
        данных заново — он только транслирует уже готовый сигнал в
        единый формат, понятный dispatch().

        TODO:
        1. alerts = []
        2. Для каждого violation в violations:
           severity = self._severity_for(violation.value, violation.threshold,
               self._thresholds.critical_multiplier)
           alerts.append(Alert(signal=f"quality:{violation.metric}",
               value=violation.value, threshold=violation.threshold,
               severity=severity, message=violation.message))
        3. Вернуть alerts.
        """
        ...

    # ------------------------------------------------------------------
    # Обёртка над нарушениями data contract из урока 7.3
    # ------------------------------------------------------------------
    def from_contract_violations(self, violations: list[ContractViolation]) -> list[Alert]:
        """
        Обернуть ContractViolation (DocumentContractValidator.validate_document()
        / .validate_batch(), урок 7.3) в Alert. В отличие от
        from_quality_violations(), severity здесь НЕ вычисляется через
        self._severity_for() — нарушение data contract не измеряет,
        насколько сильно значение отклонилось от порога, оно фиксирует
        бинарный факт "документ не соответствует согласованной схеме
        источника". Это всегда критично: документ с невалидным полем
        не может быть тихо обработан со значением по умолчанию (см.
        текст урока 7.3, инцидент "ГранитПресс") — severity всегда
        "critical".

        TODO:
        1. alerts = []
        2. Для каждого violation в violations:
           alerts.append(Alert(
               signal=f"contract:{violation.field}:{violation.violation_type}",
               value=1.0,
               threshold=0.0,
               severity="critical",
               message=f"[{violation.document_id}] {violation.message}",
           ))
        3. Вернуть alerts.
        """
        ...

    # ------------------------------------------------------------------
    # Отправка с защитой от повторов (cooldown)
    # ------------------------------------------------------------------
    def dispatch(self, alerts: list[Alert]) -> list[Alert]:
        """
        Отправить каждый alert в self._sink — но не чаще одного раза на
        signal за self._cooldown_seconds (см. текст урока 7.2,
        "Alert fatigue: почему нельзя слать алерт на каждое нарушение").

        Returns:
            Список alert, которые ДЕЙСТВИТЕЛЬНО были отправлены в sink
            (прошли cooldown). Alert, подавленные cooldown, в
            результат не входят.

        TODO:
        1. dispatched = []
        2. now = self._clock()
        3. Для alert в alerts:
           a. last = self._last_fired.get(alert.signal)
           b. Если last is not None и (now - last) < self._cooldown_seconds
              — пропустить alert (ещё в cooldown).
           c. self._last_fired[alert.signal] = now
           d. self._sink(alert)
           e. self._history.append(alert)
           f. dispatched.append(alert)
        4. Вернуть dispatched.
        """
        ...

    def history(self) -> list[Alert]:
        """
        Все alert, когда-либо реально отправленные через dispatch()
        (прошедшие cooldown) — источник данных для отладки и для
        дашборда "что мы уже сообщали дежурному".

        TODO: вернуть list(self._history) — копию, а не ссылку на
        внутренний список.
        """
        ...
