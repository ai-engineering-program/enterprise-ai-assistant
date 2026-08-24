"""Tests for app/ingestion/pipeline_alerting.py (Lesson 7.2).

PipelineAlertMonitor only reacts to signals already computed by other
ingestion components (stage attempt/error counts, IngestionStageProfiler
stage_breakdown, queue depth/capacity, QualityViolation from
DataQualityReport) — these unit tests build those inputs directly as
plain values, without running any real parsing, OCR, embedding pipeline
or external notification channel.

Run unit tests only:
    pytest tests/test_pipeline_alerting.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.data_quality_report import QualityViolation
from app.ingestion.pipeline_alerting import (
    Alert,
    OperationalThresholds,
    PipelineAlertMonitor,
)


# ---------------------------------------------------------------------------
# check_stage_error_rates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckStageErrorRates:
    def test_below_threshold_no_alert(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_error_rate=0.05)
        )
        alerts = monitor.check_stage_error_rates(
            stage_attempts={"parse": 100}, stage_errors={"parse": 1}
        )
        assert alerts == []

    def test_above_threshold_warning(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(
                max_error_rate=0.02, critical_multiplier=3.0
            )
        )
        # 5% ошибок при пороге 2% — превышение, но не в 3x (нужно 6%+)
        alerts = monitor.check_stage_error_rates(
            stage_attempts={"parse": 100}, stage_errors={"parse": 5}
        )
        assert len(alerts) == 1
        assert alerts[0].signal == "error_rate:parse"
        assert alerts[0].severity == "warning"

    def test_far_above_threshold_critical(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(
                max_error_rate=0.02, critical_multiplier=2.0
            )
        )
        # 20% ошибок при пороге 2% — далеко за критическим множителем
        alerts = monitor.check_stage_error_rates(
            stage_attempts={"parse": 100}, stage_errors={"parse": 20}
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_zero_attempts_stage_skipped(self):
        monitor = PipelineAlertMonitor()
        alerts = monitor.check_stage_error_rates(
            stage_attempts={"embed": 0}, stage_errors={"embed": 0}
        )
        assert alerts == []

    def test_only_offending_stage_alerts(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_error_rate=0.02)
        )
        alerts = monitor.check_stage_error_rates(
            stage_attempts={"parse": 100, "embed": 100},
            stage_errors={"parse": 20, "embed": 1},
        )
        signals = {a.signal for a in alerts}
        assert signals == {"error_rate:parse"}


# ---------------------------------------------------------------------------
# check_stage_latency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckStageLatency:
    def test_below_threshold_no_alert(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(
                max_stage_latency_ms={"embed": 800.0}
            )
        )
        alerts = monitor.check_stage_latency(
            {"embed": {"count": 10, "total_ms": 4000.0, "mean_ms": 400.0, "share_pct": 100.0}}
        )
        assert alerts == []

    def test_above_threshold_alerts(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(
                max_stage_latency_ms={"embed": 800.0}, critical_multiplier=2.0
            )
        )
        alerts = monitor.check_stage_latency(
            {"embed": {"count": 10, "total_ms": 20000.0, "mean_ms": 2000.0, "share_pct": 100.0}}
        )
        assert len(alerts) == 1
        assert alerts[0].signal == "latency:embed"
        assert alerts[0].severity == "critical"

    def test_stage_without_configured_threshold_ignored(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_stage_latency_ms={"embed": 800.0})
        )
        alerts = monitor.check_stage_latency(
            {"upsert": {"count": 10, "total_ms": 999999.0, "mean_ms": 99999.9, "share_pct": 100.0}}
        )
        assert alerts == []

    def test_configured_stage_missing_from_breakdown_ignored(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_stage_latency_ms={"embed": 800.0})
        )
        alerts = monitor.check_stage_latency({})
        assert alerts == []


# ---------------------------------------------------------------------------
# check_queue_backpressure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckQueueBackpressure:
    def test_below_threshold_no_alert(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_queue_utilization=0.9)
        )
        alerts = monitor.check_queue_backpressure(queue_depth=500, queue_capacity=1000)
        assert alerts == []

    def test_above_threshold_alerts(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_queue_utilization=0.9)
        )
        alerts = monitor.check_queue_backpressure(queue_depth=950, queue_capacity=1000)
        assert len(alerts) == 1
        assert alerts[0].signal == "queue_backpressure"
        assert alerts[0].value == pytest.approx(0.95)

    def test_zero_capacity_returns_empty(self):
        monitor = PipelineAlertMonitor()
        assert monitor.check_queue_backpressure(queue_depth=10, queue_capacity=0) == []


# ---------------------------------------------------------------------------
# check_freshness_drift
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckFreshnessDrift:
    def test_no_baseline_returns_empty(self):
        monitor = PipelineAlertMonitor()
        assert monitor.check_freshness_drift(current_p95_seconds=1000.0, baseline_p95_seconds=0.0) == []

    def test_within_drift_ratio_no_alert(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_freshness_drift_ratio=2.0)
        )
        alerts = monitor.check_freshness_drift(
            current_p95_seconds=1800.0, baseline_p95_seconds=1000.0
        )
        assert alerts == []

    def test_slow_batch_source_with_matching_baseline_no_alert(self):
        # Источник с батч-синхронизацией раз в сутки: baseline == текущее значение.
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(max_freshness_drift_ratio=2.0)
        )
        twenty_hours = 20 * 3600.0
        alerts = monitor.check_freshness_drift(
            current_p95_seconds=twenty_hours, baseline_p95_seconds=twenty_hours
        )
        assert alerts == []

    def test_sudden_drift_alerts(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(
                max_freshness_drift_ratio=2.0, critical_multiplier=2.0
            )
        )
        twenty_hours = 20 * 3600.0
        four_days = 4 * 24 * 3600.0
        alerts = monitor.check_freshness_drift(
            current_p95_seconds=four_days, baseline_p95_seconds=twenty_hours
        )
        assert len(alerts) == 1
        assert alerts[0].signal == "freshness_drift"
        assert alerts[0].value == pytest.approx(four_days / twenty_hours)


# ---------------------------------------------------------------------------
# from_quality_violations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromQualityViolations:
    def test_wraps_violations_without_recomputing(self):
        monitor = PipelineAlertMonitor(
            operational_thresholds=OperationalThresholds(critical_multiplier=2.0)
        )
        violations = [
            QualityViolation(
                metric="completeness", value=0.60, threshold=0.98,
                message="completeness 0.600 ниже порога 0.980",
            ),
            QualityViolation(
                metric="duplication_rate", value=0.06, threshold=0.05,
                message="duplication_rate 0.060 выше порога 0.050",
            ),
        ]
        alerts = monitor.from_quality_violations(violations)
        assert len(alerts) == 2
        signals = {a.signal for a in alerts}
        assert signals == {"quality:completeness", "quality:duplication_rate"}
        completeness_alert = next(a for a in alerts if a.signal == "quality:completeness")
        assert completeness_alert.severity == "critical"
        assert completeness_alert.message == "completeness 0.600 ниже порога 0.980"
        duplication_alert = next(a for a in alerts if a.signal == "quality:duplication_rate")
        assert duplication_alert.severity == "warning"

    def test_empty_violations_returns_empty(self):
        monitor = PipelineAlertMonitor()
        assert monitor.from_quality_violations([]) == []


# ---------------------------------------------------------------------------
# dispatch / history — cooldown behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDispatch:
    def test_first_dispatch_calls_sink_and_records_history(self):
        received = []
        monitor = PipelineAlertMonitor(sink=received.append, cooldown_seconds=60.0)
        alert = Alert(
            signal="error_rate:parse", value=0.2, threshold=0.02,
            severity="critical", message="m",
        )
        dispatched = monitor.dispatch([alert])
        assert dispatched == [alert]
        assert received == [alert]
        assert monitor.history() == [alert]

    def test_repeat_within_cooldown_is_suppressed(self):
        fake_now = [1000.0]
        received = []
        monitor = PipelineAlertMonitor(
            sink=received.append, cooldown_seconds=60.0, clock=lambda: fake_now[0]
        )
        alert = Alert(
            signal="queue_backpressure", value=0.95, threshold=0.9,
            severity="warning", message="m",
        )

        first = monitor.dispatch([alert])
        assert first == [alert]

        fake_now[0] += 30.0  # ещё внутри окна cooldown (60s)
        second = monitor.dispatch([alert])
        assert second == []
        assert received == [alert]  # sink не вызван повторно

    def test_repeat_after_cooldown_expires_is_sent_again(self):
        fake_now = [1000.0]
        received = []
        monitor = PipelineAlertMonitor(
            sink=received.append, cooldown_seconds=60.0, clock=lambda: fake_now[0]
        )
        alert = Alert(
            signal="freshness_drift", value=4.8, threshold=2.0,
            severity="critical", message="m",
        )

        monitor.dispatch([alert])
        fake_now[0] += 70.0  # окно cooldown истекло
        second = monitor.dispatch([alert])

        assert second == [alert]
        assert received == [alert, alert]
        assert monitor.history() == [alert, alert]

    def test_different_signals_do_not_share_cooldown(self):
        received = []
        monitor = PipelineAlertMonitor(sink=received.append, cooldown_seconds=900.0)
        alert_a = Alert(signal="error_rate:parse", value=0.1, threshold=0.02, severity="warning", message="a")
        alert_b = Alert(signal="error_rate:embed", value=0.1, threshold=0.02, severity="warning", message="b")

        dispatched = monitor.dispatch([alert_a, alert_b])
        assert dispatched == [alert_a, alert_b]

    def test_history_returns_copy_not_internal_reference(self):
        monitor = PipelineAlertMonitor()
        alert = Alert(signal="s", value=1.0, threshold=0.5, severity="warning", message="m")
        monitor.dispatch([alert])
        history_copy = monitor.history()
        history_copy.append("mutated")
        assert monitor.history() == [alert]


# ---------------------------------------------------------------------------
# Integration — requires a real notification channel
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineAlertMonitorIntegration:
    """Требует реального канала уведомлений (Slack webhook / PagerDuty API).
    Скип по умолчанию: pytest -m 'not integration'."""

    def test_dispatch_to_real_slack_webhook(self):
        pytest.skip(
            "Требует настроенный Slack/PagerDuty webhook — запускайте вручную:\n"
            "  pytest tests/test_pipeline_alerting.py -m integration -v"
        )
