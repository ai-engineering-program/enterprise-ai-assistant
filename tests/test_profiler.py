import time
import pytest
from app.rag.profiler import ComponentProfiler, timed


@pytest.mark.unit
class TestComponentProfilerUnit:
    """Unit-тесты для ComponentProfiler. Работают без внешних сервисов."""

    def test_stop_returns_positive_duration(self):
        profiler = ComponentProfiler()
        profiler.start("encode_query")
        time.sleep(0.005)  # 5 мс
        elapsed = profiler.stop("encode_query")
        assert elapsed > 0, "elapsed_ms должен быть положительным"

    def test_stop_records_timing(self):
        profiler = ComponentProfiler()
        profiler.start("search")
        profiler.stop("search")
        report = profiler.report()
        assert "search" in report
        assert report["search"]["count"] == 1

    def test_report_contains_required_keys(self):
        profiler = ComponentProfiler()
        profiler.start("rerank")
        profiler.stop("rerank")
        report = profiler.report()
        entry = report["rerank"]
        for key in ("count", "mean_ms", "p50_ms", "p95_ms"):
            assert key in entry, f"Ключ '{key}' отсутствует в report()"

    def test_p95_gte_mean_after_many_samples(self):
        """При 20 замерах p95 должен быть не меньше среднего."""
        profiler = ComponentProfiler()
        for _ in range(20):
            profiler.start("encode_query")
            time.sleep(0.001)
            profiler.stop("encode_query")
        report = profiler.report()
        assert report["encode_query"]["p95_ms"] >= report["encode_query"]["mean_ms"]

    def test_stop_without_start_raises_runtime_error(self):
        profiler = ComponentProfiler()
        with pytest.raises(RuntimeError):
            profiler.stop("nonexistent_component")

    def test_multiple_components_tracked_independently(self):
        profiler = ComponentProfiler()
        for _ in range(3):
            profiler.start("encode_query")
            profiler.stop("encode_query")
        for _ in range(5):
            profiler.start("search")
            profiler.stop("search")
        report = profiler.report()
        assert report["encode_query"]["count"] == 3
        assert report["search"]["count"] == 5

    def test_empty_report_returns_empty_dict(self):
        profiler = ComponentProfiler()
        assert profiler.report() == {}


@pytest.mark.unit
class TestTimedDecoratorUnit:
    """Unit-тесты для декоратора timed()."""

    def test_timed_preserves_function_name(self):
        profiler = ComponentProfiler()

        @timed(profiler, "my_component")
        def my_func(x: int) -> int:
            return x * 2

        assert my_func.__name__ == "my_func"

    def test_timed_records_timing(self):
        profiler = ComponentProfiler()

        @timed(profiler, "encode_query")
        def encode(text: str) -> list:
            time.sleep(0.002)
            return [0.1, 0.2]

        result = encode("hello")
        assert result == [0.1, 0.2]
        report = profiler.report()
        assert "encode_query" in report
        assert report["encode_query"]["count"] == 1

    def test_timed_records_on_exception(self):
        """Замер должен записаться даже если функция поднимает исключение."""
        profiler = ComponentProfiler()

        @timed(profiler, "failing_component")
        def broken():
            raise ValueError("ошибка")

        with pytest.raises(ValueError):
            broken()

        report = profiler.report()
        assert "failing_component" in report
        assert report["failing_component"]["count"] == 1

    def test_timed_multiple_calls_accumulate(self):
        profiler = ComponentProfiler()

        @timed(profiler, "rerank")
        def rerank(items: list) -> list:
            return items

        for _ in range(7):
            rerank([1, 2, 3])

        report = profiler.report()
        assert report["rerank"]["count"] == 7


@pytest.mark.integration
class TestComponentProfilerIntegration:
    """Интеграционные тесты. Запускать вручную, не требуют внешних сервисов,
    но выполняются дольше обычных unit-тестов."""

    def test_real_timing_accuracy(self):
        """Проверяем, что задержка ~50 мс измеряется с точностью ±10 мс."""
        profiler = ComponentProfiler()
        profiler.start("slow_component")
        time.sleep(0.050)
        elapsed = profiler.stop("slow_component")
        assert 40 <= elapsed <= 80, (
            f"Ожидалось 40–80 мс, получено {elapsed:.1f} мс"
        )

    def test_p95_with_artificial_outliers(self):
        """p95 должен захватывать хвост распределения."""
        profiler = ComponentProfiler()
        # 19 быстрых замеров и 1 медленный
        for _ in range(19):
            profiler.start("encode_query")
            time.sleep(0.001)  # ~1 мс
            profiler.stop("encode_query")

        profiler.start("encode_query")
        time.sleep(0.100)  # 100 мс — outlier
        profiler.stop("encode_query")

        report = profiler.report()
        # p95 должен отражать медленный запрос
        assert report["encode_query"]["p95_ms"] > report["encode_query"]["mean_ms"] * 2
