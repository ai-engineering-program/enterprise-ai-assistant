import pytest

from app.context.message_quality import ChatMessage
from app.context.thread_filter import Thread, ThreadNoiseFilter


def _thread_olt_jan() -> Thread:
    """Ранний тред: временное решение через reboot -f, позже признано багом."""
    return Thread(
        thread_id="thread-olt-jan",
        channel="#noc-incidents",
        topic_key="olt-reboot-procedure",
        last_active_at="2025-01-10T09:10:00",
        messages=[
            ChatMessage(
                "Игорь",
                "Если OLT завис после прошивки, делаем reboot -f через консоль",
                "2025-01-10T09:00:00",
            ),
            ChatMessage("Марина", "👍", "2025-01-10T09:02:00"),
            ChatMessage("Игорь", "Закрыли, все ОЛТ подняты", "2025-01-10T09:10:00"),
        ],
    )


def _thread_olt_sep() -> Thread:
    """Поздний тред на ту же тему: reboot -f признан багом, введена новая процедура."""
    return Thread(
        thread_id="thread-olt-sep",
        channel="#noc-incidents",
        topic_key="olt-reboot-procedure",
        last_active_at="2025-09-02T14:22:00",
        messages=[
            ChatMessage(
                "Вячеслав",
                "Через reboot -f OLT виснет повторно, это баг, вендор подтвердил",
                "2025-09-02T14:00:00",
            ),
            ChatMessage("Игорь", "окей", "2025-09-02T14:02:00"),
            ChatMessage(
                "Вячеслав",
                "Новая процедура: graceful restart через веб-интерфейс, инструкция в Confluence",
                "2025-09-02T14:20:00",
            ),
            ChatMessage("Данила", "Спасибо", "2025-09-02T14:22:00"),
        ],
    )


def _thread_snacks() -> Thread:
    """Тред чистой болтовни — почти целиком шум по MessageQualityScorer."""
    return Thread(
        thread_id="thread-snacks",
        channel="#noc-incidents",
        topic_key="office-snacks",
        last_active_at="2025-03-01T10:02:00",
        messages=[
            ChatMessage("Ольга", "😂", "2025-03-01T10:00:00"),
            ChatMessage("Игорь", "ok", "2025-03-01T10:01:00"),
            ChatMessage("Марина", "круто", "2025-03-01T10:02:00"),
        ],
    )


def _thread_dns() -> Thread:
    """Единственный тред по своей теме — не может быть superseded."""
    return Thread(
        thread_id="thread-dns",
        channel="#noc-incidents",
        topic_key="dns-cache-bug",
        last_active_at="2025-05-01T11:10:00",
        messages=[
            ChatMessage(
                "Данила",
                "Кэш DNS на резолвере не обновляется, нужно перезапустить unbound",
                "2025-05-01T11:00:00",
            ),
            ChatMessage("Игорь", "+1", "2025-05-01T11:01:00"),
            ChatMessage(
                "Данила",
                "Перезапустили unbound, кэш очистился, резолвинг восстановлен",
                "2025-05-01T11:10:00",
            ),
        ],
    )


@pytest.mark.unit
class TestSignalRatio:
    def test_mixed_thread_ratio(self):
        f = ThreadNoiseFilter()
        assert f.signal_ratio(_thread_olt_jan()) == pytest.approx(2 / 3)

    def test_pure_noise_thread_ratio_is_zero(self):
        f = ThreadNoiseFilter()
        assert f.signal_ratio(_thread_snacks()) == 0.0

    def test_empty_thread_ratio_is_zero(self):
        f = ThreadNoiseFilter()
        empty = Thread("t", "#c", "topic", "2025-01-01T00:00:00", [])
        assert f.signal_ratio(empty) == 0.0


@pytest.mark.unit
class TestIsWorthIndexing:
    def test_substantive_thread_is_worth_indexing(self):
        f = ThreadNoiseFilter()
        assert f.is_worth_indexing(_thread_olt_jan()) is True

    def test_noisy_thread_is_not_worth_indexing(self):
        f = ThreadNoiseFilter()
        assert f.is_worth_indexing(_thread_snacks()) is False

    def test_min_signal_messages_threshold(self):
        f = ThreadNoiseFilter()
        short_thread = Thread(
            "t",
            "#c",
            "topic",
            "2025-01-01T00:00:00",
            [
                ChatMessage(
                    "A",
                    "Единственное содержательное сообщение в этом треде",
                    "2025-01-01T00:00:00",
                )
            ],
        )
        # ratio = 1.0, но абсолютное число содержательных сообщений (1)
        # меньше дефолтного порога min_signal_messages=2
        assert f.is_worth_indexing(short_thread) is False

    def test_min_signal_ratio_threshold(self):
        f = ThreadNoiseFilter()
        assert (
            f.is_worth_indexing(_thread_olt_sep(), min_signal_ratio=0.9) is False
        )


@pytest.mark.unit
class TestFilterWorthIndexing:
    def test_filters_and_preserves_order(self):
        f = ThreadNoiseFilter()
        threads = [_thread_olt_jan(), _thread_snacks(), _thread_olt_sep(), _thread_dns()]
        result = f.filter_worth_indexing(threads)
        assert [t.thread_id for t in result] == [
            "thread-olt-jan",
            "thread-olt-sep",
            "thread-dns",
        ]

    def test_empty_input(self):
        f = ThreadNoiseFilter()
        assert f.filter_worth_indexing([]) == []


@pytest.mark.unit
class TestDetectSuperseded:
    def test_older_thread_marked_superseded_by_newer(self):
        f = ThreadNoiseFilter()
        threads = [_thread_olt_jan(), _thread_olt_sep()]
        result = f.detect_superseded(threads)
        assert result == {"thread-olt-jan": "thread-olt-sep"}

    def test_single_thread_topic_not_in_result(self):
        f = ThreadNoiseFilter()
        threads = [_thread_dns()]
        result = f.detect_superseded(threads)
        assert result == {}

    def test_canonical_thread_not_marked_superseded(self):
        f = ThreadNoiseFilter()
        threads = [_thread_olt_jan(), _thread_olt_sep()]
        result = f.detect_superseded(threads)
        assert "thread-olt-sep" not in result


@pytest.mark.unit
class TestIndexableFraction:
    def test_full_channel_example(self):
        f = ThreadNoiseFilter()
        threads = [
            _thread_olt_jan(),
            _thread_olt_sep(),
            _thread_snacks(),
            _thread_dns(),
        ]
        # worth_indexing: jan, sep, dns (snacks отброшен как шум)
        # superseded: jan (заменён sep)
        # итог: sep, dns из 4 тредов -> 0.5
        assert f.indexable_fraction(threads) == pytest.approx(0.5)

    def test_empty_channel_returns_zero(self):
        f = ThreadNoiseFilter()
        assert f.indexable_fraction([]) == 0.0


@pytest.mark.integration
class TestThreadNoiseFilterIntegration:
    """Требует реальной выгрузки истории канала Slack (пагинация API,
    сотни тысяч сообщений) — не выполняется в юнит-тестах курса."""

    def test_full_channel_history_from_slack_export(self):
        pytest.skip("Требует настроенного Slack-коннектора и выгрузки канала")
