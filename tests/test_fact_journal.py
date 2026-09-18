"""Tests for app/context/fact_journal.py (Lesson 5.4, module M5).

Every test injects a fake, fully controllable clock function instead of the
real datetime.now(timezone.utc) — same dependency-injection principle as
ClockFn in SessionMemory (5.1), LongTermMemoryStore (5.2) and
EpisodicMemoryStore (5.3). This makes the "which version wins by timestamp"
selection rule deterministic and testable without a single real sleep() call.

Run:
    pytest tests/test_fact_journal.py -v -m unit
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.context.fact_journal import FactJournal, FactVersion


class FakeClock:
    """Controllable clock: starts at a fixed instant, advances only when
    .advance() is called explicitly — no wall-clock time involved."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


DAY = 60 * 60 * 24


@pytest.mark.unit
class TestAppendVersion:
    def test_append_returns_version_with_given_value(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        version = journal.append_version("manager_client_774521", "Игорь Круглов")

        assert isinstance(version, FactVersion)
        assert version.fact_key == "manager_client_774521"
        assert version.value == "Игорь Круглов"
        assert version.oblivion is False
        assert version.recorded_at == clock()

    def test_append_does_not_remove_previous_versions(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Игорь Круглов")
        clock.advance(60 * DAY)
        journal.append_version("manager_client_774521", "Дмитрий Осадчий")

        history = journal.get_history("manager_client_774521")
        assert len(history) == 2
        assert [v.value for v in history] == ["Игорь Круглов", "Дмитрий Осадчий"]

    def test_different_fact_keys_do_not_collide(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Игорь Круглов")
        journal.append_version("tariff_client_774521", "Премиум")

        assert len(journal.get_history("manager_client_774521")) == 1
        assert len(journal.get_history("tariff_client_774521")) == 1


@pytest.mark.unit
class TestGetCurrent:
    def test_unknown_fact_key_returns_none(self):
        journal = FactJournal(clock=FakeClock())
        assert journal.get_current("never-seen") is None

    def test_single_version_is_current(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)
        journal.append_version("manager_client_774521", "Игорь Круглов")

        current = journal.get_current("manager_client_774521")

        assert current is not None
        assert current.value == "Игорь Круглов"

    def test_latest_timestamp_wins_over_earlier_version(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Игорь Круглов")
        clock.advance(60 * DAY)
        journal.append_version("manager_client_774521", "Дмитрий Осадчий")

        current = journal.get_current("manager_client_774521")

        assert current is not None
        assert current.value == "Дмитрий Осадчий"

    def test_two_near_identical_versions_do_not_confuse_selection(self):
        # Регрессия ровно инцидента урока: два семантически почти
        # неотличимых документа (одна и та же карточка клиента с разным
        # именем менеджера) не должны давать неопределённый результат —
        # выбор всегда детерминирован по timestamp, а не по similarity.
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version(
            "manager_client_774521", "Игорь Круглов", source="crm_export_2026_05_12",
        )
        clock.advance(60 * DAY)
        journal.append_version(
            "manager_client_774521", "Дмитрий Осадчий", source="crm_export_2026_07_14",
        )

        for _ in range(5):
            assert (
                journal.get_current("manager_client_774521").value
                == "Дмитрий Осадчий"
            )

    def test_oblivion_marker_as_latest_means_no_current_value(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Дмитрий Осадчий")
        clock.advance(30 * DAY)
        journal.mark_oblivion("manager_client_774521", source="client_churned")

        assert journal.get_current("manager_client_774521") is None

    def test_oblivion_does_not_erase_history(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Дмитрий Осадчий")
        clock.advance(30 * DAY)
        journal.mark_oblivion("manager_client_774521")

        history = journal.get_history("manager_client_774521")
        assert len(history) == 2
        assert history[0].value == "Дмитрий Осадчий"
        assert history[1].oblivion is True

    def test_new_value_after_oblivion_becomes_current_again(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)

        journal.append_version("manager_client_774521", "Дмитрий Осадчий")
        clock.advance(30 * DAY)
        journal.mark_oblivion("manager_client_774521")
        clock.advance(10 * DAY)
        journal.append_version("manager_client_774521", "Елена Присяжная")

        current = journal.get_current("manager_client_774521")
        assert current is not None
        assert current.value == "Елена Присяжная"


@pytest.mark.unit
class TestGetHistory:
    def test_unknown_fact_key_returns_empty_list(self):
        journal = FactJournal(clock=FakeClock())
        assert journal.get_history("never-seen") == []

    def test_returned_list_is_a_copy(self):
        clock = FakeClock()
        journal = FactJournal(clock=clock)
        journal.append_version("manager_client_774521", "Игорь Круглов")

        history = journal.get_history("manager_client_774521")
        history.append(
            FactVersion(fact_key="manager_client_774521", value="Подделка")
        )

        assert len(journal.get_history("manager_client_774521")) == 1


@pytest.mark.integration
class TestFactJournalIntegration:
    """Требуют реального векторного хранилища (Qdrant) с payload-фильтром по
    fact_key — не выполняются по умолчанию."""

    def test_payload_filter_against_real_qdrant(self):
        pytest.skip("Требует запущенного Qdrant — запускать вручную")
