"""Tests for app/context/long_term_memory.py (Lesson 5.2, module M5).

Every test injects a fake, fully controllable clock function instead of the
real datetime.now(timezone.utc) — the same dependency-injection principle as
ClockFn in SessionMemory (5.1). This makes fact-version ordering deterministic
and testable without a single real sleep() call.

Run:
    pytest tests/test_long_term_memory.py -v -m unit
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.context.long_term_memory import Fact, LongTermMemoryStore


class FakeClock:
    """Controllable clock: starts at a fixed instant, advances only when
    .advance() is called explicitly — no wall-clock time involved."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


@pytest.mark.unit
class TestIsExtractionWorthy:
    def test_short_message_is_not_worthy(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.is_extraction_worthy("окей супер") is False

    def test_exact_acknowledgement_phrase_is_not_worthy(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.is_extraction_worthy("Спасибо") is False
        assert store.is_extraction_worthy("  хорошо  ") is False

    def test_message_with_durable_keyword_is_worthy(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.is_extraction_worthy("хочу перейти на тариф Премиум") is True

    def test_message_without_durable_signal_is_not_worthy(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.is_extraction_worthy("а можно побыстрее с ответом") is False

    def test_long_message_starting_with_acknowledgement_word_is_not_rejected_by_substring(self):
        store = LongTermMemoryStore(clock=FakeClock())
        # "хорошо" встречается как отдельное слово внутри длинной фразы, но
        # это НЕ точное совпадение со всей репликой, поэтому решает
        # keyword-эвристика, а не список подтверждений.
        text = "хорошо, мой тариф теперь Премиум"
        assert store.is_extraction_worthy(text) is True

    def test_empty_string_is_not_worthy(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.is_extraction_worthy("") is False


@pytest.mark.unit
class TestRememberFact:
    def test_remember_fact_returns_fact_with_given_values(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        fact = store.remember_fact("marina", "tariff", "Стандарт", session_id="s1")

        assert isinstance(fact, Fact)
        assert fact.user_id == "marina"
        assert fact.fact_key == "tariff"
        assert fact.value == "Стандарт"
        assert fact.source_session_id == "s1"
        assert fact.recorded_at == clock()

    def test_remember_fact_does_not_overwrite_previous_version(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт", session_id="s1")
        clock.advance(60 * 60 * 24 * 30)
        store.remember_fact("marina", "tariff", "Премиум", session_id="s2")

        history = store.get_fact_history("marina", "tariff")
        assert len(history) == 2
        assert history[0].value == "Стандарт"
        assert history[1].value == "Премиум"

    def test_remember_fact_for_different_keys_does_not_collide(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        store.remember_fact("marina", "payment_method", "приложение банка")

        assert len(store.get_fact_history("marina", "tariff")) == 1
        assert len(store.get_fact_history("marina", "payment_method")) == 1

    def test_remember_fact_for_different_users_does_not_collide(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        store.remember_fact("oleg", "tariff", "Премиум")

        assert store.get_fact_history("marina", "tariff")[0].value == "Стандарт"
        assert store.get_fact_history("oleg", "tariff")[0].value == "Премиум"


@pytest.mark.unit
class TestGetFacts:
    def test_get_facts_returns_most_recent_version(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        clock.advance(3600)
        store.remember_fact("marina", "tariff", "Премиум")

        facts = store.get_facts("marina")

        assert facts["tariff"].value == "Премиум"

    def test_get_facts_returns_empty_dict_for_unknown_user(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.get_facts("never-seen") == {}

    def test_get_facts_combines_multiple_keys(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        store.remember_fact("marina", "payment_method", "приложение банка")

        facts = store.get_facts("marina")

        assert set(facts.keys()) == {"tariff", "payment_method"}
        assert facts["tariff"].value == "Стандарт"
        assert facts["payment_method"].value == "приложение банка"

    def test_get_facts_filters_by_fact_key(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        store.remember_fact("marina", "payment_method", "приложение банка")

        facts = store.get_facts("marina", fact_key="tariff")

        assert set(facts.keys()) == {"tariff"}


@pytest.mark.unit
class TestGetFactHistory:
    def test_history_preserves_chronological_order(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)

        store.remember_fact("marina", "tariff", "Стандарт")
        clock.advance(100)
        store.remember_fact("marina", "tariff", "Плюс")
        clock.advance(100)
        store.remember_fact("marina", "tariff", "Премиум")

        history = store.get_fact_history("marina", "tariff")

        assert [f.value for f in history] == ["Стандарт", "Плюс", "Премиум"]

    def test_history_for_unknown_fact_is_empty_list(self):
        store = LongTermMemoryStore(clock=FakeClock())
        assert store.get_fact_history("marina", "tariff") == []

    def test_history_returns_a_copy_not_internal_list(self):
        clock = FakeClock()
        store = LongTermMemoryStore(clock=clock)
        store.remember_fact("marina", "tariff", "Стандарт")

        history = store.get_fact_history("marina", "tariff")
        history.append(Fact(user_id="marina", fact_key="tariff", value="Подделка"))

        assert len(store.get_fact_history("marina", "tariff")) == 1


@pytest.mark.integration
class TestLongTermMemoryIntegration:
    """Требуют реального внешнего backend'а (например, Redis/Postgres) — не
    выполняются по умолчанию."""

    def test_shared_backend_visible_across_instances(self):
        pytest.skip("Требует запущенного внешнего хранилища — запускать вручную")
