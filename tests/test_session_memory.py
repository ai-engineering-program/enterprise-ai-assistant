"""Tests for app/context/session_memory.py (Lesson 5.1, module M5, opening).

Every test injects a fake, fully controllable clock function instead of the
real datetime.now(timezone.utc) — the same dependency-injection principle as
summarize_fn in ContextSummarizer (4.1) or strategy_fn in
CompressionEvalHarness (4.5). This makes TTL expiry deterministic and
testable without a single real sleep() call.

Run:
    pytest tests/test_session_memory.py -v -m unit
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.context.session_memory import SessionMemory, Turn


def _char_len(text: str) -> int:
    """Deterministic token counter for tests: 1 character == 1 "token"."""
    return len(text)


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
class TestIsExpired:
    def test_unknown_session_is_expired(self):
        memory = SessionMemory(clock=FakeClock())
        assert memory.is_expired("never-seen") is True

    def test_fresh_session_is_not_expired(self):
        clock = FakeClock()
        memory = SessionMemory(ttl_seconds=1800, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "лицевой счёт 774521-08")

        assert memory.is_expired("s1") is False

    def test_session_expires_after_ttl_elapsed(self):
        clock = FakeClock()
        memory = SessionMemory(ttl_seconds=1800, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "лицевой счёт 774521-08")

        clock.advance(1801)

        assert memory.is_expired("s1") is True

    def test_session_not_expired_exactly_at_ttl_boundary(self):
        clock = FakeClock()
        memory = SessionMemory(ttl_seconds=1800, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "счёт 774521-08")

        clock.advance(1800)

        assert memory.is_expired("s1") is False


@pytest.mark.unit
class TestAppendTurn:
    def test_appended_turn_is_stored(self):
        clock = FakeClock()
        memory = SessionMemory(clock=clock, token_counter=_char_len)

        memory.append_turn("s1", "user", "счёт 774521-08, долг 3200 рублей")

        history = memory.get_history("s1")
        assert len(history) == 1
        assert isinstance(history[0], Turn)
        assert history[0].role == "user"
        assert "774521-08" in history[0].content

    def test_turns_preserve_chronological_order(self):
        clock = FakeClock()
        memory = SessionMemory(clock=clock, token_counter=_char_len)

        memory.append_turn("s1", "user", "первое сообщение")
        memory.append_turn("s1", "assistant", "первый ответ")
        memory.append_turn("s1", "user", "второе сообщение")

        history = memory.get_history("s1")
        assert [t.content for t in history] == [
            "первое сообщение",
            "первый ответ",
            "второе сообщение",
        ]

    def test_appending_to_expired_session_starts_fresh(self):
        clock = FakeClock()
        memory = SessionMemory(ttl_seconds=1800, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "старый разговор, счёт 111111")

        clock.advance(3600)  # TTL истёк
        memory.append_turn("s1", "user", "новый разговор, счёт 222222")

        history = memory.get_history("s1")
        assert len(history) == 1
        assert "222222" in history[0].content
        assert "111111" not in history[0].content


@pytest.mark.unit
class TestGetHistoryBudget:
    def test_empty_session_returns_empty_history(self):
        memory = SessionMemory(clock=FakeClock(), token_counter=_char_len)
        assert memory.get_history("never-seen") == []

    def test_expired_session_returns_empty_history(self):
        clock = FakeClock()
        memory = SessionMemory(ttl_seconds=100, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "привет")

        clock.advance(200)

        assert memory.get_history("s1") == []

    def test_history_within_budget_returned_in_full(self):
        clock = FakeClock()
        memory = SessionMemory(max_history_tokens=1000, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "короткое сообщение")
        memory.append_turn("s1", "assistant", "короткий ответ")

        history = memory.get_history("s1")

        assert len(history) == 2

    def test_budget_keeps_most_recent_turns_first(self):
        clock = FakeClock()
        memory = SessionMemory(clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "a" * 50)
        memory.append_turn("s1", "assistant", "b" * 50)
        memory.append_turn("s1", "user", "c" * 50)

        history = memory.get_history("s1", max_tokens=120)

        # Бюджета хватает только на две самые свежие реплики (100 токенов),
        # третья по счёту (a*50) не помещается вместе с ними.
        assert [t.content[0] for t in history] == ["b", "c"]

    def test_max_turns_further_limits_already_budgeted_history(self):
        clock = FakeClock()
        memory = SessionMemory(clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "один")
        memory.append_turn("s1", "assistant", "два")
        memory.append_turn("s1", "user", "три")

        history = memory.get_history("s1", max_tokens=1000, max_turns=1)

        assert len(history) == 1
        assert history[0].content == "три"


@pytest.mark.unit
class TestTotalTokensAndClear:
    def test_total_tokens_sums_all_stored_turns_ignoring_budget(self):
        clock = FakeClock()
        memory = SessionMemory(max_history_tokens=5, clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "abcde")
        memory.append_turn("s1", "assistant", "fghij")

        assert memory.total_tokens("s1") == 10

    def test_total_tokens_for_unknown_session_is_zero(self):
        memory = SessionMemory(clock=FakeClock(), token_counter=_char_len)
        assert memory.total_tokens("never-seen") == 0

    def test_clear_session_removes_history(self):
        clock = FakeClock()
        memory = SessionMemory(clock=clock, token_counter=_char_len)
        memory.append_turn("s1", "user", "привет")

        memory.clear_session("s1")

        assert memory.get_history("s1") == []
        assert memory.is_expired("s1") is True

    def test_clear_session_on_unknown_session_is_safe_noop(self):
        memory = SessionMemory(clock=FakeClock(), token_counter=_char_len)
        memory.clear_session("never-seen")  # не должно бросать исключений


@pytest.mark.integration
class TestSessionMemoryIntegration:
    """Требуют реального внешнего backend'а (например, Redis) — не
    выполняются по умолчанию."""

    def test_shared_backend_visible_across_instances(self):
        pytest.skip("Требует запущенного Redis — запускать вручную")
