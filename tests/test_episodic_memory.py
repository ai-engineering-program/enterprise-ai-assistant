"""Tests for app/context/episodic_memory.py (Lesson 5.3, module M5).

Every test injects a fake, fully controllable clock function instead of the
real datetime.now(timezone.utc) — the same dependency-injection principle as
ClockFn in SessionMemory (5.1) and LongTermMemoryStore (5.2). This makes
recency-weight ordering deterministic and testable without a single real
sleep() call.

Run:
    pytest tests/test_episodic_memory.py -v -m unit
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.context.episodic_memory import (
    Episode,
    EpisodicMemoryStore,
    OUTCOME_ESCALATED,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
)


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
class TestScoreSalience:
    def test_resolved_without_keywords_is_base_score(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        score = store.score_salience("подтвердила адрес доставки", OUTCOME_RESOLVED)
        assert score == pytest.approx(0.3)

    def test_unresolved_adds_to_base(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        score = store.score_salience("уточнила дату списания", OUTCOME_UNRESOLVED)
        assert score == pytest.approx(0.6)

    def test_escalated_adds_more_than_unresolved(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        score = store.score_salience("передано в техподдержку", OUTCOME_ESCALATED)
        assert score == pytest.approx(0.7)

    def test_each_distinct_keyword_adds_one_tenth(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        text = "жалоба клиента повторно, вопрос остался без решения"
        score = store.score_salience(text, OUTCOME_UNRESOLVED)
        # base 0.3 + unresolved 0.3 + 3 keywords ("жалоб", "повторно",
        # "без решения") * 0.1 = 0.9
        assert score == pytest.approx(0.9)

    def test_repeated_keyword_counts_once_not_per_occurrence(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        text = "жалоба, ещё раз жалоба на то же самое"
        score = store.score_salience(text, OUTCOME_RESOLVED)
        # base 0.3 + ОДНО распознанное ключевое слово ("жалоб"), несмотря
        # на то что оно встречается в тексте дважды.
        assert score == pytest.approx(0.4)

    def test_score_is_capped_at_one(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        text = "недовольство, жалоба, повторно, эскалация без решения"
        score = store.score_salience(text, OUTCOME_ESCALATED)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
class TestRecordEpisode:
    def test_record_episode_returns_episode_with_given_values(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        episode = store.record_episode(
            "marina",
            "жалоба на связь, эскалация, решено заменой кабеля",
            OUTCOME_ESCALATED,
            topic="connection_quality",
            ticket_id="774521",
        )

        assert isinstance(episode, Episode)
        assert episode.user_id == "marina"
        assert episode.outcome == OUTCOME_ESCALATED
        assert episode.topic == "connection_quality"
        assert episode.ticket_id == "774521"
        assert episode.occurred_at == clock()

    def test_salience_is_computed_automatically(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        summary = "жалоба на связь, эскалация"
        episode = store.record_episode("marina", summary, OUTCOME_ESCALATED)

        assert episode.salience == pytest.approx(
            store.score_salience(summary, OUTCOME_ESCALATED)
        )

    def test_record_episode_does_not_remove_previous_episodes(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        store.record_episode("marina", "жалоба на связь, эскалация", OUTCOME_ESCALATED)
        clock.advance(90 * DAY)
        store.record_episode("marina", "уточнила дату списания", OUTCOME_RESOLVED)

        history = store.retrieve_relevant_episodes("marina", top_k=10)
        assert len(history) == 2

    def test_different_users_do_not_collide(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        store.record_episode("marina", "жалоба на связь, эскалация", OUTCOME_ESCALATED)
        store.record_episode("oleg", "уточнил тариф", OUTCOME_RESOLVED)

        assert len(store.retrieve_relevant_episodes("marina", top_k=10)) == 1
        assert len(store.retrieve_relevant_episodes("oleg", top_k=10)) == 1


@pytest.mark.unit
class TestRetrieveRelevantEpisodes:
    def test_unknown_user_returns_empty_list(self):
        store = EpisodicMemoryStore(clock=FakeClock())
        assert store.retrieve_relevant_episodes("never-seen") == []

    def test_topic_filter_excludes_unrelated_episodes(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        store.record_episode(
            "marina", "жалоба на связь, эскалация", OUTCOME_ESCALATED,
            topic="connection_quality",
        )
        store.record_episode(
            "marina", "уточнила дату списания", OUTCOME_RESOLVED, topic="billing",
        )

        result = store.retrieve_relevant_episodes("marina", query_topic="connection_quality")

        assert len(result) == 1
        assert result[0].topic == "connection_quality"

    def test_topic_filter_with_no_match_returns_empty_not_fallback(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        store.record_episode(
            "marina", "уточнила дату списания", OUTCOME_RESOLVED, topic="billing",
        )

        result = store.retrieve_relevant_episodes("marina", query_topic="delivery")

        assert result == []

    def test_high_salience_old_episode_outranks_low_salience_fresh_one(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(
            clock=clock, recency_half_life_days=30.0,
        )

        # Высокая salience (эскалация + ключевые слова), записан давно.
        store.record_episode(
            "marina",
            "жалоба на связь, повторно, эскалация без решения",
            OUTCOME_ESCALATED,
            topic="connection_quality",
        )
        clock.advance(30 * DAY)  # ровно один период полураспада
        # Низкая salience (благополучно решено, без сигналов), записан свежо.
        store.record_episode(
            "marina", "уточнила дату списания", OUTCOME_RESOLVED,
            topic="connection_quality",
        )

        result = store.retrieve_relevant_episodes(
            "marina", query_topic="connection_quality", top_k=1
        )

        assert len(result) == 1
        assert result[0].outcome == OUTCOME_ESCALATED

    def test_top_k_limits_number_of_results(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        for i in range(5):
            store.record_episode(
                "marina", f"обращение номер {i}, эскалация", OUTCOME_ESCALATED,
                topic="connection_quality",
            )
            clock.advance(DAY)

        result = store.retrieve_relevant_episodes(
            "marina", query_topic="connection_quality", top_k=2
        )

        assert len(result) == 2

    def test_results_sorted_by_score_descending(self):
        clock = FakeClock()
        store = EpisodicMemoryStore(clock=clock)

        store.record_episode(
            "marina", "уточнила дату списания", OUTCOME_RESOLVED,
            topic="connection_quality",
        )
        clock.advance(DAY)
        store.record_episode(
            "marina", "жалоба на связь, эскалация без решения", OUTCOME_ESCALATED,
            topic="connection_quality",
        )

        result = store.retrieve_relevant_episodes(
            "marina", query_topic="connection_quality", top_k=2
        )

        assert result[0].outcome == OUTCOME_ESCALATED
        assert result[1].outcome == OUTCOME_RESOLVED


@pytest.mark.integration
class TestEpisodicMemoryIntegration:
    """Требуют реального внешнего backend'а (например, Redis/Postgres) — не
    выполняются по умолчанию."""

    def test_shared_backend_visible_across_instances(self):
        pytest.skip("Требует запущенного внешнего хранилища — запускать вручную")
