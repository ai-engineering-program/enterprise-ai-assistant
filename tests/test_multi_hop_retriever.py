"""Tests for app/context/multi_hop_retriever.py (Lesson 3.2, module M3).

MultiHopRetriever is a pure orchestrator — retrieve_fn, extract_fn,
reformulate_fn and is_answer_fn are all plain Python callables supplied by
the test, so every test here is a unit test with no external services
involved (no vector DB, no LLM). The single integration-marked test is a
placeholder for wiring real retrieval/extraction callbacks in a later
course.

Run:
    pytest tests/test_multi_hop_retriever.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.multi_hop_retriever import (
    HopRecord,
    MultiHopResult,
    MultiHopRetriever,
    StopReason,
)


# Fixtures modeling the Kovtun -> Smirnova incident from the lesson.

def _kovtun_retrieve(query: str) -> str:
    if "Ковтун" in query:
        return "Протокол Q1: сегмент B2B передан Смирновой Т.В."
    return "Зона ответственности: B2B, владелец Смирнова Т.В."


def _kovtun_extract(query: str, retrieved: str):
    return "Смирнова Т.В." if "Смирнова" in retrieved else None


def _kovtun_reformulate(original_query: str, entity) -> str:
    return f"зона ответственности {entity}, сегмент B2B"


def _kovtun_is_answer(original_query: str, retrieved: str, entity) -> bool:
    return "владелец" in retrieved


@pytest.mark.unit
class TestMultiHopRetrieverAnswerFound:
    def test_stops_at_hop_2_when_answer_found(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
            max_hops=4,
        )

        result = retriever.run("кто отвечает за сегмент Ковтуна?")

        assert isinstance(result, MultiHopResult)
        assert result.stop_reason is StopReason.ANSWER_FOUND
        assert result.final_entity == "Смирнова Т.В."
        assert len(result.hops) == 2

    def test_hop_records_have_expected_order_and_queries(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
            max_hops=4,
        )

        result = retriever.run("кто отвечает за сегмент Ковтуна?")

        assert [hop.hop_index for hop in result.hops] == [0, 1]
        first, second = result.hops
        assert isinstance(first, HopRecord)
        assert "Ковтун" in first.query
        assert second.query == "зона ответственности Смирнова Т.В., сегмент B2B"


@pytest.mark.unit
class TestMultiHopRetrieverMaxHops:
    def test_stops_at_max_hops_when_answer_never_found(self):
        # is_answer_fn always False, extract_fn always returns a distinct
        # new entity each time -> the loop never converges on its own and
        # must be cut off by the max_hops budget (hop explosion guard).
        counter = {"n": 0}

        def never_retrieve(query: str) -> str:
            counter["n"] += 1
            return f"промежуточный документ #{counter['n']}"

        def always_new_entity(query: str, retrieved: str):
            return f"сущность-{counter['n']}"

        def always_reformulate(original_query: str, entity) -> str:
            return f"уточнить {entity}"

        def never_answer(original_query: str, retrieved: str, entity) -> bool:
            return False

        retriever = MultiHopRetriever(
            retrieve_fn=never_retrieve,
            extract_fn=always_new_entity,
            reformulate_fn=always_reformulate,
            is_answer_fn=never_answer,
            max_hops=3,
        )

        result = retriever.run("вопрос без естественного конца цепочки")

        assert result.stop_reason is StopReason.MAX_HOPS_REACHED
        assert len(result.hops) == 3


@pytest.mark.unit
class TestMultiHopRetrieverNoNewInformation:
    def test_stops_when_extracted_entity_repeats(self):
        # Hop 1 and hop 2 both extract the same entity -> the chain is
        # spinning in place and must stop with NO_NEW_INFORMATION rather
        # than silently continuing (or running to max_hops).
        def stuck_retrieve(query: str) -> str:
            return "документ с именем Смирнова Т.В."

        def stuck_extract(query: str, retrieved: str):
            return "Смирнова Т.В."

        def stuck_reformulate(original_query: str, entity) -> str:
            return f"ещё раз про {entity}"

        def stuck_is_answer(original_query: str, retrieved: str, entity) -> bool:
            return False

        retriever = MultiHopRetriever(
            retrieve_fn=stuck_retrieve,
            extract_fn=stuck_extract,
            reformulate_fn=stuck_reformulate,
            is_answer_fn=stuck_is_answer,
            max_hops=5,
        )

        result = retriever.run("запрос, который зацикливается на одной сущности")

        assert result.stop_reason is StopReason.NO_NEW_INFORMATION
        # Must stop well before the max_hops budget is exhausted.
        assert len(result.hops) < 5


@pytest.mark.unit
class TestHasNewInformation:
    def test_first_hop_always_counts_as_progress(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
        )
        current = HopRecord(
            hop_index=0, query="q", retrieved_text="text", extracted_entity="X"
        )

        assert retriever._has_new_information(None, current) is True

    def test_same_entity_twice_is_not_new_information(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
        )
        previous = HopRecord(
            hop_index=0, query="q1", retrieved_text="text A", extracted_entity="X"
        )
        current = HopRecord(
            hop_index=1, query="q2", retrieved_text="text B", extracted_entity="X"
        )

        assert retriever._has_new_information(previous, current) is False

    def test_same_retrieved_text_twice_is_not_new_information(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
        )
        previous = HopRecord(
            hop_index=0, query="q1", retrieved_text="тот же текст", extracted_entity="X"
        )
        current = HopRecord(
            hop_index=1, query="q2", retrieved_text="тот же текст", extracted_entity="Y"
        )

        assert retriever._has_new_information(previous, current) is False

    def test_different_entity_and_text_is_new_information(self):
        retriever = MultiHopRetriever(
            retrieve_fn=_kovtun_retrieve,
            extract_fn=_kovtun_extract,
            reformulate_fn=_kovtun_reformulate,
            is_answer_fn=_kovtun_is_answer,
        )
        previous = HopRecord(
            hop_index=0, query="q1", retrieved_text="текст A", extracted_entity="X"
        )
        current = HopRecord(
            hop_index=1, query="q2", retrieved_text="текст B", extracted_entity="Y"
        )

        assert retriever._has_new_information(previous, current) is True


@pytest.mark.integration
class TestMultiHopRetrieverRealBackends:
    """Placeholder for a MultiHopRetriever wired to a real vector store and
    an LLM- or NER-based extract_fn.

    Out of scope for this lesson (see class docstring in
    app/context/multi_hop_retriever.py) — kept here so the test suite
    already has a slot for it once real retrieval/extraction backends are
    introduced later in the course.
    """

    def test_real_backends_not_covered_by_this_lesson(self):
        pytest.skip(
            "Real retrieval/extraction backends are out of scope for "
            "lesson 3.2 — requires a running vector store and is not "
            "exercised here."
        )
