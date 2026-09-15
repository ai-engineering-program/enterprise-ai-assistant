"""Tests for app/context/retrieval_tracer.py (Lesson 3.5, module M3 closer).

RetrievalTracer wraps a real RetrievalPlanner (3.4) built from real
QueryDecomposer (3.1), MultiHopRetriever (3.2) and SourceSelector (3.3)
instances — the same wiring used in test_retrieval_planner.py. No vector
store or LLM is involved, so every test here is a unit test.

Run:
    pytest tests/test_retrieval_tracer.py -v -m unit
"""
from __future__ import annotations

import json

import pytest

from app.context.multi_hop_retriever import MultiHopRetriever, StopReason
from app.context.query_decomposer import QueryDecomposer
from app.context.retrieval_planner import RetrievalPlan, RetrievalPlanner, RetrievalStrategy
from app.context.retrieval_tracer import RetrievalTrace, RetrievalTracer
from app.context.source_registry import SourceType
from app.context.source_selector import SourceSelector


TICKET_QUERY = "Какой сейчас статус тикета SUPPORT-8842?"
AMBIGUOUS_QUERY = "Какой лимит трафика в тарифе клиента?"
KOVTUN_QUERY = "Кто сейчас отвечает за сегмент, которым раньше руководил Ковтун?"


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


def _make_tracer(is_sufficient_fn=None, max_escalations: int = 2) -> RetrievalTracer:
    decomposer = QueryDecomposer()
    source_selector = SourceSelector()
    multi_hop_retriever = MultiHopRetriever(
        retrieve_fn=_kovtun_retrieve,
        extract_fn=_kovtun_extract,
        reformulate_fn=_kovtun_reformulate,
        is_answer_fn=_kovtun_is_answer,
        max_hops=4,
    )
    planner = RetrievalPlanner(
        decomposer=decomposer,
        multi_hop_retriever=multi_hop_retriever,
        source_selector=source_selector,
    )
    return RetrievalTracer(
        planner=planner,
        source_selector=source_selector,
        is_sufficient_fn=is_sufficient_fn,
        max_escalations=max_escalations,
    )


@pytest.mark.unit
class TestSimpleQueryTrace:
    def test_records_single_source_event_and_no_multi_hop(self):
        tracer = _make_tracer()
        plan, trace = tracer.run(TICKET_QUERY)

        assert isinstance(plan, RetrievalPlan)
        assert plan.strategy is RetrievalStrategy.SIMPLE
        assert isinstance(trace, RetrievalTrace)
        assert trace.strategy_history == [RetrievalStrategy.SIMPLE]
        assert trace.escalations == []
        assert len(trace.source_events) == 1
        assert trace.source_events[0].selected_sources == [SourceType.JIRA]
        assert trace.source_events[0].used_fallback is False
        assert trace.multi_hop_events == []
        assert trace.final_plan is plan


@pytest.mark.unit
class TestMultiHopQueryTrace:
    def test_records_multi_hop_event_with_stop_reason_and_final_entity(self):
        tracer = _make_tracer()
        plan, trace = tracer.run(KOVTUN_QUERY)

        assert plan.strategy is RetrievalStrategy.MULTI_HOP
        assert len(trace.multi_hop_events) == 1

        event = trace.multi_hop_events[0]
        assert event.branch_query == KOVTUN_QUERY
        assert event.stop_reason is StopReason.ANSWER_FOUND
        assert event.final_entity == "Смирнова Т.В."
        assert len(event.hops) >= 1


@pytest.mark.unit
class TestEscalationTrace:
    def test_escalates_once_when_is_sufficient_fn_rejects_simple(self):
        calls = {"count": 0}

        def is_sufficient(plan: RetrievalPlan) -> bool:
            calls["count"] += 1
            return plan.strategy is not RetrievalStrategy.SIMPLE

        tracer = _make_tracer(is_sufficient_fn=is_sufficient, max_escalations=2)
        plan, trace = tracer.run(TICKET_QUERY)

        assert plan.strategy is RetrievalStrategy.DECOMPOSE
        assert trace.strategy_history == [
            RetrievalStrategy.SIMPLE,
            RetrievalStrategy.DECOMPOSE,
        ]
        assert len(trace.escalations) == 1
        assert trace.escalations[0].from_strategy is RetrievalStrategy.SIMPLE
        assert trace.escalations[0].to_strategy is RetrievalStrategy.DECOMPOSE
        assert trace.final_plan.strategy is RetrievalStrategy.DECOMPOSE

    def test_max_escalations_bounds_the_loop(self):
        tracer = _make_tracer(is_sufficient_fn=lambda plan: False, max_escalations=1)
        plan, trace = tracer.run(TICKET_QUERY)

        # Never satisfied, but capped at one escalation attempt.
        assert len(trace.escalations) == 1
        assert plan.strategy is RetrievalStrategy.DECOMPOSE

    def test_escalation_stops_gracefully_at_multi_hop_ceiling(self):
        tracer = _make_tracer(is_sufficient_fn=lambda plan: False, max_escalations=10)
        plan, trace = tracer.run(TICKET_QUERY)

        # Ladder is SIMPLE -> DECOMPOSE -> MULTI_HOP: only two real
        # escalations are possible regardless of max_escalations.
        assert plan.strategy is RetrievalStrategy.MULTI_HOP
        assert len(trace.escalations) == 2


@pytest.mark.unit
class TestFlagRisks:
    def test_flags_fallback_usage(self):
        tracer = _make_tracer()
        _, trace = tracer.run(AMBIGUOUS_QUERY)

        risks = trace.flag_risks()

        assert any("fallback" in risk for risk in risks)

    def test_no_risks_for_clean_ticket_query(self):
        tracer = _make_tracer()
        _, trace = tracer.run(TICKET_QUERY)

        assert trace.flag_risks() == []

    def test_flags_escalation_history(self):
        tracer = _make_tracer(is_sufficient_fn=lambda plan: False, max_escalations=1)
        _, trace = tracer.run(TICKET_QUERY)

        risks = trace.flag_risks()

        assert any("эскалирован" in risk for risk in risks)


@pytest.mark.unit
class TestAsDict:
    def test_as_dict_is_json_serializable(self):
        tracer = _make_tracer()
        _, trace = tracer.run(KOVTUN_QUERY)

        payload = trace.as_dict()
        serialized = json.dumps(payload, ensure_ascii=False)

        assert isinstance(serialized, str)
        assert payload["original_query"] == KOVTUN_QUERY
        assert payload["final_strategy"] == RetrievalStrategy.MULTI_HOP.value
        assert payload["multi_hop_events"][0]["stop_reason"] == StopReason.ANSWER_FOUND.value


@pytest.mark.integration
class TestRetrievalTracerIntegration:
    """Требует реального лог-хранилища (например, для проверки retention
    и маскирования чувствительных данных, урок 3.5) — не выполняется в
    юнит-тестах курса."""

    def test_trace_persisted_to_real_log_backend(self):
        pytest.skip(
            "Требует настроенного лог-хранилища для проверки записи, "
            "retention и маскирования трасс RetrievalTracer"
        )
