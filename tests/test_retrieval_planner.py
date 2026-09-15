"""Tests for app/context/retrieval_planner.py (Lesson 3.4, module M3).

RetrievalPlanner composes three already-implemented M3 components —
QueryDecomposer (3.1), MultiHopRetriever (3.2) and SourceSelector (3.3) —
so these tests exercise real instances of all three, wired the same way a
production conveyor would. No vector store or LLM is involved: QueryDecomposer
and SourceSelector are pure rule-based logic, and MultiHopRetriever is driven
by plain Python callbacks (the same Kovtun -> Smirnova fixtures used in the
lesson 3.2 test suite). Every test here is therefore a unit test.

Run:
    pytest tests/test_retrieval_planner.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.multi_hop_retriever import MultiHopResult, MultiHopRetriever, StopReason
from app.context.query_decomposer import QueryDecomposer
from app.context.retrieval_planner import (
    RetrievalBranch,
    RetrievalPlan,
    RetrievalPlanner,
    RetrievalStrategy,
)
from app.context.source_registry import SourceType
from app.context.source_selector import SourceSelector


# Running examples reused verbatim from lessons 3.1-3.3.
TICKET_QUERY = "Какой сейчас статус тикета SUPPORT-8842?"
CONJUNCTIVE_QUERY = "Какой тариф у клиента и почему списание больше обычного?"
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


def _make_planner() -> RetrievalPlanner:
    decomposer = QueryDecomposer()
    source_selector = SourceSelector()
    multi_hop_retriever = MultiHopRetriever(
        retrieve_fn=_kovtun_retrieve,
        extract_fn=_kovtun_extract,
        reformulate_fn=_kovtun_reformulate,
        is_answer_fn=_kovtun_is_answer,
        max_hops=4,
    )
    return RetrievalPlanner(
        decomposer=decomposer,
        multi_hop_retriever=multi_hop_retriever,
        source_selector=source_selector,
    )


@pytest.mark.unit
class TestClassifyStrategy:
    def test_atomic_ticket_query_is_simple(self):
        planner = _make_planner()
        assert planner.classify_strategy(TICKET_QUERY) is RetrievalStrategy.SIMPLE

    def test_conjunctive_query_is_decompose(self):
        planner = _make_planner()
        assert planner.classify_strategy(CONJUNCTIVE_QUERY) is RetrievalStrategy.DECOMPOSE

    def test_atomic_query_with_chain_marker_is_multi_hop(self):
        planner = _make_planner()
        assert planner.classify_strategy(KOVTUN_QUERY) is RetrievalStrategy.MULTI_HOP

    def test_decompose_wins_even_if_chain_marker_also_present(self):
        # Conjunctive marker is checked via QueryDecomposer.classify_pattern
        # before the planner's own chain-reference markers are considered.
        planner = _make_planner()
        query = "Какой тариф у бывшего клиента и почему списание больше обычного?"
        assert planner.classify_strategy(query) is RetrievalStrategy.DECOMPOSE


@pytest.mark.unit
class TestBuildPlanSimple:
    def test_simple_plan_has_single_branch(self):
        planner = _make_planner()
        plan = planner.build_plan(TICKET_QUERY)

        assert isinstance(plan, RetrievalPlan)
        assert plan.strategy is RetrievalStrategy.SIMPLE
        assert plan.escalated_from is None
        assert len(plan.branches) == 1

        branch = plan.branches[0]
        assert isinstance(branch, RetrievalBranch)
        assert branch.query == TICKET_QUERY
        assert branch.selected_sources == [SourceType.JIRA]
        assert branch.requires_multi_hop is False


@pytest.mark.unit
class TestBuildPlanDecompose:
    def test_decompose_plan_has_one_branch_per_sub_query(self):
        planner = _make_planner()
        plan = planner.build_plan(CONJUNCTIVE_QUERY)

        assert plan.strategy is RetrievalStrategy.DECOMPOSE
        assert len(plan.branches) == 2
        assert all(not b.requires_multi_hop for b in plan.branches)
        assert all(isinstance(b.selected_sources, list) for b in plan.branches)
        # Each branch's query must be a genuine sub-query, not the original.
        assert all(b.query != CONJUNCTIVE_QUERY for b in plan.branches)


@pytest.mark.unit
class TestBuildPlanMultiHop:
    def test_multi_hop_plan_has_single_branch_marked_for_multi_hop(self):
        planner = _make_planner()
        plan = planner.build_plan(KOVTUN_QUERY)

        assert plan.strategy is RetrievalStrategy.MULTI_HOP
        assert len(plan.branches) == 1

        branch = plan.branches[0]
        assert branch.query == KOVTUN_QUERY
        assert branch.requires_multi_hop is True


@pytest.mark.unit
class TestEscalate:
    def test_escalate_simple_moves_to_decompose(self):
        planner = _make_planner()
        plan = planner.build_plan(TICKET_QUERY)

        escalated = planner.escalate(plan)

        assert escalated.strategy is RetrievalStrategy.DECOMPOSE
        assert escalated.escalated_from is RetrievalStrategy.SIMPLE
        assert escalated.original_query == TICKET_QUERY

    def test_escalate_decompose_moves_to_multi_hop(self):
        planner = _make_planner()
        plan = planner.build_plan(CONJUNCTIVE_QUERY)

        escalated = planner.escalate(plan)

        assert escalated.strategy is RetrievalStrategy.MULTI_HOP
        assert escalated.escalated_from is RetrievalStrategy.DECOMPOSE
        assert len(escalated.branches) == 1
        assert escalated.branches[0].requires_multi_hop is True

    def test_escalate_multi_hop_stays_at_multi_hop(self):
        # MULTI_HOP is already the heaviest rung of the ladder — there is
        # nothing left to escalate to.
        planner = _make_planner()
        plan = planner.build_plan(KOVTUN_QUERY)

        escalated = planner.escalate(plan)

        assert escalated.strategy is RetrievalStrategy.MULTI_HOP

    def test_escalation_never_skips_a_rung(self):
        planner = _make_planner()
        plan = planner.build_plan(TICKET_QUERY)  # SIMPLE
        once = planner.escalate(plan)  # -> DECOMPOSE
        twice = planner.escalate(once)  # -> MULTI_HOP

        assert once.strategy is RetrievalStrategy.DECOMPOSE
        assert twice.strategy is RetrievalStrategy.MULTI_HOP


@pytest.mark.unit
class TestRunBranch:
    def test_non_multi_hop_branch_returns_none(self):
        planner = _make_planner()
        plan = planner.build_plan(TICKET_QUERY)

        assert planner.run_branch(plan.branches[0]) is None

    def test_multi_hop_branch_delegates_to_multi_hop_retriever(self):
        planner = _make_planner()
        plan = planner.build_plan(KOVTUN_QUERY)

        result = planner.run_branch(plan.branches[0])

        assert isinstance(result, MultiHopResult)
        assert result.stop_reason is StopReason.ANSWER_FOUND
        assert result.final_entity == "Смирнова Т.В."


@pytest.mark.integration
class TestRetrievalPlannerIntegration:
    """Требует реальных коннекторов Confluence/Jira/Slack и реального
    LLM-бэкенда для extract_fn/reformulate_fn — не выполняется в
    юнит-тестах курса."""

    def test_full_pipeline_against_live_connectors(self):
        pytest.skip(
            "Требует настроенных коннекторов и реального LLM-бэкенда для "
            "MultiHopRetriever, чтобы прогнать RetrievalPlanner end-to-end"
        )
