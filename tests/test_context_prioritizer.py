"""Tests for app/context/context_prioritizer.py (Lesson 6.2, module M6).

ContextPrioritizer never recomputes relevance/trust scores and never talks
to a real LLM — tests supply a deterministic 1-char-per-token counter so
budgets are easy to compute by hand, independent of estimate_tokens
(lesson 4.1).

Run unit tests only:
    pytest tests/test_context_prioritizer.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.context_prioritizer import (
    ContextPrioritizer,
    PrioritizationResult,
    PriorityItem,
)


def _char_counter(text: str) -> int:
    """Deterministic 1-char-per-token counter."""
    return len(text)


def _item(item_id: str, score: float, length: int = 10, protected: bool = False) -> PriorityItem:
    return PriorityItem(item_id=item_id, text="x" * length, score=score, protected=protected)


# ---------------------------------------------------------------------------
# dropping lowest-priority items until in budget
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDropping:
    def test_all_items_fit_nothing_dropped(self):
        items = [_item("A", 0.9), _item("B", 0.8)]
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize(items, budget_tokens=100)

        assert result.dropped == []
        assert {i.item_id for i in result.kept} == {"A", "B"}
        assert result.kept_tokens == 20

    def test_over_budget_drops_lowest_score_first(self):
        # 5 items x 10 tokens each = 50 tokens total, budget fits only 3.
        items = [
            _item("A", 0.92),
            _item("B", 0.81),
            _item("C", 0.67),
            _item("D", 0.54),
            _item("E", 0.31),
        ]
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize(items, budget_tokens=30)

        kept_ids = {i.item_id for i in result.kept}
        dropped_ids = [i.item_id for i in result.dropped]

        assert kept_ids == {"A", "B", "C"}
        assert dropped_ids == ["D", "E"]  # descending score order among dropped
        assert result.kept_tokens == 30

    def test_dropped_order_is_descending_by_score(self):
        items = [_item("low", 0.1), _item("mid", 0.5), _item("high", 0.9)]
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        # Budget fits only one item (10 tokens).
        result = prioritizer.prioritize(items, budget_tokens=10)

        assert [i.item_id for i in result.dropped] == ["mid", "low"]

    def test_empty_items_returns_empty_result(self):
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize([], budget_tokens=100)

        assert result.kept == []
        assert result.dropped == []
        assert result.kept_tokens == 0
        assert result.budget_tokens == 100


# ---------------------------------------------------------------------------
# protected items
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtectedItems:
    def test_protected_item_never_dropped_even_over_budget(self):
        protected = _item("SYS", score=0.1, length=50, protected=True)
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize([protected], budget_tokens=10)

        assert result.dropped == []
        assert [i.item_id for i in result.kept] == ["SYS"]
        assert result.kept_tokens == 50  # over budget_tokens=10, but kept anyway

    def test_protected_tokens_reduce_room_for_droppable(self):
        protected = _item("SYS", score=0.99, length=20, protected=True)
        low_priority = _item("LOW", score=0.5, length=10)
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        # budget=25: protected already consumes 20, leaving only 5 —
        # not enough for the 10-token droppable item.
        result = prioritizer.prioritize([protected, low_priority], budget_tokens=25)

        kept_ids = {i.item_id for i in result.kept}
        assert "SYS" in kept_ids
        assert "LOW" not in kept_ids
        assert [i.item_id for i in result.dropped] == ["LOW"]

    def test_protected_item_low_score_still_survives_alongside_high_score_droppable(self):
        protected = _item("SYS", score=0.01, length=10, protected=True)
        high_priority = _item("HIGH", score=0.99, length=10)
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize([protected, high_priority], budget_tokens=20)

        assert {i.item_id for i in result.kept} == {"SYS", "HIGH"}
        assert result.dropped == []


# ---------------------------------------------------------------------------
# beginning/end-weighted arrangement of survivors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArrangement:
    def test_arrangement_places_best_at_edges_worst_in_middle(self):
        # A..E by descending score; edge-weighted arrangement is
        # [A, C, E, D, B] — best (A) at the start, 2nd-best (B) at the
        # end, worst (E) buried in the literal middle.
        items = [
            _item("A", 0.92),
            _item("B", 0.81),
            _item("C", 0.67),
            _item("D", 0.54),
            _item("E", 0.31),
        ]
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize(items, budget_tokens=1000)

        assert [i.item_id for i in result.kept] == ["A", "C", "E", "D", "B"]

    def test_arrangement_order_differs_from_pure_score_descending(self):
        items = [_item("A", 0.9), _item("B", 0.7), _item("C", 0.5), _item("D", 0.3)]
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize(items, budget_tokens=1000)
        kept_ids = [i.item_id for i in result.kept]

        assert kept_ids != ["A", "B", "C", "D"]  # not naive descending order
        assert kept_ids[0] == "A"  # best item still opens the block
        assert kept_ids[-1] == "B"  # 2nd-best closes the block, not buried

    def test_arrangement_of_empty_list_is_empty(self):
        prioritizer = ContextPrioritizer(token_counter=_char_counter)

        result = prioritizer.prioritize([], budget_tokens=100)

        assert result.kept == []


# ---------------------------------------------------------------------------
# integration — placeholder, requires a real upstream scorer / live pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContextPrioritizerIntegration:
    """Tests that exercise the full M2-M6 pipeline end to end.

    Skip with: pytest -m "not integration"
    """

    def test_with_real_knowledge_hierarchy_resolver(self):
        pytest.skip("Requires a configured KnowledgeHierarchyResolver — run manually")
