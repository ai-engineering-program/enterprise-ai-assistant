"""Tests for app/context/context_assembler.py (Lesson 6.3, module M6 — course finale).

ContextAssembler composes TokenBudgetAllocator (6.1) and ContextPrioritizer
(6.2) — both must already be implemented for these tests to pass. If they
are still skeletons, go back to lessons 6.1 and 6.2 first: this file tests
integration, not the underlying budgeting/prioritization logic itself.

All tests use a deterministic 1-char-per-token counter (`len`) for both
the allocator and the prioritizer, so budgets are easy to verify by hand.

Run unit tests only:
    pytest tests/test_context_assembler.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.context_assembler import (
    AssembledPrompt,
    CategoryContent,
    ContextAssembler,
)
from app.context.context_prioritizer import ContextPrioritizer, PriorityItem
from app.context.token_budget_allocator import BudgetCategory, TokenBudgetAllocator


def _item(item_id: str, score: float, length: int, protected: bool = False) -> PriorityItem:
    return PriorityItem(item_id=item_id, text=item_id[0] * length, score=score, protected=protected)


def _build(weights: dict[BudgetCategory, float], window: int = 100) -> ContextAssembler:
    allocator = TokenBudgetAllocator(
        context_window_tokens=window,
        reserved_output_tokens=0,
        category_weights=weights,
        token_counter=len,
    )
    prioritizer = ContextPrioritizer(token_counter=len)
    return ContextAssembler(allocator, prioritizer, separator="")


# ---------------------------------------------------------------------------
# everything fits — only the final cross-category arrangement matters
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoTruncationNeeded:
    def test_categories_within_budget_are_not_sent_through_prioritizer(self):
        assembler = _build({BudgetCategory.SYSTEM_PROMPT: 0.2, BudgetCategory.RETRIEVED_CONTEXT: 0.8})

        system = _item("SYS", score=1.0, length=10, protected=True)
        doc_a = _item("A", score=0.9, length=30)
        doc_b = _item("B", score=0.5, length=30)

        result = assembler.assemble([
            CategoryContent(BudgetCategory.SYSTEM_PROMPT, [system]),
            CategoryContent(BudgetCategory.RETRIEVED_CONTEXT, [doc_a, doc_b]),
        ])

        assert isinstance(result, AssembledPrompt)
        assert result.trace.budget_report.categories_needing_truncation == []
        # No category needed per-category trimming -> category_results stays empty.
        assert result.trace.category_results == {}

    def test_final_arrangement_is_edge_weighted_across_categories(self):
        assembler = _build({BudgetCategory.SYSTEM_PROMPT: 0.2, BudgetCategory.RETRIEVED_CONTEXT: 0.8})

        system = _item("SYS", score=1.0, length=10, protected=True)
        doc_a = _item("A", score=0.9, length=30)
        doc_b = _item("B", score=0.5, length=30)

        result = assembler.assemble([
            CategoryContent(BudgetCategory.SYSTEM_PROMPT, [system]),
            CategoryContent(BudgetCategory.RETRIEVED_CONTEXT, [doc_a, doc_b]),
        ])

        # Descending by score: SYS(1.0), A(0.9), B(0.5) -> edge-weighted
        # arrangement for 3 items: [SYS, B, A].
        kept_ids = [i.item_id for i in result.trace.final_result.kept]
        assert kept_ids == ["SYS", "B", "A"]
        assert result.trace.final_result.dropped == []
        assert result.trace.all_dropped == []
        assert result.text == "S" * 10 + "B" * 30 + "A" * 30


# ---------------------------------------------------------------------------
# one category overflows its own budget and must be trimmed first
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerCategoryTruncation:
    def test_overflowing_category_is_trimmed_before_final_stage(self):
        assembler = _build({BudgetCategory.SYSTEM_PROMPT: 0.2, BudgetCategory.RETRIEVED_CONTEXT: 0.8})

        system = _item("SYS", score=1.0, length=10, protected=True)
        doc_a = _item("A", score=0.9, length=40)
        doc_b = _item("B", score=0.5, length=40)
        doc_c = _item("C", score=0.2, length=40)  # 3x40=120 > 80 allocated

        result = assembler.assemble([
            CategoryContent(BudgetCategory.SYSTEM_PROMPT, [system]),
            CategoryContent(BudgetCategory.RETRIEVED_CONTEXT, [doc_a, doc_b, doc_c]),
        ])

        assert result.trace.budget_report.categories_needing_truncation == [
            BudgetCategory.RETRIEVED_CONTEXT
        ]
        # SYSTEM_PROMPT fit its own budget -> never went through the prioritizer.
        assert BudgetCategory.SYSTEM_PROMPT not in result.trace.category_results

        retrieved_result = result.trace.category_results[BudgetCategory.RETRIEVED_CONTEXT]
        assert [i.item_id for i in retrieved_result.dropped] == ["C"]

        # Final cross-category pass only re-arranges what survived per-category
        # trimming (SYS, A, B) — nothing new should be dropped at this stage.
        final_kept_ids = [i.item_id for i in result.trace.final_result.kept]
        assert set(final_kept_ids) == {"SYS", "A", "B"}
        assert result.trace.final_result.dropped == []

        assert [i.item_id for i in result.trace.all_dropped] == ["C"]

    def test_protected_item_survives_even_in_overflowing_category(self):
        assembler = _build({BudgetCategory.RETRIEVED_CONTEXT: 1.0})

        protected = _item("PROT", score=0.1, length=50, protected=True)
        doc_x = _item("X", score=0.9, length=50)
        doc_y = _item("Y", score=0.4, length=10)  # pushes category over its 100-token budget

        result = assembler.assemble([
            CategoryContent(BudgetCategory.RETRIEVED_CONTEXT, [protected, doc_x, doc_y]),
        ])

        all_dropped_ids = [i.item_id for i in result.trace.all_dropped]
        assert "PROT" not in all_dropped_ids
        assert "Y" in all_dropped_ids


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_category_contents_returns_empty_prompt(self):
        assembler = _build({BudgetCategory.RETRIEVED_CONTEXT: 1.0})

        result = assembler.assemble([])

        assert result.text == ""
        assert result.trace.category_results == {}
        assert result.trace.all_dropped == []
        assert result.trace.final_result.kept == []


# ---------------------------------------------------------------------------
# integration — placeholder, requires the full M1-M6 pipeline wired end to end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContextAssemblerIntegration:
    """Tests that exercise the full M1-M6 pipeline end to end.

    Skip with: pytest -m "not integration"
    """

    def test_with_real_pipeline_output(self):
        pytest.skip("Requires a fully wired app/context/ pipeline — run manually")
