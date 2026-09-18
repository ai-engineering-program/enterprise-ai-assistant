"""Tests for app/context/token_budget_allocator.py (Lesson 6.1, opening M6).

TokenBudgetAllocator never talks to a real LLM or tokenizer — tests supply
their own deterministic token_counter (character count) so they do not
depend on the estimate_tokens heuristic implemented elsewhere (lesson 4.1).

Run unit tests only:
    pytest tests/test_token_budget_allocator.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.token_budget_allocator import (
    DEFAULT_CATEGORY_WEIGHTS,
    BudgetCategory,
    BudgetReport,
    CategoryAllocation,
    ContextComponent,
    TokenBudgetAllocator,
)


def _char_counter(text: str) -> int:
    """Deterministic 1-char-per-token counter — makes budgets easy to
    compute by hand in test assertions, independent of estimate_tokens."""
    return len(text)


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_content_budget_subtracts_reserved_output(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=1000,
            reserved_output_tokens=200,
            token_counter=_char_counter,
        )

        assert allocator.content_budget_tokens == 800

    def test_reserved_output_equal_to_window_raises(self):
        with pytest.raises(ValueError):
            TokenBudgetAllocator(
                context_window_tokens=500,
                reserved_output_tokens=500,
                token_counter=_char_counter,
            )

    def test_reserved_output_larger_than_window_raises(self):
        with pytest.raises(ValueError):
            TokenBudgetAllocator(
                context_window_tokens=500,
                reserved_output_tokens=600,
                token_counter=_char_counter,
            )

    def test_weights_summing_over_one_raise(self):
        bad_weights = dict(DEFAULT_CATEGORY_WEIGHTS)
        bad_weights[BudgetCategory.USER_MESSAGE] += 0.5

        with pytest.raises(ValueError):
            TokenBudgetAllocator(
                context_window_tokens=1000,
                reserved_output_tokens=100,
                category_weights=bad_weights,
                token_counter=_char_counter,
            )

    def test_default_category_weights_sum_to_one(self):
        assert sum(DEFAULT_CATEGORY_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# allocate_category_budgets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllocateCategoryBudgets:
    def test_budgets_match_weights_of_content_budget(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=1100,
            reserved_output_tokens=100,
            category_weights={
                BudgetCategory.RETRIEVED_CONTEXT: 0.5,
                BudgetCategory.USER_MESSAGE: 0.5,
            },
            token_counter=_char_counter,
        )
        # content_budget_tokens == 1000

        budgets = allocator.allocate_category_budgets()

        assert budgets[BudgetCategory.RETRIEVED_CONTEXT] == 500
        assert budgets[BudgetCategory.USER_MESSAGE] == 500

    def test_category_missing_from_weights_is_not_allocated(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=1000,
            reserved_output_tokens=0,
            category_weights={BudgetCategory.USER_MESSAGE: 1.0},
            token_counter=_char_counter,
        )

        budgets = allocator.allocate_category_budgets()

        assert BudgetCategory.RETRIEVED_CONTEXT not in budgets
        assert budgets[BudgetCategory.USER_MESSAGE] == 1000


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReport:
    def test_everything_fits_under_budget(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=1000,
            reserved_output_tokens=0,
            category_weights={
                BudgetCategory.RETRIEVED_CONTEXT: 0.5,
                BudgetCategory.USER_MESSAGE: 0.5,
            },
            token_counter=_char_counter,
        )
        components = [
            ContextComponent(BudgetCategory.RETRIEVED_CONTEXT, "x" * 100),
            ContextComponent(BudgetCategory.USER_MESSAGE, "y" * 50),
        ]

        report = allocator.build_report(components)

        assert isinstance(report, BudgetReport)
        assert report.fits is True
        assert report.total_actual_tokens == 150
        assert report.categories_needing_truncation == []

    def test_category_over_its_own_budget_is_flagged(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=200,
            reserved_output_tokens=0,
            category_weights={
                BudgetCategory.RETRIEVED_CONTEXT: 0.5,   # budget 100
                BudgetCategory.USER_MESSAGE: 0.5,        # budget 100
            },
            token_counter=_char_counter,
        )
        components = [
            ContextComponent(BudgetCategory.RETRIEVED_CONTEXT, "x" * 150),
            ContextComponent(BudgetCategory.USER_MESSAGE, "y" * 20),
        ]

        report = allocator.build_report(components)

        assert report.fits is False
        assert report.categories_needing_truncation == [
            BudgetCategory.RETRIEVED_CONTEXT
        ]

        retrieved_allocation = next(
            a for a in report.allocations if a.category == BudgetCategory.RETRIEVED_CONTEXT
        )
        assert isinstance(retrieved_allocation, CategoryAllocation)
        assert retrieved_allocation.over_budget is True
        assert retrieved_allocation.overflow_tokens == 50

    def test_category_missing_a_weight_is_forced_over_budget(self):
        # Models the exact mistake from the lesson incident: a category
        # (here, the user's own message) was never given a budget at
        # all, so ANY non-empty content in it is immediately flagged as
        # over budget instead of silently competing for leftover space.
        allocator = TokenBudgetAllocator(
            context_window_tokens=1000,
            reserved_output_tokens=0,
            category_weights={BudgetCategory.RETRIEVED_CONTEXT: 1.0},
            token_counter=_char_counter,
        )
        components = [
            ContextComponent(BudgetCategory.RETRIEVED_CONTEXT, "x" * 200),
            ContextComponent(BudgetCategory.USER_MESSAGE, "Какой у меня тариф?"),
        ]

        report = allocator.build_report(components)

        assert BudgetCategory.USER_MESSAGE in report.categories_needing_truncation
        user_allocation = next(
            a for a in report.allocations if a.category == BudgetCategory.USER_MESSAGE
        )
        assert user_allocation.allocated_tokens == 0
        assert user_allocation.over_budget is True

    def test_multiple_components_of_same_category_are_summed(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=1000,
            reserved_output_tokens=0,
            category_weights={BudgetCategory.EPISODIC_MEMORY: 1.0},
            token_counter=_char_counter,
        )
        components = [
            ContextComponent(BudgetCategory.EPISODIC_MEMORY, "a" * 300),
            ContextComponent(BudgetCategory.EPISODIC_MEMORY, "b" * 300),
        ]

        report = allocator.build_report(components)

        episodic_allocation = next(
            a for a in report.allocations if a.category == BudgetCategory.EPISODIC_MEMORY
        )
        assert episodic_allocation.actual_tokens == 600
        assert report.total_actual_tokens == 600

    def test_categories_needing_truncation_preserve_component_order(self):
        allocator = TokenBudgetAllocator(
            context_window_tokens=100,
            reserved_output_tokens=0,
            category_weights={
                BudgetCategory.EPISODIC_MEMORY: 0.5,
                BudgetCategory.RETRIEVED_CONTEXT: 0.5,
            },
            token_counter=_char_counter,
        )
        components = [
            ContextComponent(BudgetCategory.EPISODIC_MEMORY, "e" * 80),
            ContextComponent(BudgetCategory.RETRIEVED_CONTEXT, "r" * 80),
        ]

        report = allocator.build_report(components)

        assert report.categories_needing_truncation == [
            BudgetCategory.EPISODIC_MEMORY,
            BudgetCategory.RETRIEVED_CONTEXT,
        ]


@pytest.mark.integration
class TestTokenBudgetAllocatorIntegration:
    """Placeholder for an end-to-end check against a real model tokenizer
    and a fully assembled production prompt. Skip with: pytest -m 'not integration'"""

    def test_budget_against_real_tokenizer(self):
        pytest.skip("Requires a real model tokenizer and a live prompt pipeline — run manually")
