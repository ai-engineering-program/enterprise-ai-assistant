import pytest

from app.ingestion.layout_routing import LayoutComplexityAdvisor, PageProfile


@pytest.mark.unit
class TestEstimateExtraCost:
    def test_extra_cost_proportional_to_page_count(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.02, cost_per_page_layout_aware=0.05
        )
        profile = PageProfile(
            has_multiple_columns=True,
            has_table_regions=False,
            page_count=50,
        )
        assert advisor.estimate_extra_cost(profile) == pytest.approx(1.5)

    def test_zero_extra_cost_when_prices_equal(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.03, cost_per_page_layout_aware=0.03
        )
        profile = PageProfile(
            has_multiple_columns=True, has_table_regions=True, page_count=200
        )
        assert advisor.estimate_extra_cost(profile) == pytest.approx(0.0)


@pytest.mark.unit
class TestDecideSimpleLayout:
    def test_no_columns_no_tables_uses_simple_ocr(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.01, cost_per_page_layout_aware=0.05
        )
        profile = PageProfile(
            has_multiple_columns=False,
            has_table_regions=False,
            page_count=100,
            is_safety_critical=True,  # даже критичность не меняет решение
        )
        decision = advisor.decide(profile, max_acceptable_extra_cost=1_000_000.0)
        assert decision.use_layout_aware is False
        assert decision.reason == "simple_layout_no_benefit"
        assert decision.estimated_extra_cost == 0.0


@pytest.mark.unit
class TestDecideSafetyCritical:
    def test_safety_critical_forces_layout_aware_despite_cost(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.01, cost_per_page_layout_aware=0.05
        )
        profile = PageProfile(
            has_multiple_columns=True,
            has_table_regions=True,
            page_count=1000,
            is_safety_critical=True,
        )
        # Бюджет намеренно намного меньше реальной дополнительной стоимости
        decision = advisor.decide(profile, max_acceptable_extra_cost=1.0)
        assert decision.use_layout_aware is True
        assert decision.reason == "safety_critical_overrides_cost"
        assert decision.estimated_extra_cost == pytest.approx(40.0)

    def test_table_only_without_columns_is_still_complex_structure(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.01, cost_per_page_layout_aware=0.02
        )
        profile = PageProfile(
            has_multiple_columns=False,
            has_table_regions=True,
            page_count=10,
            is_safety_critical=True,
        )
        decision = advisor.decide(profile, max_acceptable_extra_cost=0.0)
        assert decision.use_layout_aware is True
        assert decision.reason == "safety_critical_overrides_cost"


@pytest.mark.unit
class TestDecideCostComparison:
    def test_cost_within_budget_is_justified(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.01, cost_per_page_layout_aware=0.05
        )
        profile = PageProfile(
            has_multiple_columns=True,
            has_table_regions=False,
            page_count=100,
            is_safety_critical=False,
        )
        decision = advisor.decide(profile, max_acceptable_extra_cost=10.0)
        assert decision.use_layout_aware is True
        assert decision.reason == "cost_justified"
        assert decision.estimated_extra_cost == pytest.approx(4.0)

    def test_cost_exceeding_budget_falls_back_to_simple(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.01, cost_per_page_layout_aware=0.05
        )
        profile = PageProfile(
            has_multiple_columns=True,
            has_table_regions=False,
            page_count=1000,
            is_safety_critical=False,
        )
        decision = advisor.decide(profile, max_acceptable_extra_cost=10.0)
        assert decision.use_layout_aware is False
        assert decision.reason == "cost_exceeds_budget"
        assert decision.estimated_extra_cost == pytest.approx(40.0)

    def test_cost_exactly_at_budget_boundary_is_justified(self):
        advisor = LayoutComplexityAdvisor(
            cost_per_page_simple=0.0, cost_per_page_layout_aware=0.01
        )
        profile = PageProfile(
            has_multiple_columns=True,
            has_table_regions=False,
            page_count=100,
            is_safety_critical=False,
        )
        # extra_cost == 1.0 == max_acceptable_extra_cost -> граница включительно
        decision = advisor.decide(profile, max_acceptable_extra_cost=1.0)
        assert decision.use_layout_aware is True
        assert decision.reason == "cost_justified"


@pytest.mark.integration
class TestLayoutComplexityAdvisorIntegration:
    """Требует реальной статистики стоимости OCR-провайдера по production-потоку."""

    def test_with_real_cost_metrics(self):
        pytest.skip(
            "Требует реальных данных о стоимости OCR по потоку документов"
            " — запускать вручную с метриками из observability"
        )
