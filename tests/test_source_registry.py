import pytest

from app.context.source_registry import SourceRegistry, SourceType


@pytest.mark.unit
class TestRegister:
    def test_register_returns_profile(self):
        registry = SourceRegistry()
        profile = registry.register(
            "confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low"
        )
        assert profile.name == "confluence_pdn_plan"
        assert profile.source_type == SourceType.CONFLUENCE
        assert profile.authority_weight == 0.7
        assert profile.volatility == "low"

    def test_register_stores_profile_retrievable_via_get(self):
        registry = SourceRegistry()
        registry.register("jira_pdn_142", SourceType.JIRA, 0.6, "medium")
        profile = registry.get("jira_pdn_142")
        assert profile is not None
        assert profile.source_type == SourceType.JIRA

    @pytest.mark.parametrize("bad_weight", [-0.1, 1.1, 2.0, -5.0])
    def test_register_rejects_out_of_range_weight(self, bad_weight):
        registry = SourceRegistry()
        with pytest.raises(ValueError):
            registry.register("some_source", SourceType.SLACK, bad_weight, "high")

    def test_register_accepts_boundary_weights(self):
        registry = SourceRegistry()
        registry.register("min_weight", SourceType.OTHER, 0.0, "low")
        registry.register("max_weight", SourceType.OTHER, 1.0, "low")
        assert registry.get("min_weight").authority_weight == 0.0
        assert registry.get("max_weight").authority_weight == 1.0


@pytest.mark.unit
class TestGet:
    def test_get_unknown_source_returns_none(self):
        registry = SourceRegistry()
        assert registry.get("does_not_exist") is None


@pytest.mark.unit
class TestRankByAuthority:
    def test_orders_descending_by_weight(self):
        registry = SourceRegistry()
        registry.register("confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low")
        registry.register("jira_pdn_142", SourceType.JIRA, 0.6, "medium")
        registry.register("slack_platform_migrations", SourceType.SLACK, 0.3, "high")

        assert registry.rank_by_authority() == [
            "confluence_pdn_plan",
            "jira_pdn_142",
            "slack_platform_migrations",
        ]

    def test_ties_broken_alphabetically(self):
        registry = SourceRegistry()
        registry.register("zebra_source", SourceType.OTHER, 0.5, "medium")
        registry.register("alpha_source", SourceType.OTHER, 0.5, "medium")

        assert registry.rank_by_authority() == ["alpha_source", "zebra_source"]

    def test_empty_registry_returns_empty_list(self):
        registry = SourceRegistry()
        assert registry.rank_by_authority() == []


@pytest.mark.unit
class TestResolveConflict:
    def test_returns_higher_authority_source(self):
        registry = SourceRegistry()
        registry.register("confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low")
        registry.register("jira_pdn_142", SourceType.JIRA, 0.6, "medium")

        assert (
            registry.resolve_conflict("confluence_pdn_plan", "jira_pdn_142")
            == "confluence_pdn_plan"
        )

    def test_order_of_arguments_does_not_matter(self):
        registry = SourceRegistry()
        registry.register("confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low")
        registry.register("jira_pdn_142", SourceType.JIRA, 0.6, "medium")

        assert (
            registry.resolve_conflict("jira_pdn_142", "confluence_pdn_plan")
            == "confluence_pdn_plan"
        )

    def test_raises_key_error_for_unknown_source(self):
        registry = SourceRegistry()
        registry.register("confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low")

        with pytest.raises(KeyError):
            registry.resolve_conflict("confluence_pdn_plan", "ghost_source")

    def test_raises_value_error_on_equal_weights(self):
        registry = SourceRegistry()
        registry.register("confluence_pdn_plan", SourceType.CONFLUENCE, 0.7, "low")
        registry.register("jira_pdn_999", SourceType.JIRA, 0.7, "medium")

        with pytest.raises(ValueError):
            registry.resolve_conflict("confluence_pdn_plan", "jira_pdn_999")


@pytest.mark.integration
class TestSourceRegistryIntegration:
    """Требует реальных API Confluence/Jira/Slack для автоматического
    обнаружения источников и их метаданных. В этом уроке не используется —
    заготовка для более поздних курсов, где реестр наполняется из живых
    коннекторов, а не регистрируется вручную."""

    def test_auto_discovery_from_live_sources(self):
        pytest.skip("Требует настроенных коннекторов Confluence/Jira/Slack")
