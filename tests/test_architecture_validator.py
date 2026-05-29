"""
Tests for ArchitectureValidator — lesson 6.5 final project.
"""

import pytest

from app.core.architecture_validator import ArchitectureValidator
from app.core.arch_spec import ARCHITECTURE_SPEC


BAD_SPEC = {
    "llm": {"provider": "openai", "model": "gpt-4o"},
}

GOOD_SPEC = {
    "api_layer": {"async_support": True},
    "caching": {"enabled": True},
    "circuit_breaker": {"enabled": True},
    "queue": {"enabled": True},
    "event_bus": {"enabled": True},
    "observability": {"enabled": True},
    "auth": {"enabled": True},
    "rate_limiting": {"enabled": True},
    "router": {"enabled": True},
    "graceful_degradation": {"enabled": True},
}


class TestValidate:
    def test_bad_spec_has_many_issues(self):
        v = ArchitectureValidator()
        issues = v.validate(BAD_SPEC)
        assert len(issues) >= 7

    def test_good_spec_has_no_issues(self):
        v = ArchitectureValidator()
        assert v.validate(GOOD_SPEC) == []

    def test_event_store_satisfies_event_requirement(self):
        v = ArchitectureValidator()
        spec = {**GOOD_SPEC, "event_store": {"enabled": True}}
        spec.pop("event_bus", None)
        issues = v.validate(spec)
        assert not any("event" in i.lower() for i in issues)

    def test_validate_returns_all_issues(self):
        v = ArchitectureValidator()
        issues = v.validate(BAD_SPEC)
        assert len(issues) > 1


class TestGetScore:
    def test_bad_spec_score_zero(self):
        v = ArchitectureValidator()
        assert v.get_score(BAD_SPEC) == 0

    def test_good_spec_score_hundred(self):
        v = ArchitectureValidator()
        assert v.get_score(GOOD_SPEC) == 100

    def test_score_range(self):
        v = ArchitectureValidator()
        score = v.get_score(BAD_SPEC)
        assert 0 <= score <= 100


class TestSuggestImprovements:
    def test_returns_suggestions_for_bad_spec(self):
        v = ArchitectureValidator()
        suggestions = v.suggest_improvements(BAD_SPEC)
        assert len(suggestions) >= 7
        for s in suggestions:
            assert len(s) > 10

    def test_no_suggestions_for_good_spec(self):
        v = ArchitectureValidator()
        assert v.suggest_improvements(GOOD_SPEC) == []


class TestEnterpriseSpec:
    def test_enterprise_spec_scores_100(self):
        v = ArchitectureValidator()
        score = v.get_score(ARCHITECTURE_SPEC)
        assert score == 100, (
            f"enterprise-ai-assistant architecture should score 100/100, got {score}. "
            f"Issues: {v.validate(ARCHITECTURE_SPEC)}"
        )

    def test_enterprise_spec_has_no_issues(self):
        v = ArchitectureValidator()
        issues = v.validate(ARCHITECTURE_SPEC)
        assert issues == [], (
            f"enterprise-ai-assistant should have no architecture issues: {issues}"
        )
