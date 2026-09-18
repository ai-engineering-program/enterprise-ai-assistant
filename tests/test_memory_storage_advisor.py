"""Tests for app/context/memory_storage_advisor.py (Lesson 5.5, synthesis of M5).

These tests check the decision logic only — MemoryStorageAdvisor never talks
to FactJournal (5.4) or LongTermMemoryStore (5.2) directly, it only decides
WHERE a given field of knowledge belongs (see the lesson text, "hybrid
architecture" and "diagnosis" sections).

Run unit tests only:
    pytest tests/test_memory_storage_advisor.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.context.memory_storage_advisor import (
    DEFAULT_SLOW_CHANGE_THRESHOLD_DAYS,
    DEFAULT_VOLATILE_THRESHOLD_DAYS,
    FieldClassification,
    FieldProfile,
    MemoryStorageAdvisor,
    MigrationDiagnosis,
    MigrationSignals,
    StorageRecommendation,
)


# ---------------------------------------------------------------------------
# classify_field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyField:
    def test_static_reference_goes_to_document_store(self):
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(field_name="gpon_connection_regulation")

        result = advisor.classify_field(profile)

        assert isinstance(result, FieldClassification)
        assert result.field_name == "gpon_connection_regulation"
        assert result.recommendation == StorageRecommendation.DOCUMENT_STORE
        assert result.rationale

    def test_real_time_operational_wins_regardless_of_other_fields(self):
        # Even if entity_scoped and highly volatile, is_real_time_operational
        # must take priority — a live telemetry field is out of scope no
        # matter how it would otherwise be classified.
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="contact_center_queue_length",
            update_frequency_days=0.01,
            entity_scoped=True,
            is_real_time_operational=True,
        )

        result = advisor.classify_field(profile)

        assert result.recommendation == StorageRecommendation.OUT_OF_SCOPE_OPERATIONAL

    def test_entity_scoped_volatile_field_goes_to_fact_journal(self):
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="personal_manager",
            update_frequency_days=30.0,
            entity_scoped=True,
        )

        result = advisor.classify_field(profile)

        assert result.recommendation == StorageRecommendation.FACT_JOURNAL

    def test_entity_scoped_but_slow_changing_does_not_go_to_fact_journal(self):
        # entity_scoped alone is not sufficient — it must ALSO change
        # faster than the volatile threshold.
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="employee_office_location",
            update_frequency_days=200.0,
            entity_scoped=True,
        )

        result = advisor.classify_field(profile)

        assert (
            result.recommendation
            == StorageRecommendation.DOCUMENT_STORE_WITH_FRESHNESS_MONITOR
        )

    def test_slow_changing_shared_policy_gets_freshness_monitor(self):
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="tariff_plan_sla_terms",
            update_frequency_days=180.0,
            entity_scoped=False,
        )

        result = advisor.classify_field(profile)

        assert (
            result.recommendation
            == StorageRecommendation.DOCUMENT_STORE_WITH_FRESHNESS_MONITOR
        )

    def test_field_changing_slower_than_slow_threshold_is_plain_document_store(self):
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="terms_of_service_boilerplate",
            update_frequency_days=DEFAULT_SLOW_CHANGE_THRESHOLD_DAYS + 1,
            entity_scoped=False,
        )

        result = advisor.classify_field(profile)

        assert result.recommendation == StorageRecommendation.DOCUMENT_STORE

    def test_volatile_threshold_boundary_is_inclusive(self):
        advisor = MemoryStorageAdvisor()
        profile = FieldProfile(
            field_name="ticket_status",
            update_frequency_days=DEFAULT_VOLATILE_THRESHOLD_DAYS,
            entity_scoped=True,
        )

        result = advisor.classify_field(profile)

        assert result.recommendation == StorageRecommendation.FACT_JOURNAL


# ---------------------------------------------------------------------------
# classify_corpus / migration_candidates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorpusAudit:
    def _sample_profiles(self) -> list[FieldProfile]:
        return [
            FieldProfile(field_name="gpon_connection_regulation"),
            FieldProfile(
                field_name="personal_manager",
                update_frequency_days=30.0,
                entity_scoped=True,
            ),
            FieldProfile(
                field_name="employee_department",
                update_frequency_days=45.0,
                entity_scoped=True,
            ),
            FieldProfile(
                field_name="tariff_plan_sla_terms",
                update_frequency_days=180.0,
                entity_scoped=False,
            ),
            FieldProfile(
                field_name="contact_center_queue_length",
                is_real_time_operational=True,
            ),
        ]

    def test_classify_corpus_returns_one_entry_per_field(self):
        advisor = MemoryStorageAdvisor()
        results = advisor.classify_corpus(self._sample_profiles())

        assert set(results.keys()) == {
            "gpon_connection_regulation",
            "personal_manager",
            "employee_department",
            "tariff_plan_sla_terms",
            "contact_center_queue_length",
        }
        assert results["personal_manager"].recommendation == StorageRecommendation.FACT_JOURNAL

    def test_migration_candidates_filters_only_fact_journal(self):
        advisor = MemoryStorageAdvisor()
        candidates = advisor.migration_candidates(self._sample_profiles())

        candidate_names = {c.field_name for c in candidates}
        assert candidate_names == {"personal_manager", "employee_department"}
        assert all(
            c.recommendation == StorageRecommendation.FACT_JOURNAL
            for c in candidates
        )


# ---------------------------------------------------------------------------
# diagnose_wrong_storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiagnoseWrongStorage:
    def test_smoking_gun_signal_alone_triggers_migration(self):
        advisor = MemoryStorageAdvisor()
        signals = MigrationSignals(correct_version_exists_but_wrong_surfaces=True)

        diagnosis = advisor.diagnose_wrong_storage(signals)

        assert isinstance(diagnosis, MigrationDiagnosis)
        assert diagnosis.should_migrate is True
        assert diagnosis.matched_signals == [
            "correct_version_exists_but_wrong_surfaces"
        ]

    def test_two_weaker_signals_together_trigger_migration(self):
        advisor = MemoryStorageAdvisor()
        signals = MigrationSignals(
            accuracy_drop_on_entity_queries=True,
            non_deterministic_same_entity=True,
        )

        diagnosis = advisor.diagnose_wrong_storage(signals)

        assert diagnosis.should_migrate is True
        assert diagnosis.matched_signals == [
            "accuracy_drop_on_entity_queries",
            "non_deterministic_same_entity",
        ]

    def test_single_weak_signal_is_not_sufficient(self):
        advisor = MemoryStorageAdvisor()
        signals = MigrationSignals(accuracy_drop_on_entity_queries=True)

        diagnosis = advisor.diagnose_wrong_storage(signals)

        assert diagnosis.should_migrate is False
        assert diagnosis.matched_signals == ["accuracy_drop_on_entity_queries"]

    def test_no_signals_means_no_migration(self):
        advisor = MemoryStorageAdvisor()
        signals = MigrationSignals()

        diagnosis = advisor.diagnose_wrong_storage(signals)

        assert diagnosis.should_migrate is False
        assert diagnosis.matched_signals == []


@pytest.mark.integration
class TestMemoryStorageAdvisorIntegration:
    """Placeholder for an end-to-end audit against a real, populated
    document store + FactJournal deployment. Skip with:
    pytest -m 'not integration'"""

    def test_audit_against_real_corpus(self):
        pytest.skip("Requires a populated Qdrant corpus and FactJournal — run manually")
