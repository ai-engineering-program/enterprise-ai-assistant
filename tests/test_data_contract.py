"""Tests for app/ingestion/data_contract.py (Lesson 7.3).

DocumentContractValidator only checks a raw input document (a plain
dict, as it arrives from the source, before parsing/normalization)
against an explicit DataContract schema — these unit tests build
contracts and raw documents directly, without running any real parsing,
OCR or embedding pipeline.

Run unit tests only:
    pytest tests/test_data_contract.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.data_contract import (
    ContractViolation,
    DataContract,
    DocumentContractValidator,
    FieldSpec,
    FieldType,
)


def _grantpress_contract_v1() -> DataContract:
    """Контракт источника "ГранитПресс" версии 1.0 из истории урока:
    published_at приходит строго в ISO-8601 с явным часовым поясом."""
    return DataContract(
        source_name="grantpress",
        version="1.0",
        fields=(
            FieldSpec(name="document_id", field_type=FieldType.STRING, required=True),
            FieldSpec(name="title", field_type=FieldType.STRING, required=True),
            FieldSpec(
                name="category",
                field_type=FieldType.ENUM,
                required=True,
                allowed_values=("news", "safety_notice", "procurement"),
            ),
            FieldSpec(
                name="published_at",
                field_type=FieldType.DATETIME,
                required=True,
                date_formats=("%Y-%m-%dT%H:%M:%S%z",),
                max_future_skew_seconds=300.0,
            ),
            FieldSpec(
                name="priority",
                field_type=FieldType.INTEGER,
                required=False,
                min_value=1,
                max_value=5,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# field_by_name
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFieldByName:
    def test_finds_existing_field(self):
        contract = _grantpress_contract_v1()
        spec = contract.field_by_name("published_at")
        assert spec is not None
        assert spec.field_type == FieldType.DATETIME

    def test_missing_field_returns_none(self):
        contract = _grantpress_contract_v1()
        assert contract.field_by_name("nonexistent") is None


# ---------------------------------------------------------------------------
# validate_document — required fields
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRequiredFields:
    def test_valid_document_has_no_violations(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-1",
            {
                "document_id": "doc-1",
                "title": "Плановая остановка крана №4",
                "category": "safety_notice",
                "published_at": "2026-03-01T09:00:00+03:00",
                "priority": 1,
            },
        )
        assert violations == []

    def test_missing_required_field(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-2",
            {
                "document_id": "doc-2",
                "category": "news",
                "published_at": "2026-03-01T09:00:00+03:00",
            },
        )
        assert len(violations) == 1
        assert violations[0].field == "title"
        assert violations[0].violation_type == "missing_required"

    def test_null_required_field_is_treated_as_missing(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-3",
            {
                "document_id": "doc-3",
                "title": None,
                "category": "news",
                "published_at": "2026-03-01T09:00:00+03:00",
            },
        )
        assert any(v.field == "title" and v.violation_type == "missing_required" for v in violations)

    def test_missing_optional_field_is_not_a_violation(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-4",
            {
                "document_id": "doc-4",
                "title": "Тендер на поставку буровых насадок",
                "category": "procurement",
                "published_at": "2026-03-01T09:00:00+03:00",
                # priority отсутствует — поле необязательное
            },
        )
        assert violations == []


# ---------------------------------------------------------------------------
# validate_document — types and ranges
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTypesAndRanges:
    def test_wrong_string_type(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-5",
            {
                "document_id": "doc-5",
                "title": 12345,
                "category": "news",
                "published_at": "2026-03-01T09:00:00+03:00",
            },
        )
        assert any(v.field == "title" and v.violation_type == "wrong_type" for v in violations)

    def test_enum_out_of_range(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-6",
            {
                "document_id": "doc-6",
                "title": "Внутренний бюллетень",
                "category": "gossip",
                "published_at": "2026-03-01T09:00:00+03:00",
            },
        )
        assert any(v.field == "category" and v.violation_type == "out_of_range" for v in violations)

    def test_integer_range_violation(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-7",
            {
                "document_id": "doc-7",
                "title": "Срочное уведомление",
                "category": "safety_notice",
                "published_at": "2026-03-01T09:00:00+03:00",
                "priority": 9,
            },
        )
        assert any(v.field == "priority" and v.violation_type == "out_of_range" for v in violations)

    def test_bool_is_not_a_valid_integer(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-8",
            {
                "document_id": "doc-8",
                "title": "Заметка",
                "category": "news",
                "published_at": "2026-03-01T09:00:00+03:00",
                "priority": True,
            },
        )
        assert any(v.field == "priority" and v.violation_type == "wrong_type" for v in violations)


# ---------------------------------------------------------------------------
# validate_document — datetime (the core "ГранитПресс" incident)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDatetimeValidation:
    def test_new_source_format_rejected_by_v1_contract(self):
        """Ровно инцидент урока: CMS-вендор ГранитПресс молча сменил
        формат published_at с ISO-8601 на "ДД.ММ.ГГГГ ЧЧ:ММ". Контракт
        v1.0 не знает этот формат — документ должен явно провалить
        проверку, а не быть тихо принят с датой по умолчанию."""
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-9",
            {
                "document_id": "doc-9",
                "title": "Плановая остановка крана №4",
                "category": "safety_notice",
                "published_at": "01.03.2026 09:00",
            },
        )
        assert len(violations) == 1
        assert violations[0].field == "published_at"
        assert violations[0].violation_type == "unparseable_datetime"

    def test_far_future_date_rejected(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-10",
            {
                "document_id": "doc-10",
                "title": "Заметка из будущего",
                "category": "news",
                "published_at": "2099-01-01T00:00:00+00:00",
            },
        )
        assert any(v.field == "published_at" and v.violation_type == "out_of_range" for v in violations)

    def test_non_string_datetime_is_wrong_type(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        violations = validator.validate_document(
            "doc-11",
            {
                "document_id": "doc-11",
                "title": "Заметка",
                "category": "news",
                "published_at": 20260301,
            },
        )
        assert any(v.field == "published_at" and v.violation_type == "wrong_type" for v in violations)

    def test_transition_window_accepts_both_formats(self):
        """Версионирование через переходное окно: контракт v1.1 временно
        принимает и старый ISO-формат, и новый локализованный формат
        источника одновременно (см. текст урока, "Версионирование:
        переходное окно")."""
        contract_v1_1 = DataContract(
            source_name="grantpress",
            version="1.1",
            fields=(
                FieldSpec(
                    name="published_at",
                    field_type=FieldType.DATETIME,
                    required=True,
                    date_formats=("%Y-%m-%dT%H:%M:%S%z", "%d.%m.%Y %H:%M"),
                ),
            ),
        )
        validator = DocumentContractValidator(contract_v1_1)

        old_format_violations = validator.validate_document(
            "doc-12", {"published_at": "2026-03-01T09:00:00+03:00"}
        )
        new_format_violations = validator.validate_document(
            "doc-13", {"published_at": "01.03.2026 09:00"}
        )
        assert old_format_violations == []
        assert new_format_violations == []


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateBatch:
    def test_only_offending_documents_appear_in_result(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        result = validator.validate_batch(
            {
                "doc-ok": {
                    "document_id": "doc-ok",
                    "title": "Норма",
                    "category": "news",
                    "published_at": "2026-03-01T09:00:00+03:00",
                },
                "doc-bad": {
                    "document_id": "doc-bad",
                    "title": "Плановая остановка крана №4",
                    "category": "safety_notice",
                    "published_at": "01.03.2026 09:00",
                },
            }
        )
        assert set(result.keys()) == {"doc-bad"}
        assert result["doc-bad"][0].violation_type == "unparseable_datetime"

    def test_empty_batch_returns_empty_dict(self):
        validator = DocumentContractValidator(_grantpress_contract_v1())
        assert validator.validate_batch({}) == {}


# ---------------------------------------------------------------------------
# Integration — requires a real source export sample
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDocumentContractValidatorIntegration:
    """Требует реального образца выгрузки источника (например, свежего
    экспорта ГранитПресс) для регрессионной проверки контракта.
    Скип по умолчанию: pytest -m 'not integration'."""

    def test_validate_against_real_source_export_sample(self):
        pytest.skip(
            "Требует доступ к реальному экспорту источника — запускайте вручную:\n"
            "  pytest tests/test_data_contract.py -m integration -v"
        )
