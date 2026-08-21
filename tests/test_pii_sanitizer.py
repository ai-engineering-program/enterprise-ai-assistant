"""Tests for app/ingestion/pii_sanitizer.py (Lesson 4.4).

Everything here operates on plain Python strings and regular expressions
and does not touch any external service or model (no NER model, no
Qdrant), so the whole module is covered by unit tests only.

Run:
    pytest tests/test_pii_sanitizer.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.pii_sanitizer import PIIMatch, PIISanitizer


# Синтетическое обращение клиента - тот же тип текста, что подключился
# к пайплайну "Гранит-Инвест" в истории урока 4.4: контактные данные,
# СНИЛС и паспорт для доверенности на получение оборудования.
CUSTOMER_MESSAGE = (
    "Прошу связаться со мной по номеру +7 912 345-67-89 или на почту "
    "ivanov@postavshik.ru для уточнения деталей поставки. Для оформления "
    "доверенности на получение оборудования направляю данные представителя: "
    "СНИЛС 123-456-789 00, паспорт 45 04 123456."
)

# Три формата записи одного и того же телефонного номера, встречающиеся
# в одном корпусе (см. таблицу в тексте урока 4.4).
PHONE_FORMATS = [
    "+79123456789",
    "89123456789",
    "+7 (912) 345-67-89",
]

# ИНН физического лица (12 цифр) и юридического лица (10 цифр).
INN_INDIVIDUAL = "500100732259"
INN_ORGANIZATION = "7707083893"

# Реквизит организации, который НЕ является персональными данными по
# 152-ФЗ: ОГРН - 13-значный номер, не должен быть спутан с ИНН.
OGRN_NOT_PII = "1027700132195"


@pytest.mark.unit
class TestEmailDetection:
    def test_finds_email_in_customer_message(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        emails = [m for m in matches if m.pii_type == "email"]
        assert len(emails) == 1
        assert emails[0].value == "ivanov@postavshik.ru"


@pytest.mark.unit
class TestPhoneDetection:
    @pytest.mark.parametrize("phone", PHONE_FORMATS)
    def test_detects_all_three_phone_formats(self, phone):
        sanitizer = PIISanitizer()
        text = f"Контакт для связи: {phone}."
        matches = sanitizer.detect(text)
        phones = [m for m in matches if m.pii_type == "phone"]
        assert len(phones) == 1

    def test_finds_phone_in_customer_message(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        phones = [m for m in matches if m.pii_type == "phone"]
        assert len(phones) == 1


@pytest.mark.unit
class TestSnilsDetection:
    def test_finds_snils_in_customer_message(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        snils = [m for m in matches if m.pii_type == "snils"]
        assert len(snils) == 1
        assert snils[0].value == "123-456-789 00"


@pytest.mark.unit
class TestPassportDetection:
    def test_finds_passport_with_spaced_series(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        passports = [m for m in matches if m.pii_type == "passport"]
        assert len(passports) == 1
        assert passports[0].value == "45 04 123456"

    def test_finds_passport_with_merged_series(self):
        sanitizer = PIISanitizer()
        text = "Паспорт 4504 123456 приложен к заявке."
        matches = sanitizer.detect(text)
        passports = [m for m in matches if m.pii_type == "passport"]
        assert len(passports) == 1


@pytest.mark.unit
class TestInnDetection:
    def test_finds_individual_inn_12_digits(self):
        sanitizer = PIISanitizer()
        text = f"ИНН представителя: {INN_INDIVIDUAL}."
        matches = sanitizer.detect(text)
        inns = [m for m in matches if m.pii_type == "inn"]
        assert len(inns) == 1
        assert inns[0].value == INN_INDIVIDUAL

    def test_finds_organization_inn_10_digits(self):
        sanitizer = PIISanitizer()
        text = f"ИНН поставщика: {INN_ORGANIZATION}."
        matches = sanitizer.detect(text)
        inns = [m for m in matches if m.pii_type == "inn"]
        assert len(inns) == 1
        assert inns[0].value == INN_ORGANIZATION

    def test_does_not_split_ogrn_into_false_inn_match(self):
        """ОГРН - 13-значный номер организации, не персональные данные
        по 152-ФЗ (см. текст урока 4.4). Паттерн ИНН с границами слова
        не должен находить внутри него 10- или 12-значную подстроку."""
        sanitizer = PIISanitizer()
        text = f"ОГРН организации: {OGRN_NOT_PII}."
        matches = sanitizer.detect(text)
        inns = [m for m in matches if m.pii_type == "inn"]
        assert inns == []


@pytest.mark.unit
class TestDetectOrdering:
    def test_matches_are_sorted_by_start_position(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        starts = [m.start for m in matches]
        assert starts == sorted(starts)

    def test_returns_pii_match_instances(self):
        sanitizer = PIISanitizer()
        matches = sanitizer.detect(CUSTOMER_MESSAGE)
        assert len(matches) >= 4
        assert all(isinstance(m, PIIMatch) for m in matches)


@pytest.mark.unit
class TestSanitizeMask:
    def test_mask_mode_replaces_values_with_typed_tokens(self):
        sanitizer = PIISanitizer()
        sanitized, matches = sanitizer.sanitize(CUSTOMER_MESSAGE, mode="mask")
        assert "[PHONE_REDACTED]" in sanitized
        assert "[EMAIL_REDACTED]" in sanitized
        assert "[SNILS_REDACTED]" in sanitized
        assert "[PASSPORT_REDACTED]" in sanitized
        # Исходные значения не должны остаться в тексте.
        assert "+7 912 345-67-89" not in sanitized
        assert "ivanov@postavshik.ru" not in sanitized
        assert len(matches) >= 4

    def test_mask_mode_preserves_surrounding_text(self):
        sanitizer = PIISanitizer()
        text = "Звоните по номеру +79123456789 в любое время."
        sanitized, _ = sanitizer.sanitize(text, mode="mask")
        assert sanitized.startswith("Звоните по номеру ")
        assert sanitized.endswith(" в любое время.")

    def test_no_pii_returns_text_unchanged(self):
        sanitizer = PIISanitizer()
        text = "Обычный текст без персональных данных вообще."
        sanitized, matches = sanitizer.sanitize(text, mode="mask")
        assert sanitized == text
        assert matches == []

    def test_returned_matches_reflect_original_values_before_masking(self):
        """Аудиторский след: matches должны содержать значения ДО
        замены, а не маски - иначе невозможно восстановить, что было
        найдено (см. текст урока 4.4)."""
        sanitizer = PIISanitizer()
        text = "Email для связи: ivanov@postavshik.ru."
        _, matches = sanitizer.sanitize(text, mode="mask")
        assert len(matches) == 1
        assert matches[0].value == "ivanov@postavshik.ru"


@pytest.mark.unit
class TestSanitizeRemove:
    def test_remove_mode_deletes_values_without_replacement(self):
        sanitizer = PIISanitizer()
        text = "Контакт: ivanov@postavshik.ru, звонок позже."
        sanitized, _ = sanitizer.sanitize(text, mode="remove")
        assert "ivanov@postavshik.ru" not in sanitized
        assert "[EMAIL_REDACTED]" not in sanitized

    def test_remove_mode_handles_multiple_matches(self):
        sanitizer = PIISanitizer()
        text = "Телефон +79123456789, почта ivanov@postavshik.ru."
        sanitized, matches = sanitizer.sanitize(text, mode="remove")
        assert len(matches) == 2
        assert "+79123456789" not in sanitized
        assert "ivanov@postavshik.ru" not in sanitized


@pytest.mark.unit
class TestFullCustomerMessageScenario:
    def test_customer_message_end_to_end(self):
        """Ровно сценарий истории урока 4.4: обращение клиента с
        телефоном, email, СНИЛС и паспортом - после sanitize() ни одно
        исходное значение не должно остаться в тексте, готовом для
        chunking и индексирования."""
        sanitizer = PIISanitizer()
        sanitized, matches = sanitizer.sanitize(CUSTOMER_MESSAGE, mode="mask")

        found_types = {m.pii_type for m in matches}
        assert found_types == {"email", "phone", "snils", "passport"}

        for original_value in (
            "+7 912 345-67-89",
            "ivanov@postavshik.ru",
            "123-456-789 00",
            "45 04 123456",
        ):
            assert original_value not in sanitized


@pytest.mark.integration
class TestHybridDetectionWithNer:
    """Тесты гибридного детектора (regex + NER) требуют загрузки
    NER-модели (natasha/spaCy) и не входят в объём этого упражнения -
    PIISanitizer в этом уроке ограничен regex-слоем (см. текст урока
    4.4, "NER для имён и адресов"). Пропускается по умолчанию:
    pytest -m "not integration"
    """

    def test_ner_layer_extension_is_out_of_scope_for_this_lesson(self):
        pytest.skip(
            "NER-детекция имён и адресов не реализуется в уроке 4.4 - "
            "расширение PIISanitizer NER-слоем выходит за рамки этого "
            "упражнения и требует отдельной модели (natasha/spaCy)."
        )
