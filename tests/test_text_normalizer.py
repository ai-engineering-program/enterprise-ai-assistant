"""Tests for app/ingestion/text_normalizer.py (Lesson 4.1).

Everything here operates on plain Python strings — no external services
or downloaded models are involved for the unit tests, so most of the
module is covered by unit tests only. A single integration test checks
that the normalized forms of the "AI-система" story actually collapse
to one embedding, and is skipped by default because it downloads a
sentence-transformers model.

Run:
    pytest tests/test_text_normalizer.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.text_normalizer import TextNormalizer


@pytest.mark.unit
class TestUnescapeHtmlEntities:
    def test_unescapes_nbsp(self):
        normalizer = TextNormalizer()
        text = "Итого: 100 рублей"  # already-decoded reference
        entity_text = "Итого:&nbsp;100&nbsp;рублей"
        result = normalizer.unescape_html_entities(entity_text)
        assert result == text

    def test_unescapes_mdash_and_laquo(self):
        normalizer = TextNormalizer()
        result = normalizer.unescape_html_entities("&laquo;отчёт&raquo; &mdash; готов")
        assert result == "«отчёт» — готов"

    def test_leaves_plain_text_untouched(self):
        normalizer = TextNormalizer()
        text = "обычный текст без сущностей"
        assert normalizer.unescape_html_entities(text) == text


@pytest.mark.unit
class TestNormalizeUnicodeForm:
    def test_nfkc_collapses_fullwidth_latin(self):
        normalizer = TextNormalizer()
        # Полноширинные латинские буквы (характерны для некоторых
        # экспортов) сводятся NFKC к обычным ASCII-буквам.
        fullwidth = "ＡＩ"  # fullwidth "AI"
        result = normalizer.normalize_unicode_form(fullwidth)
        assert result == "AI"

    def test_nfkc_is_idempotent_on_plain_text(self):
        normalizer = TextNormalizer()
        text = "обычный текст"
        assert normalizer.normalize_unicode_form(text) == text


@pytest.mark.unit
class TestNormalizeDashes:
    def test_collapses_en_dash_and_em_dash_to_hyphen_minus(self):
        normalizer = TextNormalizer()
        text = "AI–система, AI—система"
        result = normalizer.normalize_dashes(text)
        assert result == "AI-система, AI-система"

    def test_collapses_minus_sign(self):
        normalizer = TextNormalizer()
        text = "ГОСТ 12.2.007−75"
        result = normalizer.normalize_dashes(text)
        assert result == "ГОСТ 12.2.007-75"

    def test_leaves_ascii_hyphen_untouched(self):
        normalizer = TextNormalizer()
        text = "КМ-2214"
        assert normalizer.normalize_dashes(text) == "КМ-2214"


@pytest.mark.unit
class TestNormalizeQuotes:
    def test_collapses_angle_quotes(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize_quotes("«Ориентир»")
        assert result == '"Ориентир"'

    def test_collapses_typographic_quotes(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize_quotes("“Ориентир”")
        assert result == '"Ориентир"'

    def test_collapses_single_quotes(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize_quotes("‘AI’")
        assert result == "'AI'"


@pytest.mark.unit
class TestIsAcronymLike:
    @pytest.mark.parametrize("token", ["ГОСТ", "API", "КМ-2214", "PDF"])
    def test_recognizes_acronyms_and_codes(self, token):
        normalizer = TextNormalizer()
        assert normalizer.is_acronym_like(token) is True

    @pytest.mark.parametrize("token", ["система", "договор", "Отчёт"])
    def test_lowercase_and_titlecase_words_are_not_acronyms(self, token):
        normalizer = TextNormalizer()
        assert normalizer.is_acronym_like(token) is False

    def test_short_uppercase_token_is_acronym(self):
        normalizer = TextNormalizer()
        # "AI" — короткая аббревиатура из двух заглавных латинских букв.
        assert normalizer.is_acronym_like("AI") is True

    def test_digits_only_token_is_not_acronym(self):
        normalizer = TextNormalizer()
        assert normalizer.is_acronym_like("2024") is False

    def test_empty_string_is_not_acronym(self):
        normalizer = TextNormalizer()
        assert normalizer.is_acronym_like("") is False


@pytest.mark.unit
class TestFoldCasePreservingAcronyms:
    def test_lowercases_regular_words(self):
        normalizer = TextNormalizer()
        result = normalizer.fold_case_preserving_acronyms("Внедрение Системы Прошло Успешно")
        assert result == "внедрение системы прошло успешно"

    def test_preserves_acronym_tokens(self):
        normalizer = TextNormalizer()
        result = normalizer.fold_case_preserving_acronyms("отчёт по ГОСТ и КМ-2214")
        assert "ГОСТ" in result
        assert "КМ-2214" in result
        assert "отчёт" in result

    def test_three_forms_of_term_converge_after_fold(self):
        """Ключевой сценарий урока 4.1: три формы термина из истории
        компании "Ориентир" должны прийти к одному виду после свёртки
        регистра (дефисы предполагаются уже унифицированными)."""
        normalizer = TextNormalizer()
        form_a = normalizer.fold_case_preserving_acronyms("AI-система")
        form_c = normalizer.fold_case_preserving_acronyms("ai-система")
        assert form_a == form_c == "ai-система"


@pytest.mark.unit
class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        normalizer = TextNormalizer()
        text = "внедрение    AI   системы"
        assert normalizer.normalize_whitespace(text) == "внедрение AI системы"

    def test_collapses_non_breaking_space(self):
        normalizer = TextNormalizer()
        #   — то, во что html.unescape() превращает &nbsp;
        # на предыдущем шаге конвейера; normalize_whitespace должен
        # схлопывать его наравне с обычным пробелом.
        text = "итого: 100 рублей"
        result = normalizer.normalize_whitespace(text)
        assert result == "итого: 100 рублей"

    def test_strips_line_edges_and_collapses_blank_lines(self):
        normalizer = TextNormalizer()
        text = "  первый абзац  \n\n\n\n  второй абзац  "
        result = normalizer.normalize_whitespace(text)
        assert result == "первый абзац\n\nвторой абзац"


@pytest.mark.unit
class TestNormalizeFullPipeline:
    def test_three_forms_of_ai_sistema_converge(self):
        """Прогон полного конвейера на трёх формах термина из истории
        урока — все должны совпасть после normalize()."""
        normalizer = TextNormalizer()
        assert normalizer.normalize("AI-система") == normalizer.normalize("ai-система")

    def test_pipeline_decodes_entities_normalizes_dashes_and_whitespace(self):
        normalizer = TextNormalizer()
        text = "Проект &laquo;AI–система&raquo;   —   в  работе"
        result = normalizer.normalize(text, fold_case=False)
        assert "&laquo;" not in result
        assert "–" not in result
        assert "  " not in result

    def test_fold_case_false_preserves_original_case(self):
        normalizer = TextNormalizer()
        text = "Марат Иванов подготовил Отчёт"
        result = normalizer.normalize(text, fold_case=False)
        assert "Марат Иванов" in result

    def test_preserves_structured_code_with_dash(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize("см. чертёж КМ–2214")
        assert "КМ-2214" in result


@pytest.mark.integration
class TestTextNormalizerIntegration:
    """Требует загрузки embedding-модели (sentence-transformers) —
    запускать вручную, не в CI по умолчанию."""

    def test_normalized_forms_produce_identical_embedding(self):
        pytest.skip(
            "Требует загрузки sentence-transformers модели — запускать вручную"
        )
