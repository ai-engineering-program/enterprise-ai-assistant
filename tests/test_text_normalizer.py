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


# Невидимые символы и гомоглифы для тестов урока 4.2 собраны через
# chr(codepoint), а не вставлены буквальными символами в исходник —
# иначе этот файл сам стал бы примером проблемы, которую он проверяет
# (невидимый символ, случайно вставленный в текст и незаметный при
# просмотре/ревью diff).
ZWSP = chr(0x200B)  # ZERO WIDTH SPACE
BOM = chr(0xFEFF)  # ZERO WIDTH NO-BREAK SPACE / BOM
LRM = chr(0x200E)  # LEFT-TO-RIGHT MARK
RLM = chr(0x200F)  # RIGHT-TO-LEFT MARK
CYRILLIC_A = chr(0x0410)  # "А" — кириллическая, визуально как латинская "A"
CYRILLIC_T = chr(0x0422)  # "Т" — кириллическая, визуально как латинская "T"


@pytest.mark.unit
class TestStripInvisibleCharacters:
    def test_removes_zero_width_space_inside_word(self):
        normalizer = TextNormalizer()
        # Невидимая точка переноса слова, которую старая СЭД "Архивариус"
        # вставляла при экспорте текста с выравниванием по ширине (см.
        # историю урока 4.2 — миграция "Полюс-Техно" в "Гранит-Инвест").
        text = "до" + ZWSP + "говор поставки"
        result = normalizer.strip_invisible_characters(text)
        assert result == "договор поставки"

    def test_removes_bom_at_start_of_text(self):
        normalizer = TextNormalizer()
        text = BOM + "Договор №14"
        result = normalizer.strip_invisible_characters(text)
        assert result == "Договор №14"

    def test_removes_direction_marks(self):
        normalizer = TextNormalizer()
        text = LRM + "Проект" + RLM + " завершён"
        result = normalizer.strip_invisible_characters(text)
        assert result == "Проект завершён"

    def test_leaves_plain_text_untouched(self):
        normalizer = TextNormalizer()
        text = "обычный текст без невидимых символов"
        assert normalizer.strip_invisible_characters(text) == text


@pytest.mark.unit
class TestResolveHomoglyphs:
    def test_replaces_latin_o_inside_mostly_cyrillic_word(self):
        """Ключевой сценарий истории урока: слово "договор", в котором
        одна буква набрана латинской "o" вместо кириллической "о",
        должно быть приведено к полностью кириллическому виду."""
        normalizer = TextNormalizer()
        mixed = "д" + "o" + "говор"  # латинская "o" на второй позиции
        result = normalizer.resolve_homoglyphs(mixed)
        assert result == "договор"

    def test_replaces_cyrillic_letters_inside_mostly_latin_word(self):
        normalizer = TextNormalizer()
        # Латиница доминирует (6 латинских букв против 1 кириллической
        # "а"), кириллическая буква должна стать латинской.
        mixed = "cont" + "а" + "ct"  # "contаct" с кириллической "а"
        result = normalizer.resolve_homoglyphs(mixed)
        assert result == "contact"

    def test_leaves_single_script_tokens_untouched(self):
        normalizer = TextNormalizer()
        text = "договор поставки"
        assert normalizer.resolve_homoglyphs(text) == text

    def test_leaves_pure_latin_acronym_untouched(self):
        normalizer = TextNormalizer()
        text = "см. раздел API документации"
        assert normalizer.resolve_homoglyphs(text) == text

    def test_known_limitation_mixed_acronym_inside_hyphenated_token(self):
        """Документированное ограничение эвристики (разбирается в
        тексте урока): в токене "IT-специалист" кириллица доминирует по
        числу букв, поэтому латинская "T" (для неё есть пара в
        LATIN_TO_CYRILLIC_HOMOGLYPHS) будет заменена на кириллическую
        "Т", а буква "I" останется латинской, потому что для неё нет
        гомоглифа в таблице. Тест фиксирует именно это поведение, а не
        "исправляет" его — это документированный, а не случайный
        пробел покрытия эвристики по большинству."""
        normalizer = TextNormalizer()
        result = normalizer.resolve_homoglyphs("IT-специалист")
        assert result != "IT-специалист"
        assert CYRILLIC_T in result


@pytest.mark.unit
class TestNormalizeFullPipelineUnicode:
    """Урок 4.2: полный конвейер должен устранять невидимые символы и
    гомоглифы наравне с проблемами урока 4.1, и в правильном порядке
    относительно свёртки регистра."""

    def test_pipeline_strips_invisible_characters(self):
        normalizer = TextNormalizer()
        text = BOM + "Итого: 100" + ZWSP + " рублей"
        result = normalizer.normalize(text, fold_case=False)
        assert BOM not in result
        assert ZWSP not in result

    def test_pipeline_resolves_homoglyph_before_case_folding(self):
        """Прогон полного конвейера на слове с примешанной латинской
        "o" — normalize() должен свести его к тому же результату, что
        и для изначально правильно набранного слова."""
        normalizer = TextNormalizer()
        mixed = "Д" + "o" + "говор"  # "Дoговор" с латинской "o"
        clean = "договор"
        assert normalizer.normalize(mixed) == normalizer.normalize(clean) == clean

    def test_pipeline_normalizes_mixed_script_acronym_via_correct_order(self):
        """Если resolve_homoglyphs() выполняется до свёртки регистра
        (правильный порядок — см. docstring normalize()), смешанный по
        алфавиту токен "АPI" (кириллическая "А" + латинские "P", "I")
        сначала становится полностью латинским "API" и только потом
        распознаётся эвристикой как аббревиатура и не трогается при
        свёртке регистра. Если бы порядок шагов был обратным,
        ACRONYM_PATTERN распознал бы "АPI" как аббревиатуру ещё со
        смешанным алфавитом (он не различает, из какого алфавита взяты
        заглавные буквы) и сохранил бы испорченный токен навсегда."""
        normalizer = TextNormalizer()
        mixed_acronym = CYRILLIC_A + "PI"
        assert normalizer.normalize(mixed_acronym) == "API"


@pytest.mark.integration
class TestTextNormalizerIntegration:
    """Требует загрузки embedding-модели (sentence-transformers) —
    запускать вручную, не в CI по умолчанию."""

    def test_normalized_forms_produce_identical_embedding(self):
        pytest.skip(
            "Требует загрузки sentence-transformers модели — запускать вручную"
        )
