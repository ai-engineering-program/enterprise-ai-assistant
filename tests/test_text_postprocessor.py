from __future__ import annotations

import pytest

from app.ingestion.text_postprocessor import TextPostProcessor


@pytest.mark.unit
class TestMergeHyphenatedLinebreaks:
    def test_merges_lowercase_hyphenated_word(self):
        processor = TextPostProcessor()
        text = "по условиям догово-\nра неустойка составляет"
        result = processor.merge_hyphenated_linebreaks(text)
        assert "договора" in result
        assert "-\n" not in result

    def test_does_not_merge_numeric_range(self):
        processor = TextPostProcessor()
        text = "период действия договора: 2019-\n2021 годы"
        result = processor.merge_hyphenated_linebreaks(text)
        # цифра после дефиса — это диапазон лет, а не перенос слова
        assert "2019-\n2021" in result

    def test_leaves_text_without_hyphen_breaks_untouched(self):
        processor = TextPostProcessor()
        text = "простой текст без переносов"
        assert processor.merge_hyphenated_linebreaks(text) == text


@pytest.mark.unit
class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        processor = TextPostProcessor()
        text = "штраф    составляет   2  процента"
        result = processor.normalize_whitespace(text)
        assert result == "штраф составляет 2 процента"

    def test_strips_line_edges(self):
        processor = TextPostProcessor()
        text = "  строка с пробелами по краям  \nвторая строка  "
        result = processor.normalize_whitespace(text)
        lines = result.split("\n")
        assert all(line == line.strip() for line in lines)

    def test_collapses_excessive_blank_lines(self):
        processor = TextPostProcessor()
        text = "первый абзац\n\n\n\n\nвторой абзац"
        result = processor.normalize_whitespace(text)
        assert "\n\n\n" not in result
        assert result == "первый абзац\n\nвторой абзац"


@pytest.mark.unit
class TestProcess:
    def test_full_pipeline_merges_and_normalizes(self):
        processor = TextPostProcessor()
        text = "согласно догово-\nру   от  2020  года\n\n\n\nстраница 2"
        result = processor.process(text)
        assert "договору" in result
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_order_matters_hyphen_merge_before_whitespace_collapse(self):
        processor = TextPostProcessor()
        # Одиночный перенос строки не трогается normalize_whitespace
        # (сворачиваются только 3+ подряд), поэтому склейку переноса
        # нужно делать первым шагом — иначе "-\n" останется в тексте.
        text = "неустой-\nка составляет два процента"
        result = processor.process(text)
        assert "неустойка" in result


@pytest.mark.integration
class TestTextPostProcessorIntegration:
    """Требует реального вывода Tesseract на корпусе тестовых сканов."""

    def test_real_tesseract_output_sample(self):
        pytest.skip("Требует запущенного Tesseract и тестовых сканов — запускать вручную")
