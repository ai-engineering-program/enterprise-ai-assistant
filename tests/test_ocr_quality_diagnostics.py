import pytest

from app.ingestion.ocr_quality_diagnostics import OCRQualityDiagnostics


DICTIONARY = {
    "штраф", "два", "процента", "остатка", "долга", "согласно",
    "пункту", "договора", "накладная", "груз", "доставлен",
    "срок", "доставки", "составляет", "дней", "рабочих",
}


@pytest.mark.unit
class TestDictionaryWordRatio:
    def test_all_words_in_dictionary(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        ratio = diag.dictionary_word_ratio("груз доставлен согласно пункту договора")
        assert ratio == 1.0

    def test_partial_dictionary_match(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        ratio = diag.dictionary_word_ratio("груз доставлен зкпмтр вфщхыъ")
        assert 0.0 < ratio < 1.0

    def test_only_numeric_or_code_tokens_returns_one(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        ratio = diag.dictionary_word_ratio("СМ-0184527 12.05.2024")
        assert ratio == 1.0

    def test_case_insensitive_match(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        ratio = diag.dictionary_word_ratio("ГРУЗ ДОСТАВЛЕН")
        assert ratio == 1.0

    def test_no_dictionary_matches(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        ratio = diag.dictionary_word_ratio("зкпмтр вфщхыъ юлдчсн")
        assert ratio == 0.0


@pytest.mark.unit
class TestHomoglyphTokens:
    def test_pure_cyrillic_token_not_flagged(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        assert diag.find_homoglyph_tokens("накладная СМ-0184527 доставлена") == []

    def test_mixed_script_token_flagged(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        # "C" — латинская, "М" — кириллическая: смешение внутри одного токена
        result = diag.find_homoglyph_tokens("накладная CМ-0184527 доставлена")
        assert "CМ-0184527" in result

    def test_pure_latin_token_not_flagged(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        assert diag.find_homoglyph_tokens("tracking CM-0184527 delivered") == []

    def test_no_duplicates_in_result(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        result = diag.find_homoglyph_tokens("CМ-01 повтор CМ-01 снова")
        assert result.count("CМ-01") == 1


@pytest.mark.unit
class TestDiagnoseBlock:
    def test_clean_block_not_flagged(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY, min_dictionary_ratio=0.7)
        flag = diag.diagnose_block(
            "b1", "груз доставлен в срок согласно накладной СМ-0184527"
        )
        assert flag.flagged is False
        assert flag.reasons == []

    def test_low_dictionary_ratio_flagged(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY, min_dictionary_ratio=0.7)
        flag = diag.diagnose_block("b2", "зкпмтр вфщхыъ юлдчсн ыврап")
        assert flag.flagged is True
        assert "low_dictionary_ratio" in flag.reasons

    def test_homoglyph_flagged_even_with_high_dictionary_ratio(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY, min_dictionary_ratio=0.5)
        text = "груз доставлен накладная CМ-0184527"
        flag = diag.diagnose_block("b3", text)
        assert flag.flagged is True
        assert "homoglyph_tokens" in flag.reasons
        assert "low_dictionary_ratio" not in flag.reasons


@pytest.mark.unit
class TestDiagnoseBatch:
    def test_batch_returns_only_flagged_blocks(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY, min_dictionary_ratio=0.7)
        blocks = [
            ("b1", "груз доставлен в срок"),
            ("b2", "зкпмтр вфщхыъ юлдчсн"),
            ("b3", "накладная CМ-0184527 доставлена"),
        ]
        report = diag.diagnose_batch(blocks)
        assert report.total_blocks == 3
        flagged_ids = {f.block_id for f in report.flagged_blocks}
        assert flagged_ids == {"b2", "b3"}

    def test_empty_batch(self):
        diag = OCRQualityDiagnostics(dictionary=DICTIONARY)
        report = diag.diagnose_batch([])
        assert report.total_blocks == 0
        assert report.flagged_blocks == []


@pytest.mark.integration
class TestOCRQualityDiagnosticsIntegration:
    """Требует реального OCR-вывода Tesseract на смешанных rus+eng сканах."""

    def test_with_real_tesseract_output(self):
        pytest.skip("Требует запущенного Tesseract на тестовых сканах — запускать вручную")
