import pytest

from app.ingestion.ocr_trigger import (
    OCRTrigger,
    PageTextStats,
    OCRNecessityDetector,
)


@pytest.mark.unit
class TestAssessEmptyInput:
    def test_no_pages_needs_no_ocr(self):
        detector = OCRNecessityDetector()
        report = detector.assess([])
        assert report.needs_ocr is False
        assert report.trigger == OCRTrigger.NONE
        assert report.pages_for_ocr == []


@pytest.mark.unit
class TestAssessHealthyDocument:
    def test_all_pages_healthy_no_ocr_needed(self):
        detector = OCRNecessityDetector()
        pages = [PageTextStats(1, 900), PageTextStats(2, 850), PageTextStats(3, 1100)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.NONE
        assert report.needs_ocr is False
        assert report.pages_for_ocr == []


@pytest.mark.unit
class TestAssessFullScan:
    def test_all_pages_empty_is_full_scan(self):
        detector = OCRNecessityDetector()
        pages = [PageTextStats(i, 0) for i in range(1, 6)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.FULL_SCAN
        assert report.needs_ocr is True
        assert report.pages_for_ocr == [1, 2, 3, 4, 5]

    def test_ratio_at_full_scan_threshold_is_full_scan(self):
        # 9 из 10 страниц пусты (90%) — ровно на пороге full_scan_ratio_threshold
        detector = OCRNecessityDetector(full_scan_ratio_threshold=0.9)
        pages = [PageTextStats(i, 0) for i in range(1, 10)] + [PageTextStats(10, 900)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.FULL_SCAN


@pytest.mark.unit
class TestAssessPartialScan:
    def test_single_low_page_in_hybrid_document_is_partial_scan(self):
        # Гибридный договор: 9 цифровых страниц + 1 скан подписи в конце —
        # ровно случай из истории урока про "Северный Мост".
        detector = OCRNecessityDetector()
        pages = [PageTextStats(i, 800) for i in range(1, 10)] + [PageTextStats(10, 0)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.PARTIAL_SCAN
        assert report.needs_ocr is True
        assert report.pages_for_ocr == [10]

    def test_partial_scan_flags_only_low_pages_not_whole_document(self):
        detector = OCRNecessityDetector()
        pages = [
            PageTextStats(1, 900),
            PageTextStats(2, 0),
            PageTextStats(3, 950),
            PageTextStats(4, 5),
            PageTextStats(5, 870),
        ]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.PARTIAL_SCAN
        assert report.pages_for_ocr == [2, 4]
        assert report.empty_page_ratio == pytest.approx(0.4)


@pytest.mark.unit
class TestAssessLowDensity:
    def test_low_density_when_text_present_but_sparse(self):
        # Каждая страница по отдельности проходит порог min_chars_per_page (40),
        # но общая плотность текста всё равно подозрительно низкая.
        detector = OCRNecessityDetector(min_chars_per_page=40, low_density_multiplier=2.0)
        pages = [PageTextStats(i, 45) for i in range(1, 6)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.LOW_DENSITY
        assert report.needs_ocr is True

    def test_low_density_flags_all_pages_not_a_subset(self):
        detector = OCRNecessityDetector(min_chars_per_page=40, low_density_multiplier=2.0)
        pages = [PageTextStats(1, 45), PageTextStats(2, 50), PageTextStats(3, 42)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.LOW_DENSITY
        assert report.pages_for_ocr == [1, 2, 3]

    def test_healthy_average_density_is_not_flagged(self):
        detector = OCRNecessityDetector(min_chars_per_page=40, low_density_multiplier=2.0)
        pages = [PageTextStats(1, 500), PageTextStats(2, 480)]
        report = detector.assess(pages)
        assert report.trigger == OCRTrigger.NONE


@pytest.mark.unit
class TestFromPdfParseResult:
    def test_bridges_pages_without_text_layer_correctly(self):
        detector = OCRNecessityDetector()
        report = detector.from_pdf_parse_result(
            total_pages=5,
            pages_without_text_layer=[3, 4, 5],
            char_counts_by_page={1: 900, 2: 850},
        )
        assert report.trigger == OCRTrigger.PARTIAL_SCAN
        assert report.pages_for_ocr == [3, 4, 5]
        assert report.empty_page_ratio == pytest.approx(0.6)

    def test_missing_char_counts_default_to_zero(self):
        detector = OCRNecessityDetector()
        report = detector.from_pdf_parse_result(
            total_pages=3,
            pages_without_text_layer=[],
            char_counts_by_page={1: 900},
            # страницы 2 и 3 не переданы в char_counts_by_page вовсе
        )
        assert report.pages_for_ocr == [2, 3]


@pytest.mark.integration
class TestOCRNecessityIntegration:
    """Требует реального корпуса отсканированных PDF для сквозной проверки."""

    def test_real_scanned_corpus_sample(self):
        pytest.skip("Требует набора реальных PDF из архива — запускать вручную")
