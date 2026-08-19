import pytest

from app.ingestion.reference_audit import ReferenceSampleAuditor


@pytest.mark.unit
class TestLevenshteinDistance:
    def test_identical_strings(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.levenshtein_distance("накладная", "накладная") == 0

    def test_single_substitution(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.levenshtein_distance("kitten", "sitten") == 1

    def test_classic_example(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.levenshtein_distance("kitten", "sitting") == 3

    def test_empty_strings(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.levenshtein_distance("", "") == 0
        assert auditor.levenshtein_distance("груз", "") == 4
        assert auditor.levenshtein_distance("", "груз") == 4


@pytest.mark.unit
class TestCharacterErrorRate:
    def test_no_errors(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.character_error_rate("накладная", "накладная") == 0.0

    def test_some_errors(self):
        auditor = ReferenceSampleAuditor()
        cer = auditor.character_error_rate("sitten", "kitten")
        assert cer == pytest.approx(1 / 6)

    def test_both_empty(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.character_error_rate("", "") == 0.0

    def test_reference_empty_ocr_not(self):
        auditor = ReferenceSampleAuditor()
        assert auditor.character_error_rate("текст", "") == 1.0


@pytest.mark.unit
class TestAuditSample:
    def test_mean_and_worst_indices(self):
        auditor = ReferenceSampleAuditor()
        pairs = [
            ("накладная доставлена", "накладная доставлена"),
            ("накладнля дlставлена", "накладная доставлена"),
            ("груз доставлен", "груз доставлен"),
        ]
        report = auditor.audit_sample(pairs)
        assert len(report.per_block_cer) == 3
        assert report.per_block_cer[0] == 0.0
        assert report.per_block_cer[2] == 0.0
        assert report.per_block_cer[1] > 0.0
        assert report.worst_block_indices[0] == 1
        assert report.mean_cer == pytest.approx(sum(report.per_block_cer) / 3)

    def test_empty_sample(self):
        auditor = ReferenceSampleAuditor()
        report = auditor.audit_sample([])
        assert report.mean_cer == 0.0
        assert report.per_block_cer == []
        assert report.worst_block_indices == []

    def test_fewer_than_three_blocks(self):
        auditor = ReferenceSampleAuditor()
        pairs = [("a", "a"), ("b", "c")]
        report = auditor.audit_sample(pairs)
        assert len(report.worst_block_indices) == 2


@pytest.mark.integration
class TestReferenceSampleAuditorIntegration:
    """Требует размеченной выборки из реального production-архива."""

    def test_with_real_labeled_sample(self):
        pytest.skip("Требует размеченной выборки из production-архива — запускать вручную")
