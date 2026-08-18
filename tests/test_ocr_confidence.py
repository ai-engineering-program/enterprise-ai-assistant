import pytest

from app.ingestion.ocr_confidence import OCRBlock, OCRConfidenceGate


@pytest.mark.unit
class TestAverageConfidence:
    def test_basic_average(self):
        gate = OCRConfidenceGate()
        block = OCRBlock("b1", "текст блока", [0.9, 0.8, 1.0])
        assert gate.average_confidence(block) == pytest.approx(0.9)

    def test_empty_confidences_returns_zero(self):
        gate = OCRConfidenceGate()
        block = OCRBlock("b1", "", [])
        assert gate.average_confidence(block) == 0.0


@pytest.mark.unit
class TestSuspiciousTokens:
    def test_detects_mixed_digit_letter_token(self):
        gate = OCRConfidenceGate()
        # "п.9.7" не подозрителен сам по себе — только смешение цифры и
        # похожей буквы внутри одного токена
        text = "согласно п. О58217 договора"
        suspicious = gate.find_suspicious_tokens(text)
        assert "О58217" in suspicious

    def test_clean_text_has_no_suspicious_tokens(self):
        gate = OCRConfidenceGate()
        text = "согласно пункту 6.4 договора штраф составляет 2 процента"
        assert gate.find_suspicious_tokens(text) == []

    def test_multiple_suspicious_pairs(self):
        gate = OCRConfidenceGate()
        text = "номер б4 и код S3 требуют проверки"
        suspicious = gate.find_suspicious_tokens(text)
        assert "б4" in suspicious
        assert "S3" in suspicious


@pytest.mark.unit
class TestEvaluate:
    def test_high_confidence_clean_text_is_indexed(self):
        gate = OCRConfidenceGate(confidence_threshold=0.85)
        block = OCRBlock("b1", "штраф составляет 2 процента", [0.97, 0.95, 0.99])
        decision = gate.evaluate(block)
        assert decision.action == "index"
        assert decision.suspicious_tokens == []

    def test_low_confidence_is_quarantined(self):
        gate = OCRConfidenceGate(confidence_threshold=0.85)
        block = OCRBlock("b1", "штраф составляет 2 процента", [0.4, 0.5, 0.6])
        decision = gate.evaluate(block)
        assert decision.action == "quarantine"

    def test_suspicious_token_quarantines_even_with_high_confidence(self):
        gate = OCRConfidenceGate(confidence_threshold=0.85)
        # Движок был "уверен" в каждом символе по отдельности, но токен
        # всё равно смешивает цифру и похожую букву — то самое ложное
        # ощущение надёжности, о котором говорится в уроке.
        block = OCRBlock("b1", "см. приложение О58217", [0.96, 0.95, 0.94])
        decision = gate.evaluate(block)
        assert decision.action == "quarantine"
        assert "О58217" in decision.suspicious_tokens


@pytest.mark.unit
class TestFilterBlocks:
    def test_splits_index_and_quarantine(self):
        gate = OCRConfidenceGate(confidence_threshold=0.85)
        blocks = [
            OCRBlock("good", "штраф два процента", [0.97, 0.96, 0.95]),
            OCRBlock("bad_conf", "штраф два процента", [0.3, 0.4, 0.2]),
            OCRBlock("bad_token", "приложение О58217", [0.97, 0.96]),
        ]
        to_index, decisions = gate.filter_blocks(blocks)

        assert [b.block_id for b in to_index] == ["good"]
        assert len(decisions) == 3
        actions = {d.block_id: d.action for d in decisions}
        assert actions["good"] == "index"
        assert actions["bad_conf"] == "quarantine"
        assert actions["bad_token"] == "quarantine"


@pytest.mark.integration
class TestOCRConfidenceIntegration:
    """Требует реального OCR-движка (например Tesseract) и тестовых сканов."""

    def test_real_tesseract_confidence_output(self):
        pytest.skip("Требует установленного Tesseract и тестовых изображений — запускать вручную")
