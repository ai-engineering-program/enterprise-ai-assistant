from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ingestion.image_preprocessor import ImagePreprocessor, PreprocessingConfig


def _straight_text_block(size: int = 200) -> np.ndarray:
    """Белый холст с горизонтальным чёрным прямоугольником — упрощённая
    заглушка "строки текста", уже выровненной по горизонтали."""
    canvas = np.full((size, size), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 90), (160, 110), color=0, thickness=-1)
    return canvas


def _rotated_text_block(angle_deg: float, size: int = 200) -> np.ndarray:
    """Тот же холст, повёрнутый на заданный угол — имитация перекоса
    страницы при сканировании."""
    canvas = _straight_text_block(size)
    center = (size / 2, size / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        matrix,
        (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated


@pytest.mark.unit
class TestConfig:
    def test_default_config_used_when_none_provided(self):
        preprocessor = ImagePreprocessor()
        assert preprocessor.config.denoise_kernel_size == 3
        assert preprocessor.config.binarize_block_size == 25

    def test_custom_config_is_respected(self):
        config = PreprocessingConfig(denoise_kernel_size=5, binarize_c=10)
        preprocessor = ImagePreprocessor(config)
        assert preprocessor.config.denoise_kernel_size == 5
        assert preprocessor.config.binarize_c == 10


@pytest.mark.unit
class TestEstimateSkewAngle:
    def test_straight_block_has_near_zero_angle(self):
        preprocessor = ImagePreprocessor()
        angle = preprocessor.estimate_skew_angle(_straight_text_block())
        assert abs(angle) < 1.0

    def test_blank_page_returns_zero(self):
        preprocessor = ImagePreprocessor()
        blank = np.full((100, 100), 255, dtype=np.uint8)
        assert preprocessor.estimate_skew_angle(blank) == 0.0


@pytest.mark.unit
class TestDeskew:
    def test_straight_image_is_returned_unchanged(self):
        preprocessor = ImagePreprocessor()
        image = _straight_text_block()
        result = preprocessor.deskew(image)
        assert np.array_equal(result, image)

    def test_deskew_reduces_measured_skew(self):
        preprocessor = ImagePreprocessor()
        rotated = _rotated_text_block(angle_deg=12.0)
        angle_before = preprocessor.estimate_skew_angle(rotated)
        deskewed = preprocessor.deskew(rotated)
        angle_after = preprocessor.estimate_skew_angle(deskewed)
        assert abs(angle_before) > 1.0
        assert abs(angle_after) < abs(angle_before)

    def test_deskew_preserves_shape(self):
        preprocessor = ImagePreprocessor()
        rotated = _rotated_text_block(angle_deg=7.0)
        result = preprocessor.deskew(rotated)
        assert result.shape == rotated.shape


@pytest.mark.unit
class TestDenoise:
    def test_removes_isolated_noise_pixel(self):
        preprocessor = ImagePreprocessor()
        image = np.full((50, 50), 255, dtype=np.uint8)
        image[25, 25] = 0  # изолированный "шумовой" пиксель факса
        result = preprocessor.denoise(image)
        assert result[25, 25] == 255

    def test_untouched_background_stays_uniform(self):
        preprocessor = ImagePreprocessor()
        image = np.full((50, 50), 255, dtype=np.uint8)
        image[25, 25] = 0
        result = preprocessor.denoise(image)
        assert result[5, 5] == 255
        assert result[45, 45] == 255


@pytest.mark.unit
class TestBinarize:
    def test_output_is_strictly_binary(self):
        preprocessor = ImagePreprocessor()
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(60, 60), dtype=np.uint8)
        result = preprocessor.binarize(image)
        assert set(np.unique(result).tolist()).issubset({0, 255})

    def test_preserves_shape_and_dtype(self):
        preprocessor = ImagePreprocessor()
        image = np.full((30, 40), 200, dtype=np.uint8)
        result = preprocessor.binarize(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


@pytest.mark.unit
class TestPreprocessPipeline:
    def test_full_pipeline_returns_binary_image_of_same_shape(self):
        preprocessor = ImagePreprocessor()
        image = _rotated_text_block(angle_deg=8.0)
        result = preprocessor.preprocess(image)
        assert result.shape == image.shape
        assert set(np.unique(result).tolist()).issubset({0, 255})


@pytest.mark.integration
class TestImagePreprocessorIntegration:
    """Требует набора реальных сканов из архива для сравнения точности
    распознавания до/после предобработки (см. таблицу в тексте урока)."""

    def test_real_scan_accuracy_improvement(self):
        pytest.skip("Требует корпуса реальных сканов и Tesseract — запускать вручную")
