import math
import pytest
from unittest.mock import patch

from app.rag.embedding_utils import cosine_similarity, semantic_test


# ---------------------------------------------------------------------------
# Вспомогательные векторы для unit-тестов (не требуют внешних сервисов)
# ---------------------------------------------------------------------------

def _unit_vec(components: list[float]) -> list[float]:
    """Нормализовать вектор до единичной длины."""
    norm = math.sqrt(sum(x * x for x in components))
    return [x / norm for x in components] if norm > 0 else components


VEC_A = _unit_vec([1.0, 0.0, 0.0])   # единичный вектор вдоль оси X
VEC_B = _unit_vec([1.0, 0.0, 0.0])   # идентичен A → сходство 1.0
VEC_C = _unit_vec([0.0, 1.0, 0.0])   # ортогонален A → сходство 0.0
VEC_D = _unit_vec([-1.0, 0.0, 0.0])  # противоположен A → сходство -1.0


# ---------------------------------------------------------------------------
# Unit-тесты для cosine_similarity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineSimilarity:
    """Тесты функции cosine_similarity без внешних зависимостей."""

    def test_identical_vectors_return_one(self):
        result = cosine_similarity(VEC_A, VEC_B)
        assert abs(result - 1.0) < 1e-6, f"Ожидалось 1.0, получено {result}"

    def test_orthogonal_vectors_return_zero(self):
        result = cosine_similarity(VEC_A, VEC_C)
        assert abs(result - 0.0) < 1e-6, f"Ожидалось 0.0, получено {result}"

    def test_opposite_vectors_return_minus_one(self):
        result = cosine_similarity(VEC_A, VEC_D)
        assert abs(result - (-1.0)) < 1e-6, f"Ожидалось -1.0, получено {result}"

    def test_zero_vector_returns_zero(self):
        zero = [0.0, 0.0, 0.0]
        result = cosine_similarity(zero, VEC_A)
        assert result == 0.0, "Нулевой вектор должен давать сходство 0.0"

    def test_symmetric(self):
        v1 = _unit_vec([1.0, 2.0, 3.0])
        v2 = _unit_vec([3.0, 2.0, 1.0])
        assert abs(cosine_similarity(v1, v2) - cosine_similarity(v2, v1)) < 1e-9


# ---------------------------------------------------------------------------
# Unit-тесты для semantic_test (с мокированием get_embedding)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSemanticTestUnit:
    """Тесты semantic_test с замоканным get_embedding."""

    def test_high_similarity_label(self):
        # Два одинаковых вектора → score ~ 1.0 → label "высокое"
        with patch("app.rag.embedding_utils.get_embedding", return_value=VEC_A):
            result = semantic_test("текст A", "текст A")
        assert result["label"] == "высокое"
        assert result["score"] > 0.8

    def test_low_similarity_label(self):
        # Ортогональные векторы → score ~ 0.0 → label "низкое"
        vecs = iter([VEC_A, VEC_C])
        with patch("app.rag.embedding_utils.get_embedding", side_effect=lambda t: next(vecs)):
            result = semantic_test("текст A", "текст C")
        assert result["label"] == "низкое"
        assert result["score"] < 0.5

    def test_medium_similarity_label(self):
        # Векторы с умеренным сходством → label "среднее"
        v1 = _unit_vec([1.0, 0.5, 0.0])
        v2 = _unit_vec([0.5, 1.0, 0.0])
        vecs = iter([v1, v2])
        with patch("app.rag.embedding_utils.get_embedding", side_effect=lambda t: next(vecs)):
            result = semantic_test("текст 1", "текст 2")
        assert result["label"] in ("среднее", "высокое")
        assert 0.0 <= result["score"] <= 1.0

    def test_empty_string_returns_low(self):
        result = semantic_test("", "текст")
        assert result == {"score": 0.0, "label": "низкое"}

    def test_both_empty_returns_low(self):
        result = semantic_test("", "")
        assert result == {"score": 0.0, "label": "низкое"}

    def test_return_type(self):
        with patch("app.rag.embedding_utils.get_embedding", return_value=VEC_A):
            result = semantic_test("a", "b")
        assert isinstance(result, dict)
        assert "score" in result
        assert "label" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["label"], str)


# ---------------------------------------------------------------------------
# Integration-тесты (требуют установленной sentence-transformers или OPENAI_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSemanticTestIntegration:
    """Тесты с реальной моделью векторных представлений.

    Запуск: pytest tests/test_embedding_utils.py -v -m integration
    Требуется: pip install sentence-transformers
    """

    def test_synonyms_score_high(self):
        result = semantic_test(
            "Как получить потребительский кредит",
            "Как оформить займ для физических лиц"
        )
        assert result["label"] == "высокое", (
            f"Синонимичные запросы должны давать высокое сходство, "
            f"получено {result['score']:.4f} ({result['label']})"
        )

    def test_antonyms_score_not_low(self):
        # Антонимы должны давать ВЫСОКОЕ сходство (демонстрация ловушки)
        result = semantic_test(
            "Ваша заявка на кредит успешно одобрена",
            "Ваша заявка на кредит отклонена"
        )
        # Ловушка: ожидаем НЕ низкое — модель не различает одобрение и отказ
        assert result["score"] > 0.7, (
            f"Антонимы в одном контексте должны давать высокое сходство — это ловушка. "
            f"Получено {result['score']:.4f}"
        )

    def test_different_domains_score_low(self):
        result = semantic_test(
            "Техническая поддержка по вопросам личного кабинета",
            "Ипотека на квартиру в Москве"
        )
        assert result["label"] == "низкое", (
            f"Тексты разных доменов должны давать низкое сходство, "
            f"получено {result['score']:.4f} ({result['label']})"
        )
