import pytest

from app.rag.diagnostics import (
    FailureClass,
    FailureLayer,
    RAGDiagnostics,
    RAGSymptoms,
)


@pytest.mark.unit
class TestRAGDiagnosticsUnit:
    """Unit-тесты: работают без внешних сервисов."""

    # ------------------------------------------------------------------
    # Базовое поведение
    # ------------------------------------------------------------------

    def test_empty_symptoms_returns_no_results(self):
        diagnostics = RAGDiagnostics()
        results = diagnostics.classify(RAGSymptoms())
        assert results == []

    def test_result_has_non_empty_recommendation(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(index_model_mismatch=True)
        results = diagnostics.classify(symptoms)
        assert len(results) > 0
        for r in results:
            assert r.recommendation, "recommendation не должен быть пустым"

    def test_result_has_non_empty_evidence(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(index_model_mismatch=True)
        results = diagnostics.classify(symptoms)
        assert len(results) > 0
        for r in results:
            assert r.evidence, "evidence не должен быть пустым списком"

    # ------------------------------------------------------------------
    # Слой индексирования
    # ------------------------------------------------------------------

    def test_i3_detected_on_model_mismatch(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(index_model_mismatch=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.I3_INCOMPATIBLE_VECTORS in classes

    def test_i3_confidence_at_least_90(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(index_model_mismatch=True)
        results = diagnostics.classify(symptoms)
        i3 = next(r for r in results if r.failure_class == FailureClass.I3_INCOMPATIBLE_VECTORS)
        assert i3.confidence >= 0.9

    def test_i3_layer_is_indexing(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(index_model_mismatch=True)
        results = diagnostics.classify(symptoms)
        i3 = next(r for r in results if r.failure_class == FailureClass.I3_INCOMPATIBLE_VECTORS)
        assert i3.layer == FailureLayer.INDEXING

    def test_i2_detected_when_index_outdated(self):
        diagnostics = RAGDiagnostics(stale_index_threshold_days=7)
        symptoms = RAGSymptoms(index_outdated_days=30)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.I2_STALE_INDEX in classes

    def test_i2_not_detected_when_index_fresh(self):
        diagnostics = RAGDiagnostics(stale_index_threshold_days=7)
        symptoms = RAGSymptoms(index_outdated_days=3)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.I2_STALE_INDEX not in classes

    def test_i2_threshold_is_configurable(self):
        diagnostics_strict = RAGDiagnostics(stale_index_threshold_days=1)
        diagnostics_loose = RAGDiagnostics(stale_index_threshold_days=30)
        symptoms = RAGSymptoms(index_outdated_days=5)
        strict_classes = [r.failure_class for r in diagnostics_strict.classify(symptoms)]
        loose_classes = [r.failure_class for r in diagnostics_loose.classify(symptoms)]
        assert FailureClass.I2_STALE_INDEX in strict_classes
        assert FailureClass.I2_STALE_INDEX not in loose_classes

    def test_i1_detected_on_chunk_boundary_breaks(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(chunk_boundary_breaks_fact=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.I1_DESTRUCTIVE_CHUNKING in classes

    # ------------------------------------------------------------------
    # Слой поиска
    # ------------------------------------------------------------------

    def test_r1_detected_when_doc_not_in_top_k(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(relevant_doc_in_top_k=False)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.R1_LOW_RECALL in classes

    def test_r1_not_detected_when_doc_in_top_k(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(relevant_doc_in_top_k=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.R1_LOW_RECALL not in classes

    def test_r2_detected_when_precision_low(self):
        diagnostics = RAGDiagnostics(precision_threshold=0.5)
        symptoms = RAGSymptoms(precision_at_k=0.3)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.R2_LOW_PRECISION in classes

    def test_r2_not_detected_when_precision_ok(self):
        diagnostics = RAGDiagnostics(precision_threshold=0.5)
        symptoms = RAGSymptoms(precision_at_k=0.8)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.R2_LOW_PRECISION not in classes

    def test_r3_detected_on_context_overflow(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(context_tokens_exceed_limit=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.R3_CONTEXT_OVERFLOW in classes

    # ------------------------------------------------------------------
    # Слой генерации
    # ------------------------------------------------------------------

    def test_g1_detected_when_correct_context_gives_wrong_answer(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(
            correct_context_given_wrong_answer=True,
            answer_ignores_provided_context=False,
        )
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.G1_HALLUCINATION in classes

    def test_g2_detected_when_context_ignored(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(answer_ignores_provided_context=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.G2_CONTEXT_IGNORED in classes

    def test_g2_takes_priority_over_g1_when_both_signals_present(self):
        """Если модель игнорирует контекст (G-2), G-1 не должен диагностироваться."""
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(
            correct_context_given_wrong_answer=True,
            answer_ignores_provided_context=True,
        )
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.G2_CONTEXT_IGNORED in classes
        assert FailureClass.G1_HALLUCINATION not in classes

    def test_g3_detected_on_attribution_mismatch(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(attribution_mismatch=True)
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.G3_WRONG_ATTRIBUTION in classes

    # ------------------------------------------------------------------
    # Множественные классы
    # ------------------------------------------------------------------

    def test_multiple_classes_returned_for_multiple_symptoms(self):
        """Симптомы из разных слоёв → несколько диагнозов одновременно."""
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(
            chunk_boundary_breaks_fact=True,
            correct_context_given_wrong_answer=True,
        )
        results = diagnostics.classify(symptoms)
        classes = [r.failure_class for r in results]
        assert FailureClass.I1_DESTRUCTIVE_CHUNKING in classes
        assert FailureClass.G1_HALLUCINATION in classes
        assert len(results) >= 2

    def test_confidence_is_between_0_and_1(self):
        diagnostics = RAGDiagnostics()
        symptoms = RAGSymptoms(
            index_model_mismatch=True,
            relevant_doc_in_top_k=False,
            attribution_mismatch=True,
        )
        results = diagnostics.classify(symptoms)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"Уверенность вне диапазона [0, 1]: {r.confidence} для {r.failure_class}"
            )


@pytest.mark.integration
class TestRAGDiagnosticsIntegration:
    """Тесты, требующие запущенных сервисов. Запускать вручную."""

    def test_placeholder(self):
        pytest.skip(
            "Интеграционные тесты диагностики не требуют внешних сервисов — "
            "этот класс зарезервирован для будущих расширений."
        )
