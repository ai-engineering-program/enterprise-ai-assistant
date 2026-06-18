import pytest

from app.rag.analyzer import (
    GoldenItem,
    RetrievalAnalyzer,
    RetrievalReport,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------


def _make_result(doc_id: str, score: float = 0.8, **metadata) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#1",
        score=score,
        text_preview=f"Текст фрагмента {doc_id}",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Unit-тесты — не требуют внешних сервисов
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeFound:
    """Сценарии, когда нужный документ присутствует в результатах."""

    def test_found_on_first_position(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_55", 0.91),
            _make_result("doc_42", 0.75),
        ]
        report = analyzer.analyze("запрос", results, expected_doc_id="doc_55")
        assert report.found is True
        assert report.rank == 1
        assert report.low_recall is False

    def test_found_on_second_position(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_42", 0.82),
            _make_result("doc_55", 0.79),
        ]
        report = analyzer.analyze("запрос", results, expected_doc_id="doc_55")
        assert report.found is True
        assert report.rank == 2
        assert report.low_recall is False

    def test_found_at_last_position(self):
        analyzer = RetrievalAnalyzer()
        results = [_make_result(f"doc_{i}", 0.9 - i * 0.05) for i in range(5)]
        results[-1] = _make_result("doc_target", 0.65)
        report = analyzer.analyze("запрос", results, expected_doc_id="doc_target")
        assert report.found is True
        assert report.rank == 5


@pytest.mark.unit
class TestAnalyzeLowRecall:
    """Сценарии низкой полноты (нужный документ не найден)."""

    def test_low_recall_when_expected_absent(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_42", 0.71),
            _make_result("doc_7", 0.68),
        ]
        report = analyzer.analyze("запрос", results, expected_doc_id="doc_55")
        assert report.found is False
        assert report.rank is None
        assert report.low_recall is True

    def test_no_low_recall_without_expected_doc(self):
        """Если expected_doc_id не задан — low_recall не выставляется."""
        analyzer = RetrievalAnalyzer()
        results = [_make_result("doc_42")]
        report = analyzer.analyze("запрос", results, expected_doc_id=None)
        assert report.low_recall is False

    def test_empty_results_with_expected(self):
        analyzer = RetrievalAnalyzer()
        report = analyzer.analyze("запрос", [], expected_doc_id="doc_55")
        assert report.found is False
        assert report.low_recall is True
        assert report.rank is None


@pytest.mark.unit
class TestAnalyzeContextPollution:
    """Сценарии засорения контекста (конфликты версий)."""

    def test_conflict_detected_different_versions(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_55", 0.81,
                         base_doc_id="reg_101", document_version="2022"),
            _make_result("doc_55v2", 0.78,
                         base_doc_id="reg_101", document_version="2024"),
        ]
        report = analyzer.analyze("запрос", results)
        assert report.context_pollution is True
        assert len(report.conflicts) == 1

    def test_no_conflict_same_version(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_a", 0.81,
                         base_doc_id="reg_101", document_version="2024"),
            _make_result("doc_b", 0.78,
                         base_doc_id="reg_101", document_version="2024"),
        ]
        report = analyzer.analyze("запрос", results)
        assert report.context_pollution is False
        assert len(report.conflicts) == 0

    def test_no_conflict_different_base_docs(self):
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_a", 0.81,
                         base_doc_id="reg_101", document_version="2022"),
            _make_result("doc_b", 0.78,
                         base_doc_id="reg_202", document_version="2024"),
        ]
        report = analyzer.analyze("запрос", results)
        assert report.context_pollution is False

    def test_no_conflict_missing_metadata(self):
        """Фрагменты без metadata["base_doc_id"] не порождают конфликт."""
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_a", 0.81),
            _make_result("doc_b", 0.78),
        ]
        report = analyzer.analyze("запрос", results)
        assert report.context_pollution is False

    def test_multiple_conflicts(self):
        """Три версии одного документа образуют три пары конфликтов."""
        analyzer = RetrievalAnalyzer()
        results = [
            _make_result("doc_v1", 0.85,
                         base_doc_id="reg_101", document_version="2020"),
            _make_result("doc_v2", 0.82,
                         base_doc_id="reg_101", document_version="2022"),
            _make_result("doc_v3", 0.79,
                         base_doc_id="reg_101", document_version="2024"),
        ]
        report = analyzer.analyze("запрос", results)
        assert report.context_pollution is True
        assert len(report.conflicts) == 3

    def test_empty_results_no_pollution(self):
        analyzer = RetrievalAnalyzer()
        report = analyzer.analyze("запрос", [])
        assert report.context_pollution is False
        assert report.conflicts == []


@pytest.mark.unit
class TestComputeRecallAtK:
    """Тесты для compute_recall_at_k."""

    def test_perfect_recall(self):
        analyzer = RetrievalAnalyzer()
        golden = [
            GoldenItem("вопрос 1", "doc_1"),
            GoldenItem("вопрос 2", "doc_2"),
        ]

        def search_fn(query, k):
            if "1" in query:
                return [_make_result("doc_1")]
            return [_make_result("doc_2")]

        recall = analyzer.compute_recall_at_k(golden, search_fn, k=5)
        assert recall == 1.0

    def test_zero_recall(self):
        analyzer = RetrievalAnalyzer()
        golden = [GoldenItem("вопрос", "doc_target")]

        def search_fn(query, k):
            return [_make_result("doc_wrong")]

        recall = analyzer.compute_recall_at_k(golden, search_fn, k=5)
        assert recall == 0.0

    def test_partial_recall(self):
        analyzer = RetrievalAnalyzer()
        golden = [
            GoldenItem("вопрос А", "doc_A"),
            GoldenItem("вопрос Б", "doc_B"),
        ]

        def search_fn(query, k):
            if "А" in query:
                return [_make_result("doc_A")]
            return [_make_result("doc_wrong")]

        recall = analyzer.compute_recall_at_k(golden, search_fn, k=5)
        assert recall == pytest.approx(0.5)

    def test_empty_golden_set(self):
        analyzer = RetrievalAnalyzer()
        recall = analyzer.compute_recall_at_k([], lambda q, k: [], k=5)
        assert recall == 0.0

    def test_recall_respects_k(self):
        """Документ на позиции k+1 не учитывается."""
        analyzer = RetrievalAnalyzer()
        golden = [GoldenItem("вопрос", "doc_target")]

        def search_fn(query, k):
            # Возвращаем target только если k >= 3
            if k >= 3:
                return [
                    _make_result("doc_1"),
                    _make_result("doc_2"),
                    _make_result("doc_target"),
                ]
            return [_make_result("doc_1"), _make_result("doc_2")]

        assert analyzer.compute_recall_at_k(golden, search_fn, k=2) == 0.0
        assert analyzer.compute_recall_at_k(golden, search_fn, k=3) == 1.0


@pytest.mark.unit
class TestReportStructure:
    """Проверки структуры возвращаемого объекта."""

    def test_returns_retrieval_report(self):
        analyzer = RetrievalAnalyzer()
        report = analyzer.analyze("запрос", [])
        assert isinstance(report, RetrievalReport)

    def test_report_contains_query(self):
        analyzer = RetrievalAnalyzer()
        report = analyzer.analyze("тестовый запрос", [])
        assert report.query == "тестовый запрос"

    def test_report_contains_results(self):
        analyzer = RetrievalAnalyzer()
        results = [_make_result("doc_1")]
        report = analyzer.analyze("запрос", results)
        assert report.results == results


# ---------------------------------------------------------------------------
# Integration-тесты — требуют реального Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRetrievalAnalyzerIntegration:
    """Тесты с реальным Qdrant. Запускать вручную."""

    def test_with_real_qdrant(self):
        pytest.skip("Требует запущенный Qdrant — запустите вручную")
