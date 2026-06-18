import json
import tempfile
import os
import pytest

from app.evaluation.ragas_eval import GoldenRecord, EvaluationReport, RagasEvaluator


# ---------------------------------------------------------------------------
# Unit tests — run without any external services or API keys
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGoldenRecord:
    def test_golden_record_fields(self):
        record = GoldenRecord(
            question="Каков срок действия дебетовой карты?",
            reference_answer="Срок действия — 4 года.",
            reference_contexts=["Карта выпускается сроком на 4 года..."],
        )
        assert record.question == "Каков срок действия дебетовой карты?"
        assert len(record.reference_contexts) == 1
        assert record.metadata == {}

    def test_golden_record_with_metadata(self):
        record = GoldenRecord(
            question="Вопрос",
            reference_answer="Ответ",
            reference_contexts=["Контекст"],
            metadata={"category": "cards", "difficulty": "easy"},
        )
        assert record.metadata["category"] == "cards"


@pytest.mark.unit
class TestEvaluationReport:
    def _make_report(self, **kwargs) -> EvaluationReport:
        defaults = dict(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.78,
            num_questions=10,
        )
        defaults.update(kwargs)
        return EvaluationReport(**defaults)

    def test_passes_thresholds_all_pass(self):
        report = self._make_report()
        thresholds = {
            "faithfulness": 0.85,
            "context_precision": 0.70,
        }
        assert report.passes_thresholds(thresholds) is True

    def test_passes_thresholds_one_fails(self):
        report = self._make_report(faithfulness=0.80)
        thresholds = {"faithfulness": 0.85}
        assert report.passes_thresholds(thresholds) is False

    def test_passes_thresholds_exact_boundary(self):
        report = self._make_report(context_precision=0.80)
        thresholds = {"context_precision": 0.80}
        assert report.passes_thresholds(thresholds) is True

    def test_passes_thresholds_empty_thresholds(self):
        report = self._make_report()
        assert report.passes_thresholds({}) is True

    def test_to_dict_contains_all_metrics(self):
        report = self._make_report()
        d = report.to_dict()
        assert "faithfulness" in d
        assert "answer_relevancy" in d
        assert "context_precision" in d
        assert "context_recall" in d
        assert "num_questions" in d

    def test_to_dict_values_correct(self):
        report = self._make_report(faithfulness=0.91, num_questions=42)
        d = report.to_dict()
        assert d["faithfulness"] == 0.91
        assert d["num_questions"] == 42

    def test_to_dict_includes_per_question(self):
        report = self._make_report(per_question=[{"q": "test", "score": 0.9}])
        d = report.to_dict()
        assert "per_question" in d
        assert len(d["per_question"]) == 1


@pytest.mark.unit
class TestRagasEvaluatorFromJson:
    def _write_golden_json(self, records: list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    def test_from_json_loads_records(self, tmp_path):
        data = [
            {
                "question": "Что такое RAG?",
                "reference_answer": "RAG — Retrieval Augmented Generation.",
                "reference_contexts": ["RAG объединяет поиск и генерацию..."],
            }
        ]
        path = str(tmp_path / "golden.json")
        self._write_golden_json(data, path)

        evaluator = RagasEvaluator.from_json(path)
        assert len(evaluator.golden_records) == 1
        assert evaluator.golden_records[0].question == "Что такое RAG?"

    def test_from_json_multiple_records(self, tmp_path):
        data = [
            {
                "question": f"Вопрос {i}",
                "reference_answer": f"Ответ {i}",
                "reference_contexts": [f"Контекст {i}"],
            }
            for i in range(5)
        ]
        path = str(tmp_path / "golden.json")
        self._write_golden_json(data, path)

        evaluator = RagasEvaluator.from_json(path)
        assert len(evaluator.golden_records) == 5

    def test_from_json_preserves_metadata(self, tmp_path):
        data = [
            {
                "question": "Вопрос",
                "reference_answer": "Ответ",
                "reference_contexts": ["Контекст"],
                "metadata": {"category": "billing"},
            }
        ]
        path = str(tmp_path / "golden.json")
        self._write_golden_json(data, path)

        evaluator = RagasEvaluator.from_json(path)
        assert evaluator.golden_records[0].metadata["category"] == "billing"

    def test_from_json_multiple_contexts(self, tmp_path):
        data = [
            {
                "question": "Сложный вопрос",
                "reference_answer": "Ответ",
                "reference_contexts": ["Контекст 1", "Контекст 2", "Контекст 3"],
            }
        ]
        path = str(tmp_path / "golden.json")
        self._write_golden_json(data, path)

        evaluator = RagasEvaluator.from_json(path)
        assert len(evaluator.golden_records[0].reference_contexts) == 3


@pytest.mark.unit
class TestSaveReport:
    def test_save_report_creates_file(self, tmp_path):
        report = EvaluationReport(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.78,
            num_questions=10,
        )
        evaluator = RagasEvaluator(golden_records=[])
        path = str(tmp_path / "reports" / "report.json")

        evaluator.save_report(report, path)

        assert os.path.exists(path)

    def test_save_report_valid_json(self, tmp_path):
        report = EvaluationReport(
            faithfulness=0.92,
            answer_relevancy=0.88,
            context_precision=0.83,
            context_recall=0.79,
            num_questions=20,
        )
        evaluator = RagasEvaluator(golden_records=[])
        path = str(tmp_path / "report.json")

        evaluator.save_report(report, path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["faithfulness"] == 0.92
        assert data["num_questions"] == 20


# ---------------------------------------------------------------------------
# Integration tests — require OPENAI_API_KEY and real retriever/generator
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRagasEvaluatorIntegration:
    def test_run_full_pipeline(self):
        pytest.skip(
            "Требует OPENAI_API_KEY и реальные компоненты системы — запускайте вручную"
        )
