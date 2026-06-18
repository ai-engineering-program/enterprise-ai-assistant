import json
import os
import tempfile
import time

import pytest

from app.rag.tracer import RAGTracer, Span, Trace


@pytest.mark.unit
class TestSpanUnit:
    """Unit-тесты для класса Span — без внешних сервисов."""

    def test_span_starts_without_ended_at(self):
        from datetime import datetime, timezone
        span = Span(
            name="retrieval",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        assert span.ended_at is None
        assert span.duration_ms is None

    def test_span_finish_sets_ended_at(self):
        from datetime import datetime, timezone
        span = Span(
            name="retrieval",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        span.finish(outputs={"chunk_ids": ["a", "b"]})
        assert span.ended_at is not None
        assert span.outputs == {"chunk_ids": ["a", "b"]}

    def test_span_duration_ms_positive(self):
        from datetime import datetime, timezone
        span = Span(
            name="generation",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        time.sleep(0.02)  # 20 ms
        span.finish(outputs={"answer": "42"})
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_finish_with_metadata(self):
        from datetime import datetime, timezone
        span = Span(
            name="retrieval",
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata={"model": "all-MiniLM-L6-v2"},
        )
        span.finish(outputs={}, metadata={"k": 5})
        assert span.metadata.get("model") == "all-MiniLM-L6-v2"
        assert span.metadata.get("k") == 5


@pytest.mark.unit
class TestTraceUnit:
    """Unit-тесты для класса Trace — без внешних сервисов."""

    def test_to_dict_contains_required_fields(self):
        from datetime import datetime, timezone
        trace = Trace(
            trace_id="test-id-123",
            query="что такое RAG?",
            created_at=datetime.now(timezone.utc).isoformat(),
            final_answer="RAG — это Retrieval Augmented Generation.",
        )
        d = trace.to_dict()
        assert d["trace_id"] == "test-id-123"
        assert d["query"] == "что такое RAG?"
        assert d["final_answer"] == "RAG — это Retrieval Augmented Generation."
        assert "spans" in d
        assert isinstance(d["spans"], list)

    def test_to_dict_spans_include_duration(self):
        from datetime import datetime, timezone
        span = Span(
            name="retrieval",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        span.finish(outputs={"ids": []})
        trace = Trace(
            trace_id="t1",
            query="тест",
            created_at=datetime.now(timezone.utc).isoformat(),
            spans=[span],
        )
        d = trace.to_dict()
        assert len(d["spans"]) == 1
        assert "duration_ms" in d["spans"][0]
        assert "name" in d["spans"][0]
        assert "inputs" in d["spans"][0]
        assert "outputs" in d["spans"][0]


@pytest.mark.unit
class TestRAGTracerUnit:
    """Unit-тесты для RAGTracer — без Qdrant и внешних API."""

    def test_start_trace_returns_trace_with_unique_id(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            t1 = tracer.start_trace("первый вопрос")
            t2 = tracer.start_trace("второй вопрос")
            assert isinstance(t1, Trace)
            assert isinstance(t2, Trace)
            assert t1.trace_id != t2.trace_id
        finally:
            os.unlink(path)

    def test_start_trace_sets_query(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            trace = tracer.start_trace("какой срок гарантии?")
            assert trace.query == "какой срок гарантии?"
        finally:
            os.unlink(path)

    def test_start_span_adds_to_current_trace(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("тест")
            span = tracer.start_span("retrieval", inputs={"k": 5})
            assert isinstance(span, Span)
            assert span.name == "retrieval"
            assert span.inputs == {"k": 5}
            assert len(tracer._current_trace.spans) == 1
        finally:
            os.unlink(path)

    def test_finish_span_sets_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("тест")
            span = tracer.start_span("retrieval")
            tracer.finish_span(span, outputs={"chunk_ids": ["doc1_0", "doc2_3"]})
            assert span.outputs == {"chunk_ids": ["doc1_0", "doc2_3"]}
            assert span.ended_at is not None
        finally:
            os.unlink(path)

    def test_finish_trace_writes_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                         mode="w") as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("вопрос для лога")
            tracer.finish_trace(answer="ответ системы")

            with open(path, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["query"] == "вопрос для лога"
            assert data["final_answer"] == "ответ системы"
        finally:
            os.unlink(path)

    def test_finish_trace_two_queries_two_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                         mode="w") as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("первый")
            tracer.finish_trace(answer="ответ 1")
            tracer.start_trace("второй")
            tracer.finish_trace(answer="ответ 2")

            with open(path, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            assert len(lines) == 2
        finally:
            os.unlink(path)

    def test_finish_trace_resets_current_trace(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("тест сброса")
            tracer.finish_trace(answer="готово")
            assert tracer._current_trace is None
        finally:
            os.unlink(path)

    def test_finish_trace_with_error(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                         mode="w") as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("сбойный запрос")
            tracer.finish_trace(error="ConnectionError: Qdrant недоступен")

            with open(path, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            data = json.loads(lines[0])
            assert data["error"] == "ConnectionError: Qdrant недоступен"
            assert data["final_answer"] is None
        finally:
            os.unlink(path)

    def test_full_pipeline_trace_structure(self):
        """Полный цикл трассировки с несколькими периодами."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                         mode="w") as f:
            path = f.name
        try:
            tracer = RAGTracer(log_path=path)
            tracer.start_trace("каков ток нагрузки реле?")

            span_r = tracer.start_span("retrieval", inputs={"k": 5})
            tracer.finish_span(span_r, outputs={
                "chunk_ids": ["relay_spec_chunk_12", "relay_spec_chunk_13"],
                "scores": [0.91, 0.87],
            })

            span_g = tracer.start_span("generation", inputs={"context_tokens": 512})
            tracer.finish_span(span_g, outputs={"answer": "60 А при температуре 25°C"})

            finished = tracer.finish_trace(answer="60 А при температуре 25°C")

            assert len(finished.spans) == 2
            assert finished.spans[0].name == "retrieval"
            assert finished.spans[1].name == "generation"

            with open(path, encoding="utf-8") as fh:
                data = json.loads(fh.readline())
            assert len(data["spans"]) == 2
            assert data["spans"][0]["outputs"]["chunk_ids"][0] == "relay_spec_chunk_12"
        finally:
            os.unlink(path)


@pytest.mark.integration
class TestRAGTracerIntegration:
    """Интеграционные тесты — требуют запущенный Qdrant и реальный ретривер."""

    def test_trace_with_real_retriever(self):
        pytest.skip(
            "Требует запущенный Qdrant. "
            "Запустите вручную: pytest tests/test_tracer.py -v -m integration"
        )
