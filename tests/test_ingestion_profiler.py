"""Tests for app/ingestion/pipeline_profiler.py (Lesson 5.1).

Unit tests use a lightweight fake pipeline (not the real
MinimalIngestionPipeline, which the student implements in Lesson 1.4)
so they exercise IngestionStageProfiler in isolation. Real timing
uses small time.sleep() calls, following the same pattern as
tests/test_profiler.py (Lesson 2.7.1, app/rag/profiler.py).

Run unit tests only:
    pytest tests/test_ingestion_profiler.py -v -m unit
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pipeline_profiler import IngestionStageProfiler, StageTiming


# ---------------------------------------------------------------------------
# Fake pipeline helper
# ---------------------------------------------------------------------------

class _FakePipeline:
    """Duck-typed stand-in for MinimalIngestionPipeline.

    Avoids depending on the (student-implemented) real pipeline so
    these tests exercise only the profiler's own logic.
    """

    def __init__(self, text_by_file: dict[str, str], chunks: list[str], sleep_s: float = 0.0):
        self._text_by_file = text_by_file
        self._sleep_s = sleep_s
        self.chunker = MagicMock()
        self.chunker.split.side_effect = lambda text: list(chunks)
        self.vector_store = MagicMock()

    def load_text(self, path: Path) -> str:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return self._text_by_file[path.name]

    def discover_files(self) -> list[Path]:
        return [Path(name) for name in sorted(self._text_by_file)]


def _patched_embedding(vector=(0.1, 0.2), sleep_s: float = 0.0):
    def _fn(text: str):
        if sleep_s:
            time.sleep(sleep_s)
        return list(vector)

    return _fn


# ---------------------------------------------------------------------------
# profile_file
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProfileFile:
    def test_records_read_chunk_embed_upsert_stages(self):
        pipeline = _FakePipeline({"doc.txt": "какой-то текст"}, chunks=["чанк1", "чанк2"])
        profiler = IngestionStageProfiler(pipeline)

        with patch(
            "app.ingestion.pipeline_profiler.get_embedding",
            side_effect=_patched_embedding(),
        ):
            n = profiler.profile_file(Path("doc.txt"))

        assert n == 2
        stages = [t.stage for t in profiler._timings]
        assert stages.count("read") == 1
        assert stages.count("chunk") == 1
        assert stages.count("embed") == 2  # один замер на чанк
        assert stages.count("upsert") == 1
        pipeline.vector_store.upsert_documents.assert_called_once()

    def test_empty_text_returns_zero_without_further_stages(self):
        pipeline = _FakePipeline({"empty.txt": "   \n  "}, chunks=["чанк"])
        profiler = IngestionStageProfiler(pipeline)

        n = profiler.profile_file(Path("empty.txt"))

        assert n == 0
        stages = [t.stage for t in profiler._timings]
        assert "read" in stages
        assert "chunk" not in stages
        assert "embed" not in stages
        pipeline.vector_store.upsert_documents.assert_not_called()

    def test_no_chunks_returns_zero_without_embed_or_upsert(self):
        pipeline = _FakePipeline({"doc.txt": "текст"}, chunks=[])
        profiler = IngestionStageProfiler(pipeline)

        n = profiler.profile_file(Path("doc.txt"))

        assert n == 0
        stages = [t.stage for t in profiler._timings]
        assert "embed" not in stages
        pipeline.vector_store.upsert_documents.assert_not_called()


# ---------------------------------------------------------------------------
# profile_run
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProfileRun:
    def test_aggregates_across_multiple_files(self):
        pipeline = _FakePipeline(
            {"a.txt": "текст a", "b.txt": "текст b"}, chunks=["чанк"]
        )
        profiler = IngestionStageProfiler(pipeline)

        with patch(
            "app.ingestion.pipeline_profiler.get_embedding",
            side_effect=_patched_embedding(),
        ):
            report = profiler.profile_run()

        assert len(report.files_processed) == 2
        assert report.chunks_indexed == 2

    def test_skips_file_with_bad_encoding_without_failing_run(self):
        class _RaisingPipeline(_FakePipeline):
            def load_text(self, path: Path) -> str:
                if path.name == "bad.txt":
                    raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
                return super().load_text(path)

        pipeline = _RaisingPipeline({"good.txt": "нормальный текст", "bad.txt": ""}, chunks=["чанк"])
        profiler = IngestionStageProfiler(pipeline)

        with patch(
            "app.ingestion.pipeline_profiler.get_embedding",
            side_effect=_patched_embedding(),
        ):
            report = profiler.profile_run()

        assert len(report.files_processed) == 1
        assert len(report.files_skipped) == 1
        assert "bad.txt" in report.files_skipped[0]


# ---------------------------------------------------------------------------
# stage_breakdown / bottleneck_stage
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStageBreakdown:
    def test_empty_timings_returns_empty_dict(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())
        assert profiler.stage_breakdown() == {}

    def test_bottleneck_stage_none_without_timings(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())
        assert profiler.bottleneck_stage() is None

    def test_breakdown_contains_required_keys(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())
        profiler._timed("read", "a.txt", lambda: None)
        breakdown = profiler.stage_breakdown()
        entry = breakdown["read"]
        for key in ("count", "total_ms", "mean_ms", "share_pct"):
            assert key in entry

    def test_share_pct_sums_to_roughly_100(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())
        profiler._timed("read", "a.txt", lambda: time.sleep(0.001))
        profiler._timed("embed", "a.txt", lambda: time.sleep(0.003))
        breakdown = profiler.stage_breakdown()
        total_share = sum(entry["share_pct"] for entry in breakdown.values())
        assert 99.0 <= total_share <= 101.0

    def test_bottleneck_stage_identifies_slowest_by_total_time(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())
        # "chunk" стадия быстрая, но вызывается много раз с маленькой задержкой каждый.
        for _ in range(3):
            profiler._timed("chunk", "a.txt", lambda: time.sleep(0.001))
        # "embed" стадия — один заметно более долгий вызов, должен доминировать по total_ms.
        profiler._timed("embed", "a.txt", lambda: time.sleep(0.030))

        assert profiler.bottleneck_stage() == "embed"

    def test_timing_recorded_even_when_fn_raises(self):
        profiler = IngestionStageProfiler(pipeline=MagicMock())

        def _boom():
            raise ValueError("сбой стадии")

        with pytest.raises(ValueError):
            profiler._timed("read", "broken.txt", _boom)

        breakdown = profiler.stage_breakdown()
        assert breakdown["read"]["count"] == 1


# ---------------------------------------------------------------------------
# Integration — requires a real embedding model / Qdrant instance
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIngestionStageProfilerIntegration:
    """Требует настоящий MinimalIngestionPipeline с рабочим Qdrant.
    Скип по умолчанию: pytest -m 'not integration'."""

    def test_profile_run_against_real_pipeline(self):
        pytest.skip(
            "Требует запущенный Qdrant и настоящий MinimalIngestionPipeline — "
            "запускайте вручную:\n"
            "  pytest tests/test_ingestion_profiler.py -m integration -v"
        )
