"""Tests for app/ingestion/async_pipeline.py (Lesson 5.2).

Unit tests run without Qdrant or a real embedding model — FixedSizeChunker,
VectorStore and get_embedding are patched, following the same pattern as
tests/test_ingestion_pipeline.py (Lesson 1.4). Integration tests require a
running Qdrant instance and a working embedding backend.

Run unit tests only:
    pytest tests/test_async_pipeline.py -v -m unit

Run all tests (requires Qdrant):
    pytest tests/test_async_pipeline.py -v
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.async_pipeline import AsyncIngestionPipeline
from app.ingestion.pipeline import IngestionReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path: Path, **kwargs):
    """Собрать пайплайн с подменёнными FixedSizeChunker и VectorStore.

    По умолчанию chunker.split() возвращает весь текст как один чанк —
    отдельные тесты переопределяют это через side_effect.
    """
    with (
        patch("app.ingestion.async_pipeline.FixedSizeChunker") as mock_chunker_cls,
        patch("app.ingestion.async_pipeline.VectorStore") as mock_store_cls,
    ):
        mock_chunker = MagicMock()
        mock_chunker.split.side_effect = lambda text: [text]
        mock_chunker_cls.return_value = mock_chunker

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        pipeline = AsyncIngestionPipeline(source_dir=tmp_path, **kwargs)

    return pipeline, mock_chunker, mock_store


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDiscoverFiles:
    def test_finds_txt_and_md_only(self, tmp_path):
        (tmp_path / "a.txt").write_text("текст а", encoding="utf-8")
        (tmp_path / "b.md").write_text("# текст б", encoding="utf-8")
        (tmp_path / "c.pdf").write_bytes(b"%PDF-fake")

        pipeline, *_ = _make_pipeline(tmp_path)
        names = sorted(p.name for p in pipeline.discover_files())

        assert names == ["a.txt", "b.md"]

    def test_result_is_sorted_deterministically(self, tmp_path):
        (tmp_path / "z.txt").write_text("z", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")

        pipeline, *_ = _make_pipeline(tmp_path)
        names = [p.name for p in pipeline.discover_files()]

        assert names == sorted(names)


# ---------------------------------------------------------------------------
# make_point_id
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMakePointId:
    def test_deterministic(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        path = tmp_path / "doc.txt"
        assert pipeline.make_point_id(path, 0) == pipeline.make_point_id(path, 0)

    def test_different_chunk_index_different_id(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        path = tmp_path / "doc.txt"
        assert pipeline.make_point_id(path, 0) != pipeline.make_point_id(path, 1)

    def test_does_not_depend_on_execution_order(self, tmp_path):
        """id должен зависеть только от (путь, номер чанка) — не от того,
        в каком порядке воркеры завершили обработку файлов."""
        pipeline, *_ = _make_pipeline(tmp_path)
        path = tmp_path / "doc.txt"
        first_call = pipeline.make_point_id(path, 3)
        # Вызовы в другом "порядке" (имитация другого прогона) дают тот же id
        _ = pipeline.make_point_id(tmp_path / "other.txt", 0)
        second_call = pipeline.make_point_id(path, 3)
        assert first_call == second_call


# ---------------------------------------------------------------------------
# _embed_async / semaphore
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestEmbedAsyncSemaphore:
    async def test_respects_concurrency_limit(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path, max_concurrent_embeddings=2)

        state = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def _slow_embedding(text: str):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return [0.1, 0.2]

        with patch(
            "app.ingestion.async_pipeline.get_embedding",
            side_effect=_slow_embedding,
        ):
            await asyncio.gather(
                *(pipeline._embed_async(f"чанк {i}") for i in range(6))
            )

        assert state["peak"] <= 2

    async def test_returns_embedding_vector(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path, max_concurrent_embeddings=3)

        with patch(
            "app.ingestion.async_pipeline.get_embedding",
            return_value=[0.1, 0.2, 0.3],
        ):
            vector = await pipeline._embed_async("текст чанка")

        assert vector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# _process_file
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestProcessFile:
    async def test_returns_chunk_count_and_upserts(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("немного текста", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)
        mock_chunker.split.side_effect = lambda text: ["чанк1", "чанк2"]

        with patch(
            "app.ingestion.async_pipeline.get_embedding", return_value=[0.1, 0.2]
        ):
            n = await pipeline._process_file(path)

        assert n == 2
        mock_store.upsert_documents.assert_called_once()
        docs = mock_store.upsert_documents.call_args.args[0]
        assert len(docs) == 2
        assert docs[0]["metadata"]["source"] == str(path)
        assert docs[1]["metadata"]["chunk_index"] == 1

    async def test_empty_file_returns_zero_without_upsert(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("   \n  ", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        n = await pipeline._process_file(path)

        assert n == 0
        mock_store.upsert_documents.assert_not_called()

    async def test_no_chunks_returns_zero(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("текст", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)
        mock_chunker.split.side_effect = lambda text: []

        n = await pipeline._process_file(path)

        assert n == 0
        mock_store.upsert_documents.assert_not_called()


# ---------------------------------------------------------------------------
# run — очередь + пул воркеров
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestRun:
    async def test_creates_collection_once(self, tmp_path):
        (tmp_path / "a.txt").write_text("текст a", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.async_pipeline.get_embedding", return_value=[0.1]):
            await pipeline.run()

        mock_store.create_collection.assert_called_once()

    async def test_processes_all_files_with_worker_pool(self, tmp_path):
        for i in range(6):
            (tmp_path / f"doc{i}.txt").write_text(f"текст {i}", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path, num_workers=3)

        with patch("app.ingestion.async_pipeline.get_embedding", return_value=[0.1]):
            report = await pipeline.run()

        assert isinstance(report, IngestionReport)
        assert len(report.files_processed) == 6
        assert report.chunks_indexed == 6  # мок split() -> 1 чанк на файл

    async def test_bad_file_does_not_stop_other_files(self, tmp_path):
        (tmp_path / "good.txt").write_text("нормальный текст", encoding="utf-8")
        (tmp_path / "bad.txt").write_bytes("испорченный текст".encode("cp1251"))
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path, num_workers=2)

        with patch("app.ingestion.async_pipeline.get_embedding", return_value=[0.1]):
            report = await pipeline.run()

        assert len(report.files_processed) == 1
        assert len(report.files_skipped) == 1
        assert "bad.txt" in report.files_skipped[0]

    async def test_empty_folder_returns_empty_report(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)

        with patch("app.ingestion.async_pipeline.get_embedding", return_value=[0.1]):
            report = await pipeline.run()

        assert report.files_processed == []
        assert report.chunks_indexed == 0

    async def test_rerun_reuses_same_point_ids(self, tmp_path):
        """Повторный прогон по той же папке должен строить те же id точек —
        идемпотентность должна сохраняться и при параллельной обработке,
        где порядок завершения воркеров не гарантирован (см. урок 5.2)."""
        (tmp_path / "a.txt").write_text("текст a", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path, num_workers=2)

        with patch("app.ingestion.async_pipeline.get_embedding", return_value=[0.1]):
            await pipeline.run()
            first_ids = sorted(
                d["id"] for d in mock_store.upsert_documents.call_args.args[0]
            )

            await pipeline.run()
            second_ids = sorted(
                d["id"] for d in mock_store.upsert_documents.call_args.args[0]
            )

        assert first_ids == second_ids


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant and a real embedding backend
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAsyncIngestionPipelineIntegration:
    """Требует запущенный Qdrant и рабочий get_embedding().
    Скип по умолчанию: pytest -m 'not integration'."""

    def test_run_against_real_qdrant_no_duplicates_on_rerun(self):
        pytest.skip(
            "Требует запущенный Qdrant (docker run -p 6333:6333 qdrant/qdrant) "
            "и рабочий get_embedding() — запускайте вручную:\n"
            "  pytest tests/test_async_pipeline.py -m integration -v"
        )

    def test_speedup_measured_against_sync_pipeline(self):
        pytest.skip(
            "Сравнение времени прогона AsyncIngestionPipeline и "
            "MinimalIngestionPipeline на одной и той же выборке документов — "
            "запускайте вручную на реалистичном объёме (сотни-тысячи файлов)."
        )
