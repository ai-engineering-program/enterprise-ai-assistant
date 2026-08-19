"""Tests for app/ingestion/pipeline.py (Lesson 1.4).

Unit tests run without Qdrant or a real embedding model — FixedSizeChunker,
VectorStore and get_embedding are patched. Integration tests require a
running Qdrant instance.

Run unit tests only:
    pytest tests/test_ingestion_pipeline.py -v -m unit

Run all tests (requires Qdrant):
    pytest tests/test_ingestion_pipeline.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pipeline import (
    IngestionReport,
    MinimalIngestionPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path: Path, **kwargs):
    """Собрать пайплайн с подменёнными FixedSizeChunker и VectorStore.

    По умолчанию chunker.split() возвращает весь текст как один чанк —
    отдельные тесты переопределяют это поведение через side_effect.
    """
    with (
        patch("app.ingestion.pipeline.FixedSizeChunker") as mock_chunker_cls,
        patch("app.ingestion.pipeline.VectorStore") as mock_store_cls,
    ):
        mock_chunker = MagicMock()
        mock_chunker.split.side_effect = lambda text: [text]
        mock_chunker_cls.return_value = mock_chunker

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        pipeline = MinimalIngestionPipeline(source_dir=tmp_path, **kwargs)

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
        (tmp_path / "d.jpg").write_bytes(b"\xff\xd8\xff")

        pipeline, *_ = _make_pipeline(tmp_path)
        found = pipeline.discover_files()

        names = sorted(p.name for p in found)
        assert names == ["a.txt", "b.md"]

    def test_recurses_into_subfolders(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("вложенный текст", encoding="utf-8")

        pipeline, *_ = _make_pipeline(tmp_path)
        found = pipeline.discover_files()

        assert any(p.name == "nested.txt" for p in found)

    def test_empty_folder_returns_empty_list(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        assert pipeline.discover_files() == []

    def test_result_is_sorted_deterministically(self, tmp_path):
        (tmp_path / "z.txt").write_text("z", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")

        pipeline, *_ = _make_pipeline(tmp_path)
        names = [p.name for p in pipeline.discover_files()]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadText:
    def test_reads_utf8_file(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("Привет, мир!", encoding="utf-8")

        pipeline, *_ = _make_pipeline(tmp_path)
        assert pipeline.load_text(path) == "Привет, мир!"

    def test_raises_on_invalid_encoding(self, tmp_path):
        path = tmp_path / "broken.txt"
        # Кириллица в cp1251 не является корректной UTF-8 последовательностью
        path.write_bytes("привет мир".encode("cp1251"))

        pipeline, *_ = _make_pipeline(tmp_path)
        with pytest.raises(UnicodeDecodeError):
            pipeline.load_text(path)


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

    def test_different_files_different_id(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        p1 = tmp_path / "doc1.txt"
        p2 = tmp_path / "doc2.txt"
        assert pipeline.make_point_id(p1, 0) != pipeline.make_point_id(p2, 0)


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessFile:
    def test_returns_chunk_count_and_calls_upsert(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("немного текста", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)
        mock_chunker.split.side_effect = lambda text: ["чанк1", "чанк2"]

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1, 0.2]):
            n = pipeline.process_file(path)

        assert n == 2
        mock_store.upsert_documents.assert_called_once()
        docs = mock_store.upsert_documents.call_args.args[0]
        assert len(docs) == 2
        assert docs[0]["metadata"]["source"] == str(path)
        assert docs[0]["metadata"]["chunk_index"] == 0
        assert docs[1]["metadata"]["chunk_index"] == 1

    def test_empty_file_returns_zero_without_upsert(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("   \n  ", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        n = pipeline.process_file(path)

        assert n == 0
        mock_store.upsert_documents.assert_not_called()

    def test_no_chunks_returns_zero(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("текст", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)
        mock_chunker.split.side_effect = lambda text: []

        n = pipeline.process_file(path)

        assert n == 0
        mock_store.upsert_documents.assert_not_called()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRun:
    def test_creates_collection_once(self, tmp_path):
        (tmp_path / "a.txt").write_text("текст a", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1]):
            pipeline.run()

        mock_store.create_collection.assert_called_once()

    def test_processes_all_supported_files_only(self, tmp_path):
        (tmp_path / "a.txt").write_text("текст a", encoding="utf-8")
        (tmp_path / "b.md").write_text("# текст b", encoding="utf-8")
        (tmp_path / "c.pdf").write_bytes(b"%PDF-fake")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1]):
            report = pipeline.run()

        assert isinstance(report, IngestionReport)
        assert len(report.files_processed) == 2
        assert report.chunks_indexed == 2  # мок split() -> 1 чанк на файл

    def test_skips_file_with_bad_encoding_without_failing_run(self, tmp_path):
        (tmp_path / "good.txt").write_text("нормальный текст", encoding="utf-8")
        (tmp_path / "bad.txt").write_bytes("испорченный текст".encode("cp1251"))
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1]):
            report = pipeline.run()

        assert len(report.files_processed) == 1
        assert len(report.files_skipped) == 1
        assert "bad.txt" in report.files_skipped[0]

    def test_rerun_on_same_folder_reuses_same_point_ids(self, tmp_path):
        """Повторный прогон по той же папке должен строить те же id точек —
        именно это делает upsert идемпотентным и защищает от дублей
        в Qdrant (сам Qdrant в этом тесте замокан)."""
        (tmp_path / "a.txt").write_text("текст a", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1]):
            pipeline.run()
            first_ids = [d["id"] for d in mock_store.upsert_documents.call_args.args[0]]

            pipeline.run()
            second_ids = [d["id"] for d in mock_store.upsert_documents.call_args.args[0]]

        assert first_ids == second_ids

    def test_same_text_under_different_filename_is_indexed_again(self, tmp_path):
        """Документирует известное ограничение v1: дедупликация построена
        на пути к файлу, а не на содержимом — copy файла под другим именем
        индексируется как отдельный документ (раздел 4 закрывает этот разрыв)."""
        (tmp_path / "policy_v1.txt").write_text("одинаковый текст", encoding="utf-8")
        (tmp_path / "policy_v2.txt").write_text("одинаковый текст", encoding="utf-8")
        pipeline, mock_chunker, mock_store = _make_pipeline(tmp_path)

        with patch("app.ingestion.pipeline.get_embedding", return_value=[0.1]):
            report = pipeline.run()

        assert len(report.files_processed) == 2
        assert report.chunks_indexed == 2


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMinimalIngestionPipelineIntegration:
    """Требует запущенный Qdrant. Скип по умолчанию: pytest -m 'not integration'."""

    def test_run_against_real_qdrant_no_duplicates_on_rerun(self):
        pytest.skip(
            "Требует запущенный Qdrant (docker run -p 6333:6333 qdrant/qdrant) — "
            "запускайте вручную:\n"
            "  pytest tests/test_ingestion_pipeline.py -m integration -v"
        )
