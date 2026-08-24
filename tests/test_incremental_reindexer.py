"""
Tests for incremental_reindexer: CheckpointStore, IncrementalReindexer.run()
(resumable batch loop), validate_before_cutover() (doc-level recall) and
switch_alias() (atomic Qdrant alias cutover).

Run unit tests (no services required):
    pytest tests/test_incremental_reindexer.py -v -m unit

Run integration tests (requires Qdrant on localhost:6333):
    docker run -d -p 6333:6333 qdrant/qdrant
    pytest tests/test_incremental_reindexer.py -v -m integration
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ingestion.incremental_reindexer import (
    CheckpointStore,
    IncrementalReindexer,
    ReindexCheckpoint,
)


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

def make_documents(n: int) -> list[dict]:
    return [{"doc_id": f"doc-{i}", "text": f"content of document {i}"} for i in range(n)]


def fake_reprocess_fn(document: dict) -> list[dict]:
    """Имитирует parse -> normalize -> chunk -> embed: одна запись на вход."""
    return [{"doc_id": document["doc_id"], "chunk_index": 0, "vector": [0.1, 0.2]}]


@pytest.fixture()
def store() -> CheckpointStore:
    return CheckpointStore()


@pytest.fixture()
def write_spy():
    calls: list[tuple[str, list[dict]]] = []

    def _write(collection_name: str, chunk_records: list[dict]) -> None:
        calls.append((collection_name, list(chunk_records)))

    return _write, calls


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointStore:
    def test_load_missing_returns_none(self, store: CheckpointStore) -> None:
        assert store.load("docs_v2") is None

    def test_save_then_load_returns_same_checkpoint(self, store: CheckpointStore) -> None:
        checkpoint = ReindexCheckpoint(
            green_collection="docs_v2",
            last_completed_index=41,
            chunks_written=42,
            documents_migrated=42,
        )
        store.save(checkpoint)
        loaded = store.load("docs_v2")
        assert loaded is not None
        assert loaded.last_completed_index == 41
        assert loaded.chunks_written == 42

    def test_save_overwrites_previous_checkpoint(self, store: CheckpointStore) -> None:
        store.save(ReindexCheckpoint(green_collection="docs_v2", last_completed_index=10))
        store.save(ReindexCheckpoint(green_collection="docs_v2", last_completed_index=20))
        assert store.load("docs_v2").last_completed_index == 20

    def test_checkpoints_are_isolated_per_collection(self, store: CheckpointStore) -> None:
        store.save(ReindexCheckpoint(green_collection="docs_v2", last_completed_index=5))
        store.save(ReindexCheckpoint(green_collection="docs_v3", last_completed_index=99))
        assert store.load("docs_v2").last_completed_index == 5
        assert store.load("docs_v3").last_completed_index == 99


# ---------------------------------------------------------------------------
# IncrementalReindexer.run()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIncrementalReindexerRun:
    def test_migrates_all_documents_in_batches(self, store, write_spy) -> None:
        write_fn, calls = write_spy
        reindexer = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=10,
            show_progress=False,
        )
        documents = make_documents(25)
        progress = reindexer.run(documents, green_collection="docs_v2", resume=False)

        assert progress.documents_total == 25
        assert progress.documents_migrated == 25
        assert progress.chunks_written == 25
        assert progress.stopped_early is False
        # 25 документов, батч по 10 -> два полных батча + один неполный (5)
        assert len(calls) == 3
        assert [len(records) for _, records in calls] == [10, 10, 5]

    def test_checkpoint_saved_only_after_batch_write(self, store, write_spy) -> None:
        write_fn, calls = write_spy
        reindexer = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=5,
            show_progress=False,
        )
        reindexer.run(make_documents(12), green_collection="docs_v2", resume=False)

        checkpoint = store.load("docs_v2")
        assert checkpoint is not None
        # last_completed_index должен указывать на последний перенесённый документ (11)
        assert checkpoint.last_completed_index == 11
        assert checkpoint.chunks_written == 12
        assert checkpoint.documents_migrated == 12

    def test_resume_continues_from_last_checkpoint(self, store) -> None:
        seen_doc_ids: list[str] = []

        def write_fn(collection_name: str, chunk_records: list[dict]) -> None:
            seen_doc_ids.extend(r["doc_id"] for r in chunk_records)

        documents = make_documents(20)

        # Первый прогон: имитируем сбой/остановку после первых 8 документов --
        # запрашиваем стоп до старта, чтобы run() ушёл сразу после первого
        # батча (batch_size=5 -> первый батч из документов 0-4, затем стоп
        # проверяется перед документом 5).
        first_run = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=5,
            show_progress=False,
        )

        # Обрываем прогон вручную после обработки первых 8 документов,
        # оборачивая reprocess_fn счётчиком.
        counter = {"n": 0}

        def counting_reprocess(document: dict) -> list[dict]:
            counter["n"] += 1
            if counter["n"] > 8:
                first_run.request_stop()
            return fake_reprocess_fn(document)

        first_run._reprocess_fn = counting_reprocess  # type: ignore[attr-defined]
        progress_1 = first_run.run(documents, green_collection="docs_v2", resume=False)

        assert progress_1.stopped_early is True
        migrated_after_first_run = progress_1.documents_migrated
        assert 0 < migrated_after_first_run < 20

        # Второй прогон с тем же store и resume=True должен продолжить
        # ровно с первого ещё не перенесённого документа.
        second_run = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=5,
            show_progress=False,
        )
        progress_2 = second_run.run(documents, green_collection="docs_v2", resume=True)

        assert progress_2.stopped_early is False
        # Суммарно должны быть перенесены все 20 документов ровно по одному разу
        assert len(seen_doc_ids) == len(set(seen_doc_ids)) == 20
        assert set(seen_doc_ids) == {d["doc_id"] for d in documents}

    def test_stop_requested_before_run_migrates_nothing(self, store, write_spy) -> None:
        write_fn, calls = write_spy
        reindexer = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=5,
            show_progress=False,
        )
        reindexer.request_stop()
        progress = reindexer.run(make_documents(10), green_collection="docs_v2", resume=False)

        assert progress.stopped_early is True
        assert progress.documents_migrated == 0
        assert calls == []

    def test_empty_document_list_returns_zero_progress(self, store, write_spy) -> None:
        write_fn, calls = write_spy
        reindexer = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=write_fn,
            checkpoint_store=store,
            batch_size=5,
            show_progress=False,
        )
        progress = reindexer.run([], green_collection="docs_v2", resume=False)

        assert progress.documents_total == 0
        assert progress.documents_migrated == 0
        assert progress.chunks_written == 0
        assert calls == []


# ---------------------------------------------------------------------------
# validate_before_cutover()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateBeforeCutover:
    def _make_reindexer(self) -> IncrementalReindexer:
        return IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=lambda *_: None,
            show_progress=False,
        )

    def test_perfect_overlap_passes(self) -> None:
        reindexer = self._make_reindexer()

        def blue_search(query: str, top_k: int) -> list[str]:
            return ["doc-1", "doc-2", "doc-3"]

        def green_search(query: str, top_k: int) -> list[str]:
            return ["doc-1", "doc-2", "doc-3"]

        result = reindexer.validate_before_cutover(
            blue_search, green_search, test_queries=["q1", "q2"], threshold=0.85,
        )
        assert result["average_doc_overlap"] == pytest.approx(1.0)
        assert result["passed"] is True
        assert result["queries_below_threshold"] == []

    def test_partial_overlap_below_threshold_fails(self) -> None:
        reindexer = self._make_reindexer()

        def blue_search(query: str, top_k: int) -> list[str]:
            return ["doc-1", "doc-2", "doc-3", "doc-4"]

        def green_search(query: str, top_k: int) -> list[str]:
            # Только половина документов blue найдена в green
            return ["doc-1", "doc-2", "doc-99", "doc-100"]

        result = reindexer.validate_before_cutover(
            blue_search, green_search, test_queries=["q1"], threshold=0.85,
        )
        assert result["average_doc_overlap"] == pytest.approx(0.5)
        assert result["passed"] is False
        assert "q1" in result["queries_below_threshold"]

    def test_chunk_id_style_mismatch_does_not_break_doc_level_comparison(self) -> None:
        """
        Регрессия против ошибки Данилы из текста урока: даже если "внутренние"
        идентификаторы чанков в blue и green совершенно не пересекаются
        (другая chunking-стратегия), doc-level сравнение всё равно должно
        корректно распознать высокое перекрытие, если это одни и те же
        документы.
        """
        reindexer = self._make_reindexer()

        def blue_search(query: str, top_k: int) -> list[str]:
            return ["doc-A", "doc-B"]

        def green_search(query: str, top_k: int) -> list[str]:
            return ["doc-A", "doc-B"]

        result = reindexer.validate_before_cutover(
            blue_search, green_search, test_queries=["q"], threshold=0.9,
        )
        assert result["passed"] is True

    def test_query_with_empty_blue_baseline_is_skipped(self) -> None:
        reindexer = self._make_reindexer()

        def blue_search(query: str, top_k: int) -> list[str]:
            return []

        def green_search(query: str, top_k: int) -> list[str]:
            return ["doc-1"]

        result = reindexer.validate_before_cutover(
            blue_search, green_search, test_queries=["q_empty"], threshold=0.85,
        )
        # Ни одного валидного запроса для усреднения -> дефолт 0.0, не деление на ноль
        assert result["average_doc_overlap"] == 0.0
        assert result["queries_below_threshold"] == []


# ---------------------------------------------------------------------------
# switch_alias()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSwitchAlias:
    def test_calls_update_collection_aliases_once(self) -> None:
        reindexer = IncrementalReindexer(
            reprocess_fn=fake_reprocess_fn,
            write_batch_fn=lambda *_: None,
            show_progress=False,
        )
        mock_client = MagicMock()
        reindexer.switch_alias(mock_client, alias_name="docs_live", new_collection_name="docs_v2")

        mock_client.update_collection_aliases.assert_called_once()
        _, kwargs = mock_client.update_collection_aliases.call_args
        operations = kwargs.get("change_aliases_operations") or (
            mock_client.update_collection_aliases.call_args[0][0]
            if mock_client.update_collection_aliases.call_args[0]
            else []
        )
        assert len(operations) == 2


# ---------------------------------------------------------------------------
# Интеграционные тесты
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIncrementalReindexerIntegration:
    """
    Требуют запущенного Qdrant на localhost:6333 и реального набора
    исходных документов. Запускайте вручную:
        docker run -d -p 6333:6333 qdrant/qdrant
        pytest tests/test_incremental_reindexer.py -v -m integration
    """

    def test_full_incremental_migration_with_real_qdrant(self) -> None:
        pytest.skip("Требуется запущенный Qdrant -- запустите вручную")

    def test_checkpoint_survives_process_restart_with_durable_store(self) -> None:
        pytest.skip(
            "Требует durable CheckpointStore (Redis/БД), переживающего "
            "перезапуск процесса -- в этом модуле store хранится в памяти"
        )
