import pytest

from app.ingestion.checkpoint_tracker import IdempotentIngestionPipeline


@pytest.mark.unit
class TestMakeKey:
    def test_deterministic(self):
        pipeline = IdempotentIngestionPipeline()
        key1 = pipeline.make_key("doc_42", 3)
        key2 = pipeline.make_key("doc_42", 3)
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self):
        pipeline = IdempotentIngestionPipeline()
        assert pipeline.make_key("doc_1", 0) != pipeline.make_key("doc_2", 0)
        assert pipeline.make_key("doc_1", 0) != pipeline.make_key("doc_1", 1)


@pytest.mark.unit
class TestUpsertIdempotency:
    def test_repeated_upsert_does_not_duplicate(self):
        pipeline = IdempotentIngestionPipeline()
        pipeline.upsert_chunk("doc_1", 0, "текст фрагмента")
        pipeline.upsert_chunk("doc_1", 0, "текст фрагмента")
        pipeline.upsert_chunk("doc_1", 0, "текст фрагмента")
        assert pipeline.index_size() == 1

    def test_different_chunks_increase_index(self):
        pipeline = IdempotentIngestionPipeline()
        pipeline.upsert_chunk("doc_1", 0, "a")
        pipeline.upsert_chunk("doc_1", 1, "b")
        pipeline.upsert_chunk("doc_2", 0, "c")
        assert pipeline.index_size() == 3

    def test_is_duplicate(self):
        pipeline = IdempotentIngestionPipeline()
        key = pipeline.make_key("doc_1", 0)
        assert pipeline.is_duplicate(key) is False
        pipeline.upsert_chunk("doc_1", 0, "a")
        assert pipeline.is_duplicate(key) is True


@pytest.mark.unit
class TestCheckpoint:
    def test_confirm_offset_advances(self):
        pipeline = IdempotentIngestionPipeline()
        pipeline.confirm_offset(0)
        pipeline.confirm_offset(1)
        pipeline.confirm_offset(2)
        assert pipeline.resume_offset() == 3

    def test_confirm_offset_never_moves_backward(self):
        pipeline = IdempotentIngestionPipeline()
        pipeline.confirm_offset(5)
        pipeline.confirm_offset(2)  # опоздавшее подтверждение из старой пачки
        assert pipeline.resume_offset() == 6

    def test_resume_offset_initial_value(self):
        pipeline = IdempotentIngestionPipeline()
        assert pipeline.resume_offset() == 0


@pytest.mark.unit
class TestRestartScenario:
    def test_restart_without_duplication(self):
        """
        Эмуляция инцидента из урока: сбой на 3-м чанке, перезапуск с
        перекрытием — чанки 2-4 обрабатываются повторно, но благодаря
        идемпотентному upsert индекс не должен вырасти сверх 5 записей.
        """
        pipeline = IdempotentIngestionPipeline()
        chunks = [("doc_1", i, f"фрагмент {i}") for i in range(5)]

        # Первый проход: успели обработать и подтвердить чанки 0, 1, 2
        for offset in range(3):
            doc_id, chunk_index, content = chunks[offset]
            pipeline.upsert_chunk(doc_id, chunk_index, content)
            pipeline.confirm_offset(offset)

        assert pipeline.index_size() == 3

        # Сбой. Перезапуск с перекрытием: worker не уверен, дошла ли запись
        # последнего чанка до подтверждения, и начинает на шаг раньше.
        restart_from = max(pipeline.resume_offset() - 1, 0)
        for offset in range(restart_from, len(chunks)):
            doc_id, chunk_index, content = chunks[offset]
            pipeline.upsert_chunk(doc_id, chunk_index, content)
            pipeline.confirm_offset(offset)

        # Чанк 2 обработан дважды, но индекс не содержит дублей
        assert pipeline.index_size() == 5
        assert pipeline.resume_offset() == 5


@pytest.mark.integration
class TestCheckpointPersistenceIntegration:
    """Требует реального хранилища чекпоинтов (Redis/Postgres) и Qdrant."""

    def test_checkpoint_survives_process_restart(self):
        pytest.skip("Требует запущенных Redis и Qdrant — запускать вручную")
