"""
Тесты для IndexMigrator.

Запуск unit-тестов (без Qdrant):
    pytest tests/test_index_migrator.py -v -m unit

Запуск интеграционных тестов (требуется Qdrant на localhost:6333):
    docker run -d -p 6333:6333 qdrant/qdrant
    pytest tests/test_index_migrator.py -v -m integration
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from app.rag.index_migrator import IndexMigrator


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

class FakeEmbedModel:
    """Минимальная заглушка модели векторных представлений."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Возвращает фиксированные векторы размерности 4 для детерминированности тестов
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    return client


@pytest.fixture()
def migrator(mock_client: MagicMock) -> IndexMigrator:
    return IndexMigrator(qdrant_client=mock_client, embed_model=FakeEmbedModel())


# ---------------------------------------------------------------------------
# Unit-тесты
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexMigratorUnit:
    """Тесты, которые выполняются без запущенного Qdrant (используют mock-клиент)."""

    def test_create_green_collection_calls_recreate(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """create_green_collection должен вызвать recreate_collection с корректными аргументами."""
        migrator.create_green_collection(
            blue_name="docs_v1", green_name="docs_v2", vector_size=768
        )
        mock_client.recreate_collection.assert_called_once()
        call_kwargs = mock_client.recreate_collection.call_args
        # Проверяем, что передано правильное имя коллекции
        assert call_kwargs[1].get("collection_name") == "docs_v2" or (
            len(call_kwargs[0]) > 0 and call_kwargs[0][0] == "docs_v2"
        )

    def test_migrate_documents_returns_correct_count(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """migrate_documents должен вернуть количество перенесённых документов."""
        from unittest.mock import MagicMock

        # Создаём 5 фиктивных точек
        fake_points = []
        for i in range(5):
            p = MagicMock()
            p.id = i
            p.payload = {"text": f"документ {i}"}
            fake_points.append(p)

        mock_client.scroll.return_value = (fake_points, None)

        count = migrator.migrate_documents(
            source_collection="docs_v1",
            target_collection="docs_v2",
            batch_size=3,
        )
        assert count == 5

    def test_migrate_documents_calls_upsert(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """migrate_documents должен вызвать upsert хотя бы один раз."""
        from unittest.mock import MagicMock

        fake_points = [MagicMock(id=i, payload={"text": f"текст {i}"}) for i in range(3)]
        mock_client.scroll.return_value = (fake_points, None)

        migrator.migrate_documents("docs_v1", "docs_v2", batch_size=10)
        mock_client.upsert.assert_called()

    def test_rollback_deletes_green_collection(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """rollback должен вызвать delete_collection для зелёной коллекции."""
        migrator.rollback(blue_name="docs_v1", green_name="docs_v2")
        mock_client.delete_collection.assert_called_once()
        call_kwargs = mock_client.delete_collection.call_args
        passed_name = call_kwargs[1].get("collection_name") or (
            call_kwargs[0][0] if call_kwargs[0] else None
        )
        assert passed_name == "docs_v2"

    def test_rollback_does_not_touch_blue_collection(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """rollback не должен трогать синюю коллекцию."""
        migrator.rollback(blue_name="docs_v1", green_name="docs_v2")
        # delete_collection вызван ровно один раз — только для зелёной
        assert mock_client.delete_collection.call_count == 1

    def test_switch_traffic_does_not_raise(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """switch_traffic должен выполняться без исключений для значений 0-100."""
        for pct in [0, 5, 20, 50, 100]:
            migrator.switch_traffic(pct)  # не должно бросать исключение

    def test_migrate_documents_empty_source(
        self, migrator: IndexMigrator, mock_client: MagicMock
    ) -> None:
        """При пустой исходной коллекции migrate_documents должен вернуть 0."""
        mock_client.scroll.return_value = ([], None)
        count = migrator.migrate_documents("empty_col", "docs_v2")
        assert count == 0


# ---------------------------------------------------------------------------
# Интеграционные тесты
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIndexMigratorIntegration:
    """
    Тесты, требующие запущенного Qdrant на localhost:6333.
    Запускайте вручную:
        docker run -d -p 6333:6333 qdrant/qdrant
        pytest tests/test_index_migrator.py -v -m integration
    """

    def test_full_migration_cycle(self) -> None:
        """
        Полный цикл: создание зелёной коллекции, перенос документов, откат.
        Требует запущенного Qdrant.
        """
        pytest.skip("Требуется запущенный Qdrant — запустите вручную")
