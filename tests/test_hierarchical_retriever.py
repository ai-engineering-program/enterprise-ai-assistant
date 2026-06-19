"""
Tests for app/rag/hierarchical_retriever.py

Run unit tests (no external services):
    pytest tests/test_hierarchical_retriever.py -v -m unit

Run integration tests (requires Qdrant on localhost:6333):
    docker-compose up -d qdrant
    pytest tests/test_hierarchical_retriever.py -v -m integration
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.rag.hierarchical_retriever import (
    HierarchyNode,
    HierarchicalRetriever,
    ParentDocumentRetriever,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_hit(payload: dict, score: float = 0.9) -> MagicMock:
    hit = MagicMock()
    hit.payload = payload
    hit.score = score
    return hit


def _make_mock_point(payload: dict) -> MagicMock:
    point = MagicMock()
    point.payload = payload
    return point


def _make_retriever(
    search_results_by_collection: dict[str, list] | None = None,
    retrieve_results: list | None = None,
) -> HierarchicalRetriever:
    """Build a HierarchicalRetriever with a mocked Qdrant client and model."""
    mock_client = MagicMock()
    search_results_by_collection = search_results_by_collection or {}

    def fake_search(collection_name, query_vector, limit, query_filter=None,
                    with_payload=True, **kwargs):
        return search_results_by_collection.get(collection_name, [])

    mock_client.search.side_effect = fake_search

    if retrieve_results is not None:
        mock_client.retrieve.return_value = retrieve_results

    with patch("app.rag.hierarchical_retriever.SentenceTransformer") as MockModel:
        MockModel.return_value.encode.return_value = [0.1] * 768
        retriever = HierarchicalRetriever(
            qdrant_client=mock_client,
            model_name="mock-model",
        )
    # Replace with already-constructed mock so encode calls work
    retriever.model = MagicMock()
    retriever.model.encode.return_value = [0.1] * 768
    retriever.client = mock_client
    return retriever


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyNode:
    def test_node_defaults(self):
        node = HierarchyNode(
            node_id="fz152_ch6",
            level=1,
            level_type="chapter",
            text="Глава 6. Ответственность за нарушение...",
            title="Глава 6",
            doc_id="fz152",
        )
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.path == ""

    def test_node_with_parent(self):
        node = HierarchyNode(
            node_id="fz152_ch6_art24",
            level=2,
            level_type="article",
            text="Статья 24. Ответственность операторов...",
            title="Статья 24",
            doc_id="fz152",
            parent_id="fz152_ch6",
            children_ids=["fz152_ch6_art24_p1"],
            path="ФЗ-152 / Глава 6 / Статья 24",
        )
        assert node.parent_id == "fz152_ch6"
        assert "fz152_ch6_art24_p1" in node.children_ids


@pytest.mark.unit
class TestHierarchicalRetrieverUnit:
    """Tests that run without any external services."""

    def test_search_top_level_returns_payloads(self):
        ch6_payload = {
            "node_id": "fz152_ch6",
            "title": "Глава 6",
            "level": 1,
            "parent_id": "fz152",
        }
        retriever = _make_retriever(
            search_results_by_collection={
                "docs_level_1": [_make_mock_hit(ch6_payload, score=0.91)]
            }
        )
        results = retriever.search_top_level("ответственность за нарушение ФЗ-152")
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["node_id"] == "fz152_ch6"

    def test_search_within_section_returns_payloads(self):
        art24_payload = {
            "node_id": "fz152_ch6_art24",
            "title": "Статья 24",
            "text": "Статья 24. Ответственность за нарушение...",
            "parent_id": "fz152_ch6",
            "level": 2,
        }
        retriever = _make_retriever(
            search_results_by_collection={
                "docs_level_2": [_make_mock_hit(art24_payload, score=0.93)]
            }
        )
        results = retriever.search_within_section(
            "ответственность операторов",
            parent_ids=["fz152_ch6"],
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["node_id"] == "fz152_ch6_art24"

    def test_search_within_section_empty_parent_ids(self):
        """Если parent_ids пуст — вернуть [] без обращения к Qdrant."""
        retriever = _make_retriever()
        results = retriever.search_within_section(
            "ответственность", parent_ids=[]
        )
        assert results == []
        retriever.client.search.assert_not_called()

    def test_build_context_deduplicates_section_headers(self):
        top_sections = [
            {
                "node_id": "fz152_ch6",
                "path": "ФЗ-152 / Глава 6: Ответственность",
                "title": "Глава 6",
            }
        ]
        detail_hits = [
            {
                "node_id": "fz152_ch6_art24",
                "text": "Статья 24. Ответственность операторов.",
                "parent_id": "fz152_ch6",
            },
            {
                "node_id": "fz152_ch6_art25",
                "text": "Статья 25. Переходные положения.",
                "parent_id": "fz152_ch6",
            },
        ]
        retriever = _make_retriever()
        context = retriever.build_context(top_sections, detail_hits)

        assert isinstance(context, str)
        assert len(context) > 0
        # Заголовок раздела должен встречаться ровно один раз
        header_occurrences = context.count("Глава 6")
        assert header_occurrences == 1, (
            f"Заголовок главы должен быть один раз, найдено: {header_occurrences}"
        )
        # Оба текста статей должны быть в контексте
        assert "Статья 24" in context
        assert "Статья 25" in context

    def test_build_context_returns_string(self):
        retriever = _make_retriever()
        context = retriever.build_context([], [])
        assert isinstance(context, str)

    def test_retrieve_returns_tuple(self):
        ch6_payload = {
            "node_id": "fz152_ch6",
            "title": "Глава 6",
            "level": 1,
            "path": "ФЗ-152 / Глава 6",
            "parent_id": "fz152",
        }
        art24_payload = {
            "node_id": "fz152_ch6_art24",
            "title": "Статья 24",
            "text": "Статья 24. Ответственность операторов.",
            "parent_id": "fz152_ch6",
            "level": 2,
        }
        retriever = _make_retriever(
            search_results_by_collection={
                "docs_level_1": [_make_mock_hit(ch6_payload)],
                "docs_level_2": [_make_mock_hit(art24_payload)],
            }
        )
        result = retriever.retrieve("ответственность за нарушение ФЗ-152")
        assert isinstance(result, tuple), "retrieve() должен возвращать tuple"
        assert len(result) == 2
        context, articles = result
        assert isinstance(context, str)
        assert isinstance(articles, list)

    def test_retrieve_context_contains_article_text(self):
        ch6_payload = {
            "node_id": "fz152_ch6",
            "title": "Глава 6",
            "level": 1,
            "path": "ФЗ-152 / Глава 6",
            "parent_id": "fz152",
        }
        art24_payload = {
            "node_id": "fz152_ch6_art24",
            "title": "Статья 24",
            "text": "Статья 24. Ответственность операторов персональных данных.",
            "parent_id": "fz152_ch6",
            "level": 2,
        }
        retriever = _make_retriever(
            search_results_by_collection={
                "docs_level_1": [_make_mock_hit(ch6_payload)],
                "docs_level_2": [_make_mock_hit(art24_payload)],
            }
        )
        context, _ = retriever.retrieve("ответственность ФЗ-152")
        assert "Статья 24" in context or "Ответственность" in context


@pytest.mark.unit
class TestParentDocumentRetrieverUnit:
    """Unit tests for ParentDocumentRetriever."""

    def _make_pdr(
        self,
        chunk_hits: list,
        parent_points: list,
    ) -> ParentDocumentRetriever:
        mock_client = MagicMock()
        mock_client.search.return_value = chunk_hits
        mock_client.retrieve.return_value = parent_points

        with patch("app.rag.hierarchical_retriever.SentenceTransformer") as MockModel:
            MockModel.return_value.encode.return_value = [0.1] * 768
            pdr = ParentDocumentRetriever(mock_client, "mock-model")
        pdr.model = MagicMock()
        pdr.model.encode.return_value = [0.1] * 768
        pdr.client = mock_client
        return pdr

    def test_returns_list(self):
        pdr = self._make_pdr(chunk_hits=[], parent_points=[])
        result = pdr.retrieve_with_parent_context("тест")
        assert isinstance(result, list)

    def test_deduplicates_parents(self):
        # Два чанка из одной статьи → должен вернуть статью один раз
        chunk_hits = [
            _make_mock_hit({"parent_id": "fz152_ch6_art24"}),
            _make_mock_hit({"parent_id": "fz152_ch6_art24"}),
        ]
        art24_point = _make_mock_point({
            "node_id": "fz152_ch6_art24",
            "text": "Статья 24. Полный текст...",
        })
        pdr = self._make_pdr(chunk_hits=chunk_hits, parent_points=[art24_point])
        result = pdr.retrieve_with_parent_context("тест")
        # Qdrant.retrieve() должен быть вызван ровно один раз с одним уникальным ID
        call_args = pdr.client.retrieve.call_args
        ids_arg = call_args[1].get("ids") or call_args[0][1]
        assert len(ids_arg) == 1, (
            "Дедупликация: retrieve() должен получить только один уникальный parent_id"
        )

    def test_empty_chunks_returns_empty(self):
        pdr = self._make_pdr(chunk_hits=[], parent_points=[])
        result = pdr.retrieve_with_parent_context("тест")
        assert result == []
        pdr.client.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests (require running Qdrant)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHierarchicalRetrieverIntegration:
    """
    Tests that require a running Qdrant instance on localhost:6333
    and pre-indexed test data.

    Start services:
        docker-compose up -d qdrant

    Run:
        pytest tests/test_hierarchical_retriever.py -v -m integration
    """

    def test_full_retrieval_pipeline(self):
        pytest.skip(
            "Требует запущенного Qdrant и проиндексированных тестовых данных. "
            "Запустите вручную: docker-compose up -d qdrant"
        )
