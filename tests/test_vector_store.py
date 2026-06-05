"""Tests for app/rag/vector_store.py — Lesson 2.3.

Unit tests run without any external services.
Integration tests require Qdrant running on localhost:6333
(start with: docker-compose up -d qdrant).

Run unit tests only:
    pytest tests/test_vector_store.py -v -m unit

Run integration tests (requires Qdrant):
    pytest tests/test_vector_store.py -v -m integration
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_SIZE = 8  # small dimension for fast tests


def _make_vector(seed: float) -> list[float]:
    """Return a normalised vector of VECTOR_SIZE filled with *seed*."""
    import math
    raw = [seed + i * 0.01 for i in range(VECTOR_SIZE)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _sample_docs() -> list[dict]:
    return [
        {
            "id": 1,
            "vector": _make_vector(0.10),
            "metadata": {"department": "HR", "doc_type": "policy", "year": 2023},
        },
        {
            "id": 2,
            "vector": _make_vector(0.20),
            "metadata": {"department": "IT", "doc_type": "instruction", "year": 2024},
        },
        {
            "id": 3,
            "vector": _make_vector(0.11),
            "metadata": {"department": "HR", "doc_type": "report", "year": 2024},
        },
    ]


# ---------------------------------------------------------------------------
# Unit tests — no Qdrant required
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorStoreInterface:
    """Verify that VectorStore can be imported and has the expected interface."""

    def test_class_importable(self):
        from app.rag.vector_store import VectorStore  # noqa: F401

    def test_has_required_methods(self):
        from app.rag.vector_store import VectorStore

        for method in ("create_collection", "upsert_documents", "search", "search_with_filter"):
            assert hasattr(VectorStore, method), (
                f"VectorStore must have a '{method}' method"
            )

    def test_init_signature(self):
        """Constructor must accept host, port, collection_name, vector_size."""
        import inspect
        from app.rag.vector_store import VectorStore

        sig = inspect.signature(VectorStore.__init__)
        params = list(sig.parameters.keys())
        for expected in ("host", "port", "collection_name", "vector_size"):
            assert expected in params, (
                f"VectorStore.__init__ must have a '{expected}' parameter"
            )

    def test_upsert_accepts_list_of_dicts(self):
        """upsert_documents signature must accept a list parameter."""
        import inspect
        from app.rag.vector_store import VectorStore

        sig = inspect.signature(VectorStore.upsert_documents)
        params = list(sig.parameters.keys())
        assert "docs" in params, "upsert_documents must have a 'docs' parameter"

    def test_search_with_filter_accepts_filters_dict(self):
        import inspect
        from app.rag.vector_store import VectorStore

        sig = inspect.signature(VectorStore.search_with_filter)
        params = list(sig.parameters.keys())
        assert "filters" in params, "search_with_filter must have a 'filters' parameter"


# ---------------------------------------------------------------------------
# Integration tests — require Qdrant on localhost:6333
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVectorStoreIntegration:
    """End-to-end tests against a real Qdrant instance."""

    COLLECTION = "test_lesson_2_3"

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a fresh collection before each test and clean up after."""
        from app.rag.vector_store import VectorStore

        self.store = VectorStore(
            host="localhost",
            port=6333,
            collection_name=self.COLLECTION,
            vector_size=VECTOR_SIZE,
        )
        # Ensure clean state
        try:
            self.store.client.delete_collection(self.COLLECTION)
        except Exception:
            pass

        self.store.create_collection()
        yield
        # Cleanup
        try:
            self.store.client.delete_collection(self.COLLECTION)
        except Exception:
            pass

    def test_create_collection_idempotent(self):
        """Calling create_collection twice must not raise."""
        self.store.create_collection()  # second call — must be a no-op

    def test_upsert_and_search_returns_results(self):
        """After upserting docs, search must return at least one result."""
        docs = _sample_docs()
        self.store.upsert_documents(docs)

        query = _make_vector(0.10)
        results = self.store.search(query, limit=3)

        assert len(results) > 0, "search must return at least one result"
        assert "id" in results[0], "each result must have an 'id' key"
        assert "score" in results[0], "each result must have a 'score' key"
        assert "metadata" in results[0], "each result must have a 'metadata' key"

    def test_search_returns_closest_document(self):
        """The closest vector to id=1 should be ranked first."""
        docs = _sample_docs()
        self.store.upsert_documents(docs)

        # Query vector is identical to doc id=1
        query = _make_vector(0.10)
        results = self.store.search(query, limit=3)

        top_id = results[0]["id"]
        assert top_id == 1, (
            f"Expected id=1 as the closest result, got id={top_id}"
        )

    def test_search_with_filter_returns_only_matching_docs(self):
        """search_with_filter must return only documents where department==HR."""
        docs = _sample_docs()
        self.store.upsert_documents(docs)

        query = _make_vector(0.10)
        results = self.store.search_with_filter(
            query_vector=query,
            filters={"department": "HR"},
            limit=5,
        )

        assert len(results) > 0, "Expected at least one HR document"
        for r in results:
            assert r["metadata"].get("department") == "HR", (
                f"Expected department=HR, got metadata={r['metadata']}"
            )

    def test_search_with_filter_excludes_other_departments(self):
        """Documents from IT must not appear when filtering for HR."""
        docs = _sample_docs()
        self.store.upsert_documents(docs)

        query = _make_vector(0.20)  # close to IT doc
        results = self.store.search_with_filter(
            query_vector=query,
            filters={"department": "HR"},
            limit=5,
        )

        ids = [r["id"] for r in results]
        assert 2 not in ids, (
            "IT document (id=2) must not appear in HR-filtered results"
        )

    def test_upsert_overwrites_existing_point(self):
        """Upserting a doc with an existing id must update its metadata."""
        docs = _sample_docs()
        self.store.upsert_documents(docs)

        updated = [
            {
                "id": 1,
                "vector": _make_vector(0.10),
                "metadata": {"department": "Legal", "doc_type": "contract"},
            }
        ]
        self.store.upsert_documents(updated)

        results = self.store.search(_make_vector(0.10), limit=1)
        assert results[0]["id"] == 1
        assert results[0]["metadata"].get("department") == "Legal", (
            "upsert must overwrite existing metadata for id=1"
        )
