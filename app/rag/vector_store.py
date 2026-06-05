"""VectorStore — abstraction layer over Qdrant for the RAG pipeline.

Lesson 2.3: Qdrant architecture and integration.
Students implement this class as part of the hands-on exercise.
"""

from __future__ import annotations

from typing import Any

__all__ = ["VectorStore"]


class VectorStore:
    """Manages a single Qdrant collection: creation, upsert, and search."""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_size: int,
    ) -> None:
        # TODO: импортировать QdrantClient из qdrant_client
        # TODO: инициализировать self.client = QdrantClient(host=host, port=port)
        # TODO: сохранить self.collection_name = collection_name
        # TODO: сохранить self.vector_size = vector_size
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """Create the Qdrant collection if it does not yet exist.

        Use Distance.COSINE and HnswConfigDiff(m=16, ef_construct=100).
        If the collection already exists, do nothing (idempotent).
        """
        # TODO: проверить self.client.collection_exists(self.collection_name)
        # TODO: если не существует — вызвать self.client.create_collection(...)
        # TODO: передать vectors_config=VectorParams(size=..., distance=Distance.COSINE)
        # TODO: передать hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert_documents(self, docs: list[dict[str, Any]]) -> None:
        """Insert or update a batch of documents.

        Each doc must contain:
            - "id":       int or str  — unique identifier
            - "vector":   list[float] — embedding vector (length == vector_size)
            - "metadata": dict        — arbitrary JSON-serialisable fields

        Use PointStruct(id=..., vector=..., payload=doc["metadata"]).
        """
        # TODO: импортировать PointStruct из qdrant_client.models
        # TODO: преобразовать каждый doc в PointStruct
        # TODO: вызвать self.client.upsert(collection_name=..., points=[...])
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the most similar vectors without any metadata filtering.

        Returns a list of dicts:
            [{"id": ..., "score": float, "metadata": dict}, ...]
        """
        # TODO: вызвать self.client.search(collection_name=..., query_vector=..., limit=...)
        # TODO: преобразовать каждый ScoredPoint в {"id": r.id, "score": r.score, "metadata": r.payload}
        raise NotImplementedError

    def search_with_filter(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the most similar vectors among points matching all filter conditions.

        ``filters`` is a flat dict of exact-match conditions, e.g.::

            {"department": "HR", "year": 2024}

        Build a ``Filter(must=[FieldCondition(...), ...])`` from this dict
        and pass it as ``query_filter`` to ``client.search``.

        Returns the same shape as :meth:`search`.
        """
        # TODO: импортировать Filter, FieldCondition, MatchValue из qdrant_client.models
        # TODO: построить список условий: [FieldCondition(key=k, match=MatchValue(value=v)) for k,v in filters.items()]
        # TODO: создать query_filter = Filter(must=[...])
        # TODO: вызвать self.client.search(..., query_filter=query_filter, ...)
        # TODO: вернуть результаты в том же формате, что и search()
        raise NotImplementedError
