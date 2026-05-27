"""RAG retrieval layer — stub added in Course 1 (M2 L2), implemented in Course 2.

After completing the Stepik exercise, copy your RAGRetriever and cosine_similarity
implementations here and verify that tests/test_retriever.py passes locally.
"""
import math
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its relevance score."""
    text: str
    source: str
    score: float


def fake_embed(text: str) -> list[float]:
    """Pseudo-embedding based on character hashes.

    Used for testing only — replace with a real embedding model in Course 2.
    """
    vec = [0.0] * 8
    for i, ch in enumerate(text.lower()):
        vec[i % 8] += ord(ch) / 1000.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    TODO (M2 L2): implement this function.
    """
    raise NotImplementedError


class RAGRetriever:
    """Retrieval component of the RAG system.

    Finds the most relevant document chunks for a query using
    embedding-based cosine similarity search.

    TODO (M2 L2): implement retrieve() and build_context().
    """

    def __init__(self, index: list[dict]):
        """
        Args:
            index: list of documents, each with keys:
                   'text' (str), 'source' (str), 'embedding' (list[float])
        """
        self.index = index

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Find top_k most relevant chunks for the query.

        Args:
            query: user query string
            top_k: number of results to return

        Returns:
            List of RetrievedChunk sorted by score descending.
            Returns empty list if index is empty.
        """
        raise NotImplementedError

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks into a context string for the LLM prompt.

        Format per chunk:
            [Документ N] (source)
            text

        Chunks are separated by two newlines.

        Args:
            chunks: list of RetrievedChunk (typically from retrieve())

        Returns:
            Multi-line string with formatted documents.
        """
        raise NotImplementedError
