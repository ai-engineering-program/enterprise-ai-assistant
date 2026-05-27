"""API routes — async implementation, Course 1 M3 L2.

Refactored from sync def query() (M2 L1) to async def query().
Key changes:
  - def query() → async def query()
  - httpx.Client → httpx.AsyncClient
  - rag_service.retrieve() → asyncio.gather(primary, metadata) with timeout
  - LLM call wrapped in asyncio.wait_for()

Run the load test from M3 L1 again (concurrency=20, 50) to see the improvement.
"""
import asyncio
import time

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.rag.retriever import RAGRetriever

router = APIRouter()

# Async HTTP client — does NOT block the worker thread.
# Initialise in FastAPI lifespan; here it's a module-level stub.
_async_client: httpx.AsyncClient | None = None

# RAG retriever — stub from M2 L2; full async implementation in Course 2.
_retriever = RAGRetriever(index=[])

RETRIEVAL_TIMEOUT = 5.0   # seconds — vector DB should be fast
LLM_TIMEOUT = 30.0        # seconds — LLM can be slow


class QueryRequest(BaseModel):
    text: str


async def _retrieve_primary(query: str) -> list[str]:
    """Primary retrieval from vector DB. Full implementation in Course 2."""
    # TODO (Course 2): replace with await rag_service.aretrieve(query, top_k=5)
    return _retriever.retrieve(query, top_k=5) if _retriever.index else []


async def _retrieve_metadata(query: str) -> dict:
    """Parallel metadata fetch (category, tags). Full implementation in Course 2."""
    # TODO (Course 2): replace with real metadata service call
    return {}


def _build_context(primary_docs: list, metadata: dict) -> str:
    """Combine retrieval results into LLM context string."""
    if not primary_docs:
        return ""
    chunks = [f"[Doc {i+1}] {chunk.text}" for i, chunk in enumerate(primary_docs)]
    return "\n\n".join(chunks)


@router.post("/query")
async def query(request: QueryRequest) -> JSONResponse:
    """POST /query — async (non-blocking).

    TODO (M3 L2): fill in the implementation steps below.
      1. Run _retrieve_primary and _retrieve_metadata concurrently via asyncio.gather()
         wrapped in asyncio.wait_for(..., timeout=RETRIEVAL_TIMEOUT).
      2. Build context from results.
      3. Call LLM via _async_client wrapped in asyncio.wait_for(..., timeout=LLM_TIMEOUT).
      4. Return JSONResponse with X-Response-Time-Ms header.
    """
    start = time.perf_counter()

    try:
        # Step 1: parallel retrieval — both sources run concurrently
        primary_docs, metadata = await asyncio.wait_for(
            asyncio.gather(
                _retrieve_primary(request.text),
                _retrieve_metadata(request.text),
                return_exceptions=False,
            ),
            timeout=RETRIEVAL_TIMEOUT,
        )

        # Step 2: build context — sync, instant
        context = _build_context(primary_docs, metadata)

        # Step 3: LLM call — depends on step 2
        # TODO (M3 L2): replace stub below with real async LLM call
        # answer = await asyncio.wait_for(
        #     _call_llm(f"{context}\n\n{request.text}"),
        #     timeout=LLM_TIMEOUT,
        # )
        answer = f"[stub] context={len(context)} chars, query={request.text[:50]}"

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return JSONResponse(
            content={"answer": answer, "status": "ok"},
            headers={"X-Response-Time-Ms": str(elapsed_ms)},
        )

    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return JSONResponse(
            status_code=408,
            content={"error": "Timeout exceeded", "status": "timeout", "retryable": True},
            headers={"X-Response-Time-Ms": str(elapsed_ms)},
        )

    except httpx.HTTPStatusError as exc:
        code = 429 if exc.response.status_code == 429 else 502
        return JSONResponse(
            status_code=code,
            content={"error": "LLM provider error", "status": "rate_limited" if code == 429 else "error",
                     "retryable": code == 429},
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "status": "error", "retryable": False},
        )
