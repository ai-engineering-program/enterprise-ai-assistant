"""API routes — sync implementation, Course 1 M2 L1.

POST /query blocks the worker thread for the full duration of retrieval + LLM call.
This causes observable degradation under load (tested in M3 L1).
Refactored to async in M3 L2.
"""
import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.retriever import RAGRetriever

router = APIRouter()

# Sync HTTP client — blocks the worker thread on every I/O call.
# Replaced with httpx.AsyncClient in M3 L2.
_llm_client = httpx.Client(base_url="http://localhost:8001", timeout=30.0)

# RAG retriever — stub from M2 L2; full implementation in Course 2.
_retriever = RAGRetriever(index=[])

LLM_TIMEOUT = 30.0  # seconds


class QueryRequest(BaseModel):
    text: str


@router.post("/query")
def query(request: QueryRequest) -> dict:
    """POST /query — synchronous (blocks worker thread).

    TODO (M2 L1): implement after completing the Stepik exercise:
      1. context = _retriever.build_context(_retriever.retrieve(request.text))
      2. response = _llm_client.post(
             "/chat/completions",
             json={"model": "gpt-4o",
                   "messages": [{"role": "user", "content": f"{context}\\n\\n{request.text}"}]},
             timeout=LLM_TIMEOUT,
         )
         response.raise_for_status()
      3. Return {"answer": response.json()["choices"][0]["message"]["content"], "status": "ok"}
         on success, or error dict on timeout / rate-limit / other errors.

    Note: each request holds one FastAPI worker thread for ~6 s (retrieval + LLM).
    At concurrency > workers the queue grows without bound — this is the degradation
    you measure in M3 L1 and fix in M3 L2.
    """
    raise NotImplementedError(
        "Implement in M2 L1. See TODO above and the Stepik exercise for handle_query()."
    )
