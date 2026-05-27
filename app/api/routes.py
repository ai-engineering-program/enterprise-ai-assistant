"""API routes — Course 1 skeleton, expanded in Course 2 (M2 L1).

POST /query  ←  implement here in M2 L1:
    - call LLM with timeout (asyncio.wait_for)
    - return {"answer": ..., "status": "ok"} on success
    - return {"error": ..., "status": "timeout",      "retryable": True}  on timeout
    - return {"error": ..., "status": "rate_limited", "retryable": True}  on rate limit
    - return {"error": ..., "status": "error",        "retryable": False} on other errors
"""
from fastapi import APIRouter

router = APIRouter()


# TODO (M2 L1): implement POST /query with timeout and error handling
# @router.post("/query")
# async def query(request: QueryRequest) -> dict:
#     ...
