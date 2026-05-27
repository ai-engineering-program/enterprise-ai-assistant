"""Tests for POST /query — async implementation with timeout and error handling (lesson 3.2, M3 L2).

Async refactor replaced the sync _llm_client/M2 L1 interface with _retrieve_primary /
_retrieve_metadata (see app/api/routes.py) — tests patch those to simulate the LLM provider.
"""
import asyncio

import httpx
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.api import routes


@pytest.mark.asyncio
async def test_query_success_returns_answer():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"text": "What is the leave policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "answer" in data
    assert "X-Response-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_query_timeout_returns_504(monkeypatch):
    monkeypatch.setattr(routes, "RETRIEVAL_TIMEOUT", 0.05)

    async def slow_retrieve(query):
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr(routes, "_retrieve_primary", slow_retrieve)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"text": "slow query"})

    assert response.status_code == 504
    data = response.json()
    assert data["status"] == "timeout"
    assert data["retryable"] is True


@pytest.mark.asyncio
async def test_query_rate_limit_returns_429(monkeypatch):
    async def rate_limited_retrieve(query):
        request = httpx.Request("POST", "http://llm.internal/chat/completions")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(routes, "_retrieve_primary", rate_limited_retrieve)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"text": "busy query"})

    assert response.status_code == 429
    data = response.json()
    assert data["status"] == "rate_limited"
    assert data["retryable"] is True


@pytest.mark.asyncio
async def test_health_not_broken_by_query_changes():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
