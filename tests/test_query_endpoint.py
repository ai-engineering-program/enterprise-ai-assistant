"""Tests for POST /query — sync implementation with timeout and error handling (lesson 2.1, M2 L1).

_llm_client and _retriever are the module-level objects the TODO in app/api/routes.py
instructs students to call — tests patch _llm_client.post to simulate the LLM provider.
"""
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.api import routes

client = TestClient(app)


def _fake_llm_response(content: str) -> httpx.Response:
    request = httpx.Request("POST", "http://llm.internal/chat/completions")
    return httpx.Response(
        200,
        request=request,
        json={"choices": [{"message": {"content": content}}]},
    )


def test_query_success_returns_answer(monkeypatch):
    monkeypatch.setattr(routes._llm_client, "post", lambda *a, **k: _fake_llm_response("Hello!"))

    response = client.post("/query", json={"text": "What is the leave policy?"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "answer" in data


def test_query_timeout_returns_504(monkeypatch):
    def timeout_post(*args, **kwargs):
        raise httpx.TimeoutException("LLM did not respond in time")

    monkeypatch.setattr(routes._llm_client, "post", timeout_post)

    response = client.post("/query", json={"text": "slow query"})

    assert response.status_code == 504
    data = response.json()
    assert data["status"] == "timeout"
    assert data["retryable"] is True


def test_query_rate_limit_returns_429(monkeypatch):
    def rate_limited_post(*args, **kwargs):
        request = httpx.Request("POST", "http://llm.internal/chat/completions")
        resp = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=resp)

    monkeypatch.setattr(routes._llm_client, "post", rate_limited_post)

    response = client.post("/query", json={"text": "busy query"})

    assert response.status_code == 429
    data = response.json()
    assert data["status"] == "rate_limited"
    assert data["retryable"] is True


def test_health_not_broken_by_query_changes():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
