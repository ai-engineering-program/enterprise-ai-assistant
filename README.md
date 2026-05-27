# Enterprise AI Assistant

Production AI assistant system built incrementally across 11 courses of the AI Engineering for Production program.

## Course progression

| Course | What gets added | Git tag |
|--------|-----------------|---------|
| 1 — AI System Architecture | Skeleton FastAPI app | `v1.0-skeleton` |
| 2 — RAG Systems | RAG layer (`app/rag/`) | `v2.0-rag` |
| 3 — Data Ingestion | Ingestion pipeline (`app/ingestion/`) | `v3.0-ingestion` |
| 4 — Context Engineering | Context engine (`app/context/`) | `v4.0-context` |
| 5 — AI Routing | Routing layer (`app/routing/`) | `v5.0-routing` |
| 6 — Structured Output + Economics | Structured generation | `v6.0-structured` |
| 7 — Agents & Workflows | Agents layer (`app/agents/`) | `v7.0-agents` |
| 8 — Evaluation + LLMOps | Evaluation layer (`app/evaluation/`) | `v8.0-eval` |
| 9 — Reliability + Inference | Reliability layer (`app/reliability/`) | `v9.0-reliability` |
| 10 — Deployment & Scaling | Scaling configuration | `v10.0-scaling` |
| 11 — Final Project | Full production system | `v11.0-production` |

## Repository structure

```
app/
├── main.py          ← FastAPI entry point (Course 1)
├── api/             ← API layer — request handling (Course 1)
├── routing/         ← AI routing layer — model/cost routing (Course 5)
├── rag/             ← RAG layer — retrieval-augmented generation (Course 2)
├── ingestion/       ← Ingestion pipeline — document processing (Course 3)
├── context/         ← Context engine — compression & memory (Course 4)
├── agents/          ← Agents & workflows — LangGraph (Course 7)
├── evaluation/      ← Evaluation layer — RAGAS & regression (Course 8)
├── reliability/     ← Reliability — circuit breaker, retries (Course 9)
├── cache/           ← Semantic cache — embedding-based hit rate (Course 5)
├── observability/   ← Metrics, tracing, cost tracking (Course 8)
├── llm/             ← LLM provider clients — OpenAI, Anthropic (Course 1)
└── core/            ← Shared utilities, config, base classes (Course 1)
tests/
requirements.txt
```

## Quick start (Course 1)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://localhost:8000/health
```
