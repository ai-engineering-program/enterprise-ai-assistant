"""
Architecture specification for enterprise-ai-assistant.

This module defines the current production architecture spec used by
ArchitectureValidator (lesson 6.5) to verify production readiness.

Each key corresponds to a component built across the six modules of
the AI System Architecture course.
"""

ARCHITECTURE_SPEC = {
    # Module 1 — API layer and auth
    "api_layer": {
        "async_support": True,
        "framework": "fastapi",
        "description": "Async FastAPI with JWT auth and rate limiting",
    },
    "auth": {
        "enabled": True,
        "method": "jwt",
        "permission_cache": "redis",
        "description": "JWT validation + LDAP permission cache in gateway",
    },
    "rate_limiting": {
        "enabled": True,
        "rps_per_user": 10,
        "rps_per_department": 50,
        "backend": "redis",
    },
    # Module 2 — Architectural patterns: RAG
    "rag_pipeline": {
        "enabled": True,
        "stages": ["query_expansion", "vector_search", "permission_filter", "rerank", "context_assembly"],
        "permission_filter_position": "before_rerank",
    },
    # Module 3 — Sync/async: caching
    "caching": {
        "enabled": True,
        "backend": "redis",
        "semantic_cache": True,
        "ttl_seconds": 300,
    },
    # Module 4 — Orchestration: workflow engine
    "workflow_engine": {
        "enabled": True,
        "backend": "temporal",
        "description": "Long-running tasks: doc creation, onboarding, batch reindex",
    },
    # Module 5 — Queues: background queue
    "queue": {
        "enabled": True,
        "backend": "celery",
        "broker": "rabbitmq",
        "description": "Background ingestion workers and async tasks",
    },
    # Module 6 — Event-driven: event bus
    "event_bus": {
        "enabled": True,
        "backend": "kafka",
        "topics": ["document.created", "document.updated", "document.deleted"],
        "description": "Event-driven ingestion pipeline trigger",
    },
    # Reliability — circuit breaker and graceful degradation
    "circuit_breaker": {
        "enabled": True,
        "failure_threshold": 0.05,
        "recovery_timeout_seconds": 30,
        "description": "Per-LLM-provider circuit breaker",
    },
    "graceful_degradation": {
        "enabled": True,
        "levels": 3,
        "description": "L1: fallback model, L2: search-only, L3: static FAQ cache",
    },
    # Routing — model/cost router
    "router": {
        "enabled": True,
        "strategy": "cost_aware",
        "tiers": ["small", "medium", "large"],
        "description": "Classifies query complexity, routes to appropriate LLM tier",
    },
    # Observability
    "observability": {
        "enabled": True,
        "metrics": "prometheus",
        "tracing": "opentelemetry",
        "cost_tracking": True,
        "description": "Full-stack observability with trace_id propagation",
    },
}
