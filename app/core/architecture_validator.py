"""
ArchitectureValidator — production readiness checker for AI system architecture specs.

Used in lesson 6.5 (Final Project) of the AI System Architecture course.
Run validate_arch.py from the repo root to check the current architecture spec.
"""

from typing import Any


class ArchitectureValidator:
    """
    Validates an architecture specification dict against production readiness rules.

    Each rule corresponds to a pattern covered in one of the six course modules.
    """

    REQUIRED_COMPONENTS = [
        ("api_layer", "async_support", "Missing api_layer with async_support=True"),
        ("caching", "enabled", "Missing caching layer"),
        ("circuit_breaker", "enabled", "Missing circuit_breaker configuration"),
        ("queue", "enabled", "Missing queue for background work"),
        ("observability", "enabled", "Missing observability section"),
        ("auth", "enabled", "Missing auth (authorization)"),
        ("rate_limiting", "enabled", "Missing rate_limiting"),
    ]

    EVENT_COMPONENTS = ("event_bus", "event_store")

    BONUS_COMPONENTS = [
        ("router", "enabled"),
        ("graceful_degradation", "enabled"),
    ]

    def _component_present(self, spec: dict, key: str, field: str) -> bool:
        """Return True if spec[key][field] is truthy."""
        section = spec.get(key)
        if not isinstance(section, dict):
            return False
        return bool(section.get(field))

    def validate(self, spec: dict) -> list[str]:
        """
        Validate architecture spec against production readiness rules.

        Returns a list of issue strings. Empty list means all checks passed.
        """
        issues = []

        for key, field, message in self.REQUIRED_COMPONENTS:
            if not self._component_present(spec, key, field):
                issues.append(message)

        has_event = any(
            self._component_present(spec, ec, "enabled")
            for ec in self.EVENT_COMPONENTS
        )
        if not has_event:
            issues.append("Missing event_bus or event_store")

        return issues

    def get_score(self, spec: dict) -> int:
        """
        Return a production readiness score from 0 to 100.

        10 points per satisfied best practice (10 practices total).
        """
        count = 0

        for key, field, _ in self.REQUIRED_COMPONENTS:
            if self._component_present(spec, key, field):
                count += 1

        has_event = any(
            self._component_present(spec, ec, "enabled")
            for ec in self.EVENT_COMPONENTS
        )
        if has_event:
            count += 1

        for key, field in self.BONUS_COMPONENTS:
            if self._component_present(spec, key, field):
                count += 1

        return count * 10

    def suggest_improvements(self, spec: dict) -> list[str]:
        """
        Return a list of actionable improvement suggestions for issues found.
        """
        SUGGESTIONS = {
            "Missing api_layer with async_support=True": (
                "Add an async API layer (FastAPI/aiohttp) to handle 120+ RPS "
                "without blocking threads"
            ),
            "Missing caching layer": (
                "Add a cache (Redis/Memcached) to reduce LLM load for repeated queries"
            ),
            "Missing circuit_breaker configuration": (
                "Add a circuit breaker to prevent cascade failures "
                "when the LLM provider is unavailable"
            ),
            "Missing queue for background work": (
                "Add a task queue (Celery/RQ) to run heavy operations "
                "asynchronously without blocking the API"
            ),
            "Missing event_bus or event_store": (
                "Add an event bus (Kafka/RabbitMQ) or event store to enable "
                "event-driven ingestion with lag < 5 minutes"
            ),
            "Missing observability section": (
                "Add observability (metrics, tracing, logs) to see latency "
                "and errors across each layer"
            ),
            "Missing auth (authorization)": (
                "Add authorization in the API gateway to check access rights "
                "before passing the request to the RAG pipeline"
            ),
            "Missing rate_limiting": (
                "Add rate limiting to protect the system from bursts "
                "and prevent exceeding LLM API quotas"
            ),
        }

        return [
            SUGGESTIONS.get(issue, f"Fix issue: {issue}")
            for issue in self.validate(spec)
        ]
