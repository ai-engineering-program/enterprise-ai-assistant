import logging
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers: заглушки для ScoredPoint без реального qdrant_client
# ---------------------------------------------------------------------------

def _make_scored_point(point_id: str, tenant_id: str, text: str, score: float):
    """Создаёт объект-заглушку, имитирующий qdrant_client.http.models.ScoredPoint."""
    point = MagicMock()
    point.id = point_id
    point.score = score
    point.payload = {"tenant_id": tenant_id, "text": text}
    return point


# ---------------------------------------------------------------------------
# Unit-тесты: без запущенного Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildFilter:
    """Тесты метода _build_filter() — не требуют Qdrant."""

    def _make_retriever(self):
        from app.rag.tenant_retriever import TenantAwareRetriever
        client = MagicMock()
        return TenantAwareRetriever(client, "legal_documents")

    def test_empty_tenant_id_raises_value_error(self):
        retriever = self._make_retriever()
        with pytest.raises(ValueError):
            retriever._build_filter("")

    def test_whitespace_only_tenant_id_raises_value_error(self):
        retriever = self._make_retriever()
        with pytest.raises(ValueError):
            retriever._build_filter("   ")

    def test_valid_tenant_id_returns_filter(self):
        from qdrant_client.http.models import Filter
        retriever = self._make_retriever()
        result = retriever._build_filter("client-abc-123")
        assert isinstance(result, Filter)

    def test_filter_contains_tenant_id_condition(self):
        retriever = self._make_retriever()
        f = retriever._build_filter("client-abc-123")
        # Фильтр должен содержать ровно одно условие must
        assert len(f.must) == 1
        condition = f.must[0]
        assert condition.key == "tenant_id"
        assert condition.match.value == "client-abc-123"

    def test_custom_tenant_id_field(self):
        from app.rag.tenant_retriever import TenantAwareRetriever
        client = MagicMock()
        retriever = TenantAwareRetriever(client, "docs", tenant_id_field="org_id")
        f = retriever._build_filter("org-xyz")
        assert f.must[0].key == "org_id"


@pytest.mark.unit
class TestValidateResults:
    """Тесты метода _validate_results() — не требуют Qdrant."""

    def _make_retriever(self):
        from app.rag.tenant_retriever import TenantAwareRetriever
        client = MagicMock()
        return TenantAwareRetriever(client, "legal_documents")

    def test_valid_results_returned_as_search_results(self):
        from app.rag.tenant_retriever import SearchResult
        retriever = self._make_retriever()
        points = [
            _make_scored_point("doc-1", "tenant-a", "Договор 1", 0.92),
            _make_scored_point("doc-2", "tenant-a", "Договор 2", 0.87),
        ]
        results = retriever._validate_results(points, "tenant-a")
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].document_id == "doc-1"
        assert results[0].tenant_id == "tenant-a"
        assert results[0].text == "Договор 1"
        assert results[0].score == pytest.approx(0.92)

    def test_foreign_tenant_raises_isolation_error(self):
        from app.rag.tenant_retriever import TenantIsolationError
        retriever = self._make_retriever()
        points = [
            _make_scored_point("doc-1", "tenant-a", "Документ A", 0.95),
            _make_scored_point("doc-99", "tenant-b", "Чужой документ", 0.91),
        ]
        with pytest.raises(TenantIsolationError):
            retriever._validate_results(points, "tenant-a")

    def test_isolation_error_contains_document_info(self):
        from app.rag.tenant_retriever import TenantIsolationError
        retriever = self._make_retriever()
        points = [_make_scored_point("doc-foreign", "tenant-b", "Текст", 0.88)]
        with pytest.raises(TenantIsolationError) as exc_info:
            retriever._validate_results(points, "tenant-a")
        assert "tenant-b" in str(exc_info.value) or "doc-foreign" in str(exc_info.value)

    def test_isolation_error_logs_to_error_level(self, caplog):
        from app.rag.tenant_retriever import TenantIsolationError
        retriever = self._make_retriever()
        points = [_make_scored_point("doc-x", "tenant-b", "Чужой", 0.9)]
        with caplog.at_level(logging.ERROR, logger="app.rag.tenant_retriever"):
            with pytest.raises(TenantIsolationError):
                retriever._validate_results(points, "tenant-a")
        assert len(caplog.records) >= 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_empty_results_returns_empty_list(self):
        retriever = self._make_retriever()
        result = retriever._validate_results([], "tenant-a")
        assert result == []


@pytest.mark.unit
class TestSearch:
    """Тесты метода search() с мокированным Qdrant-клиентом."""

    def _make_retriever(self, mock_search_return=None):
        from app.rag.tenant_retriever import TenantAwareRetriever
        client = MagicMock()
        if mock_search_return is not None:
            client.search.return_value = mock_search_return
        else:
            client.search.return_value = []
        return TenantAwareRetriever(client, "legal_documents"), client

    def test_search_passes_filter_to_qdrant(self):
        from qdrant_client.http.models import Filter
        retriever, mock_client = self._make_retriever()
        retriever.search([0.1, 0.2, 0.3], tenant_id="tenant-a")
        call_kwargs = mock_client.search.call_args.kwargs
        assert "query_filter" in call_kwargs
        assert isinstance(call_kwargs["query_filter"], Filter)

    def test_search_empty_tenant_id_raises_before_qdrant_call(self):
        retriever, mock_client = self._make_retriever()
        with pytest.raises(ValueError):
            retriever.search([0.1, 0.2], tenant_id="")
        mock_client.search.assert_not_called()

    def test_search_returns_validated_results(self):
        from app.rag.tenant_retriever import SearchResult
        points = [_make_scored_point("doc-1", "tenant-a", "Текст", 0.9)]
        retriever, _ = self._make_retriever(mock_search_return=points)
        results = retriever.search([0.1] * 5, tenant_id="tenant-a")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_search_raises_if_qdrant_returns_foreign_document(self):
        from app.rag.tenant_retriever import TenantIsolationError
        # Qdrant вернул чужой документ (баг в индексе или конфигурации)
        foreign_points = [_make_scored_point("doc-x", "tenant-b", "Чужое", 0.95)]
        retriever, _ = self._make_retriever(mock_search_return=foreign_points)
        with pytest.raises(TenantIsolationError):
            retriever.search([0.1] * 5, tenant_id="tenant-a")

    def test_search_accepts_score_threshold(self):
        retriever, mock_client = self._make_retriever()
        retriever.search([0.1] * 5, tenant_id="tenant-a", score_threshold=0.8)
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs.get("score_threshold") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Integration-тесты: требуется запущенный Qdrant на localhost:6333
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTenantAwareRetrieverIntegration:
    """
    Интеграционные тесты с реальным Qdrant.
    Пропустить без сервиса: pytest -m 'not integration'
    """

    def test_two_tenants_are_isolated(self):
        pytest.skip("Требует запущенный Qdrant — запустите вручную с docker-compose up qdrant")
