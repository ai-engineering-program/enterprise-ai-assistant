import warnings
import pytest

from app.rag.model_registry import IndexMetadata, ModelRegistry


@pytest.mark.unit
class TestModelRegistryUnit:
    """Unit tests — no external services required."""

    def test_register_returns_metadata(self):
        registry = ModelRegistry()
        meta = registry.register("docs_v1", "all-MiniLM-L6-v2")
        assert isinstance(meta, IndexMetadata)
        assert meta.model_name == "all-MiniLM-L6-v2"
        assert meta.collection_name == "docs_v1"
        assert meta.indexed_at  # должна быть непустая временная метка

    def test_register_stores_metadata(self):
        registry = ModelRegistry()
        registry.register("docs_v1", "all-MiniLM-L6-v2")
        # повторная проверка совместимости должна найти запись
        result = registry.check_compatibility("docs_v1", "all-MiniLM-L6-v2")
        assert result is True

    def test_compatible_model_no_warning(self):
        registry = ModelRegistry()
        registry.register("docs_v1", "all-MiniLM-L6-v2")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = registry.check_compatibility("docs_v1", "all-MiniLM-L6-v2")
        assert result is True
        assert len(caught) == 0, "Совместимая модель не должна вызывать предупреждений"

    def test_incompatible_model_warns(self):
        registry = ModelRegistry()
        registry.register("docs_v1", "all-MiniLM-L6-v2")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = registry.check_compatibility("docs_v1", "all-mpnet-base-v2")
        assert result is False
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        msg = str(caught[0].message)
        assert "all-MiniLM-L6-v2" in msg
        assert "all-mpnet-base-v2" in msg

    def test_unknown_collection_returns_true(self):
        registry = ModelRegistry()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = registry.check_compatibility("nonexistent", "any-model")
        assert result is True
        assert len(caught) == 0, "Незарегистрированная коллекция не должна вызывать предупреждений"

    def test_multiple_collections_independent(self):
        registry = ModelRegistry()
        registry.register("col_a", "model-A")
        registry.register("col_b", "model-B")
        # col_a совместима
        assert registry.check_compatibility("col_a", "model-A") is True
        # col_b несовместима с model-A
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = registry.check_compatibility("col_b", "model-A")
        assert result is False
        assert len(caught) == 1


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests — require running Qdrant.

    Run with: pytest tests/test_model_registry.py -m integration
    Skip with: pytest tests/test_model_registry.py -m 'not integration'
    """

    def test_placeholder(self):
        pytest.skip(
            "ModelRegistry не обращается к Qdrant напрямую — интеграционные тесты "
            "добавляются в уроке 2.3 вместе с реальным Qdrant-клиентом."
        )
