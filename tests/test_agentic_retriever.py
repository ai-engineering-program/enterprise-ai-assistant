import json
import pytest

from app.rag.agentic_retriever import (
    TOOLS,
    ToolResult,
    ToolExecutor,
    AgenticRetriever,
)


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

def _make_docs(entries: list[dict]) -> str:
    """Сериализует список документов в JSON-строку для filter_by_date."""
    return json.dumps(entries, ensure_ascii=False)


def _stub_vector_search(query: str, top_k: int = 5) -> list[dict]:
    return [
        {"text": f"Документ {i}", "source": f"doc_{i}", "date": f"2022-0{i+1}-01", "score": 0.9 - i * 0.1}
        for i in range(min(top_k, 3))
    ]


def _stub_keyword_search(query: str, top_k: int = 5) -> list[dict]:
    return [
        {"text": f"Ключевой документ {i}", "source": f"kw_{i}", "date": f"2023-0{i+1}-15", "score": 10 - i}
        for i in range(min(top_k, 2))
    ]


@pytest.fixture
def executor() -> ToolExecutor:
    return ToolExecutor(
        vector_search_fn=_stub_vector_search,
        keyword_search_fn=_stub_keyword_search,
    )


# ---------------------------------------------------------------------------
# Unit-тесты: TOOLS
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTools:
    def test_tools_list_not_empty(self):
        assert len(TOOLS) >= 3

    def test_required_tool_names(self):
        names = {t["function"]["name"] for t in TOOLS}
        assert "vector_search" in names
        assert "keyword_search" in names
        assert "filter_by_date" in names

    def test_filter_by_date_has_required_params(self):
        schema = next(
            t for t in TOOLS if t["function"]["name"] == "filter_by_date"
        )
        required = schema["function"]["parameters"].get("required", [])
        assert "documents_json" in required

    def test_vector_search_has_description(self):
        schema = next(
            t for t in TOOLS if t["function"]["name"] == "vector_search"
        )
        desc = schema["function"]["description"]
        assert len(desc) > 20, "Описание инструмента слишком короткое"


# ---------------------------------------------------------------------------
# Unit-тесты: ToolExecutor.vector_search
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToolExecutorVectorSearch:
    def test_returns_list(self, executor: ToolExecutor):
        result = executor.vector_search("дивиденды Газпром", top_k=3)
        assert isinstance(result, list)

    def test_returns_correct_count(self, executor: ToolExecutor):
        result = executor.vector_search("тест", top_k=2)
        assert len(result) <= 2

    def test_result_has_text_field(self, executor: ToolExecutor):
        result = executor.vector_search("тест", top_k=1)
        assert len(result) > 0
        assert "text" in result[0]


# ---------------------------------------------------------------------------
# Unit-тесты: ToolExecutor.keyword_search
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToolExecutorKeywordSearch:
    def test_returns_list(self, executor: ToolExecutor):
        result = executor.keyword_search("санкции 2022", top_k=2)
        assert isinstance(result, list)

    def test_result_not_empty(self, executor: ToolExecutor):
        result = executor.keyword_search("тест", top_k=3)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Unit-тесты: ToolExecutor.filter_by_date
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilterByDate:
    def _docs(self) -> list[dict]:
        return [
            {"text": "Отчёт Q1", "source": "rpt_1", "date": "2021-03-01", "score": 0.9},
            {"text": "Отчёт Q2", "source": "rpt_2", "date": "2022-06-15", "score": 0.8},
            {"text": "Отчёт Q3", "source": "rpt_3", "date": "2023-09-30", "score": 0.7},
            {"text": "Без даты", "source": "rpt_4",                         "score": 0.6},
            {"text": "Пустая дата", "source": "rpt_5", "date": "",         "score": 0.5},
        ]

    def test_no_filter_returns_docs_with_date(self, executor: ToolExecutor):
        docs_json = _make_docs(self._docs())
        result = executor.filter_by_date(docs_json)
        # Документы без поля date или с пустой датой должны быть отфильтрованы
        sources = [d["source"] for d in result]
        assert "rpt_4" not in sources
        assert "rpt_5" not in sources

    def test_filter_date_from(self, executor: ToolExecutor):
        docs_json = _make_docs(self._docs())
        result = executor.filter_by_date(docs_json, date_from="2022-01-01")
        sources = [d["source"] for d in result]
        assert "rpt_1" not in sources  # 2021 < 2022
        assert "rpt_2" in sources
        assert "rpt_3" in sources

    def test_filter_date_to(self, executor: ToolExecutor):
        docs_json = _make_docs(self._docs())
        result = executor.filter_by_date(docs_json, date_to="2022-12-31")
        sources = [d["source"] for d in result]
        assert "rpt_3" not in sources  # 2023 > 2022
        assert "rpt_1" in sources
        assert "rpt_2" in sources

    def test_filter_date_range(self, executor: ToolExecutor):
        docs_json = _make_docs(self._docs())
        result = executor.filter_by_date(
            docs_json, date_from="2022-01-01", date_to="2022-12-31"
        )
        sources = [d["source"] for d in result]
        assert sources == ["rpt_2"]

    def test_empty_input(self, executor: ToolExecutor):
        result = executor.filter_by_date("[]", date_from="2022-01-01")
        assert result == []

    def test_doc_without_date_field_excluded(self, executor: ToolExecutor):
        docs = [{"text": "Документ без даты", "source": "x"}]
        result = executor.filter_by_date(_make_docs(docs))
        assert result == []

    def test_doc_with_empty_date_excluded(self, executor: ToolExecutor):
        docs = [{"text": "Документ", "source": "y", "date": ""}]
        result = executor.filter_by_date(_make_docs(docs))
        assert result == []

    def test_boundary_inclusive_from(self, executor: ToolExecutor):
        docs = [{"text": "Граница", "source": "z", "date": "2022-01-01"}]
        result = executor.filter_by_date(_make_docs(docs), date_from="2022-01-01")
        assert len(result) == 1

    def test_boundary_inclusive_to(self, executor: ToolExecutor):
        docs = [{"text": "Граница", "source": "z", "date": "2022-12-31"}]
        result = executor.filter_by_date(_make_docs(docs), date_to="2022-12-31")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Unit-тесты: ToolExecutor.dispatch
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDispatch:
    def test_dispatch_vector_search(self, executor: ToolExecutor):
        result_str = executor.dispatch("vector_search", {"query": "тест", "top_k": 2})
        result = json.loads(result_str)
        assert isinstance(result, list)

    def test_dispatch_keyword_search(self, executor: ToolExecutor):
        result_str = executor.dispatch("keyword_search", {"query": "тест", "top_k": 2})
        result = json.loads(result_str)
        assert isinstance(result, list)

    def test_dispatch_filter_by_date(self, executor: ToolExecutor):
        docs = [{"text": "Т", "source": "s", "date": "2023-01-01"}]
        result_str = executor.dispatch(
            "filter_by_date",
            {"documents_json": json.dumps(docs), "date_from": "2022-01-01"},
        )
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_dispatch_unknown_tool(self, executor: ToolExecutor):
        result_str = executor.dispatch("nonexistent_tool", {})
        result = json.loads(result_str)
        assert "error" in result

    def test_dispatch_summarize_findings(self, executor: ToolExecutor):
        result = executor.dispatch(
            "summarize_findings", {"findings": "Газпром сократил дивиденды."}
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Unit-тесты: AgenticRetriever (без вызова OpenAI API)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgenticRetrieverInit:
    def test_init_stores_executor(self, executor: ToolExecutor):
        agent = AgenticRetriever(executor=executor)
        assert agent.executor is executor

    def test_init_default_model(self, executor: ToolExecutor):
        agent = AgenticRetriever(executor=executor)
        assert agent.model == "gpt-4o-mini"

    def test_init_custom_model(self, executor: ToolExecutor):
        agent = AgenticRetriever(executor=executor, model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_init_default_max_iterations(self, executor: ToolExecutor):
        agent = AgenticRetriever(executor=executor)
        assert agent.max_iterations == 6

    def test_init_custom_system_prompt(self, executor: ToolExecutor):
        custom = "Ты специализированный финансовый аналитик."
        agent = AgenticRetriever(executor=executor, system_prompt=custom)
        assert agent.system_prompt == custom

    def test_init_default_system_prompt_not_empty(self, executor: ToolExecutor):
        agent = AgenticRetriever(executor=executor)
        assert agent.system_prompt and len(agent.system_prompt) > 10


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют OpenAI API и Qdrant)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgenticRetrieverIntegration:
    def test_run_simple_question(self, executor: ToolExecutor):
        pytest.skip(
            "Требует реальный OpenAI API ключ. "
            "Запустите вручную: pytest -m integration"
        )

    def test_run_compound_question(self, executor: ToolExecutor):
        pytest.skip(
            "Требует реальный OpenAI API ключ и запущенный Qdrant. "
            "Запустите вручную: pytest -m integration"
        )
