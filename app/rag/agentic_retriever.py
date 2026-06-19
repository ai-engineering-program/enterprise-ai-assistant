from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import openai

__all__ = [
    "TOOLS",
    "ToolResult",
    "ToolExecutor",
    "AgenticRetriever",
]


# ---------------------------------------------------------------------------
# Схемы инструментов
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Семантический поиск по смыслу запроса. "
                "Используй для концептуальных вопросов о стратегии, "
                "политике, тенденциях. "
                "НЕ используй для точных дат, чисел, названий."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на естественном языке",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Число результатов (по умолчанию 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": (
                "Точный поиск по ключевым словам (BM25). "
                "Используй для дат, числовых значений, "
                "названий компаний, нормативных актов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Число результатов (по умолчанию 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_date",
            "description": (
                "Фильтрует уже найденные документы по дате публикации. "
                "Используй ПОСЛЕ vector_search или keyword_search, "
                "если вопрос содержит временной контекст "
                "(год, период, дата события)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents_json": {
                        "type": "string",
                        "description": (
                            "JSON-строка: список документов "
                            "из предыдущего поиска"
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Дата от (YYYY-MM-DD), включительно",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Дата до (YYYY-MM-DD), включительно",
                    },
                },
                "required": ["documents_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_findings",
            "description": (
                "Суммаризировать накопленные результаты перед финальным ответом. "
                "Используй когда собрано достаточно данных из нескольких поисков."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": "Накопленные данные для суммаризации",
                    },
                },
                "required": ["findings"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Результат вызова одного инструмента в агентном цикле.

    Attributes:
        tool_name:      имя вызванного инструмента.
        args:           аргументы, переданные инструменту.
        result_preview: первые 200 символов результата (для логирования).
        iteration:      номер итерации агентного цикла (начиная с 1).
    """

    tool_name: str
    args: dict[str, Any]
    result_preview: str
    iteration: int


# ---------------------------------------------------------------------------
# Исполнитель инструментов
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Исполняет инструменты агентного поиска.

    Параметры:
        vector_search_fn:  callable(query: str, top_k: int) -> list[dict]
                           Внешняя функция векторного поиска.
                           Каждый элемент списка — словарь с ключами:
                           "text", "source", "date", "score".
        keyword_search_fn: callable(query: str, top_k: int) -> list[dict]
                           Внешняя функция ключевого поиска.
        llm_summarize_fn:  callable(text: str) -> str
                           Функция суммаризации через языковую модель.
                           Может быть None — тогда используется заглушка.
    """

    def __init__(
        self,
        vector_search_fn,
        keyword_search_fn,
        llm_summarize_fn=None,
    ) -> None:
        self._vector_search = vector_search_fn
        self._keyword_search = keyword_search_fn
        self._llm_summarize = llm_summarize_fn

    def vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        # TODO: вызвать self._vector_search(query=query, top_k=top_k)
        # TODO: вернуть результат (список словарей)
        ...

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        # TODO: вызвать self._keyword_search(query=query, top_k=top_k)
        # TODO: вернуть результат (список словарей)
        ...

    def filter_by_date(
        self,
        documents_json: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Фильтрует документы по дате публикации.

        TODO: реализуйте этот метод.

        Алгоритм:
        1. Десериализовать documents_json через json.loads().
        2. Для каждого документа взять поле "date" (строка YYYY-MM-DD).
        3. Если поле "date" отсутствует или пустое — пропустить документ.
        4. Если date_from задан — оставить только документы с date >= date_from.
        5. Если date_to задан — оставить только документы с date <= date_to.
        6. Если оба параметра None — вернуть все документы без фильтрации.
        7. Вернуть отфильтрованный список словарей.

        Граничные случаи:
        - documents_json == "[]" → вернуть []
        - Сравнение дат лексикографическое (YYYY-MM-DD сравниваются корректно)
        """
        # TODO: реализовать
        ...

    def summarize_findings(self, findings: str) -> str:
        # TODO: если self._llm_summarize задан — вызвать его с findings
        # TODO: иначе — вернуть findings[:500] как заглушку
        ...

    def dispatch(self, name: str, args: dict) -> str:
        """Диспетчеризует вызов инструмента по имени.

        TODO: реализуйте обработку всех четырёх инструментов:
        - "vector_search"    → self.vector_search(**args) → json.dumps(...)
        - "keyword_search"   → self.keyword_search(**args) → json.dumps(...)
        - "filter_by_date"   → self.filter_by_date(**args) → json.dumps(...)
        - "summarize_findings" → self.summarize_findings(**args)
        - неизвестный инструмент → json.dumps({"error": "..."})
        """
        # TODO: реализовать
        ...


# ---------------------------------------------------------------------------
# Агентный цикл
# ---------------------------------------------------------------------------

class AgenticRetriever:
    """Агентный поисковик: языковая модель управляет стратегией поиска.

    Параметры:
        executor:       экземпляр ToolExecutor с настроенными бэкендами.
        model:          идентификатор модели OpenAI (по умолчанию gpt-4o-mini).
        max_iterations: максимальное число итераций агентного цикла.
        system_prompt:  системный промпт для агента (опционально).

    Пример использования:
        executor = ToolExecutor(
            vector_search_fn=my_qdrant_search,
            keyword_search_fn=my_bm25_search,
        )
        agent = AgenticRetriever(executor=executor)
        result = agent.run("Как изменилась дивидендная политика Газпрома?")
        print(result["answer"])
        print(result["tool_calls"])  # список ToolResult
    """

    _DEFAULT_SYSTEM = (
        "Ты — аналитический ассистент. "
        "Для ответа на вопрос используй доступные инструменты поиска. "
        "Планируй стратегию: декомпозируй составные вопросы, "
        "выполняй поиск последовательно, синтезируй результаты. "
        "Отвечай только на основе найденных документов."
    )

    def __init__(
        self,
        executor: ToolExecutor,
        model: str = "gpt-4o-mini",
        max_iterations: int = 6,
        system_prompt: str | None = None,
    ) -> None:
        # TODO: сохранить executor в self.executor
        # TODO: сохранить model в self.model
        # TODO: сохранить max_iterations в self.max_iterations
        # TODO: сохранить system_prompt (или _DEFAULT_SYSTEM если None)
        #       в self.system_prompt
        # TODO: инициализировать self.client = openai.OpenAI()
        ...

    def run(self, question: str) -> dict:
        """Запускает агентный цикл для ответа на вопрос.

        TODO: реализуйте этот метод.

        Алгоритм:
        1. Инициализировать messages:
               [{"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": question}]
           Инициализировать tool_calls_log: list[ToolResult] = []
           Инициализировать iteration = 0

        2. Цикл while iteration < self.max_iterations:
               iteration += 1
               Вызвать self.client.chat.completions.create(
                   model=self.model,
                   messages=messages,
                   tools=TOOLS,
                   tool_choice="auto",
               )
               msg = response.choices[0].message

               Если msg.tool_calls is None или пустой:
                   # Модель готова дать финальный ответ
                   вернуть {
                       "answer": msg.content,
                       "iterations": iteration,
                       "tool_calls": tool_calls_log,
                   }

               Добавить msg в messages.
               Для каждого tc в msg.tool_calls:
                   args = json.loads(tc.function.arguments)
                   result_str = self.executor.dispatch(tc.function.name, args)
                   Добавить ToolResult(...) в tool_calls_log.
                   Добавить {"role": "tool", "tool_call_id": tc.id,
                              "content": result_str} в messages.

        3. После выхода из цикла (лимит исчерпан):
               Добавить сообщение пользователя с просьбой сформулировать
               лучший ответ на основе найденного.
               Выполнить финальный вызов языковой модели.
               Вернуть {
                   "answer": final_content,
                   "iterations": iteration,
                   "tool_calls": tool_calls_log,
                   "warning": "max_iterations_reached",
               }
        """
        # TODO: реализовать
        ...
