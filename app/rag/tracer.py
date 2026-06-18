from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Span:
    """Период трассировки — один шаг RAG-конвейера."""

    name: str
    started_at: str
    ended_at: Optional[str] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def finish(self, outputs: dict, metadata: dict = None) -> None:
        # TODO: зафиксировать ended_at как текущее время UTC в формате ISO 8601
        # TODO: сохранить outputs в self.outputs
        # TODO: если metadata передан — обновить self.metadata через update()
        ...

    @property
    def duration_ms(self) -> Optional[float]:
        # TODO: вычислить длительность в миллисекундах
        # Если ended_at is None — вернуть None
        # Подсказка: datetime.fromisoformat() + total_seconds() * 1000
        ...


@dataclass
class Trace:
    """Полная трассировка одного запроса к RAG-конвейеру."""

    trace_id: str
    query: str
    created_at: str
    spans: list[Span] = field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        # TODO: сериализовать трассировку в словарь
        # Включить: trace_id, query, created_at, final_answer, error, spans
        # Каждый период в spans: name, started_at, ended_at, duration_ms,
        #                        inputs, outputs, metadata
        ...


class RAGTracer:
    """Лёгкий структурированный трассировщик для RAG-конвейера.

    Записывает одну трассировку (Trace) на запрос в JSONL-файл.
    Не требует внешних зависимостей — только стандартная библиотека.

    Параметры:
        log_path: путь к файлу для записи трассировок (JSONL-формат).
    """

    def __init__(self, log_path: str = "rag_traces.jsonl") -> None:
        # TODO: сохранить log_path
        # TODO: инициализировать self._current_trace = None
        ...

    def start_trace(self, query: str) -> Trace:
        # TODO: создать Trace с уникальным trace_id (uuid4) и текущим временем UTC
        # TODO: сохранить в self._current_trace
        # TODO: вернуть созданную трассировку
        ...

    def start_span(self, name: str, inputs: dict = None) -> Span:
        # TODO: создать Span с текущим временем UTC и переданными inputs
        # TODO: добавить период в self._current_trace.spans
        # TODO: вернуть созданный Span
        ...

    def finish_span(self, span: Span, outputs: dict,
                    metadata: dict = None) -> None:
        # TODO: вызвать span.finish(outputs, metadata)
        ...

    def finish_trace(self, answer: str = None,
                     error: str = None) -> Trace:
        # TODO: сохранить answer в self._current_trace.final_answer
        # TODO: сохранить error в self._current_trace.error
        # TODO: вызвать self._write() для записи трассировки в файл
        # TODO: сохранить ссылку на завершённую трассировку
        # TODO: сбросить self._current_trace = None
        # TODO: вернуть завершённую трассировку
        ...

    def _write(self, trace: Trace) -> None:
        # TODO: открыть self._log_path в режиме дозаписи ("a", encoding="utf-8")
        # TODO: записать одну строку: json.dumps(trace.to_dict(), ensure_ascii=False)
        # TODO: добавить символ новой строки "\n" после JSON-объекта
        ...


__all__ = ["Span", "Trace", "RAGTracer"]
