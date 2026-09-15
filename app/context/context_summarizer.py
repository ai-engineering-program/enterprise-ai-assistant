from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.rag.context_builder import ContextChunk


__all__ = [
    "SummarizeFn",
    "SummarizationResult",
    "estimate_tokens",
    "DEFAULT_CHUNK_TOKEN_THRESHOLD",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "ContextSummarizer",
]


# Функция, которая выполняет собственно суммаризацию одного промпта в более
# короткий текст. В production это вызов LLM (system/user промпт ->
# completion), но для этого слоя конкретная реализация вызова неважна:
# ContextSummarizer работает с ЛЮБОЙ функцией такой сигнатуры через
# dependency injection — тот же принцип, что у RetrievalPlanner (3.4) с
# decomposer/multi_hop_retriever/source_selector. В уроке 4.2 сюда же можно
# подставить экстрактивную (rule-based, без LLM) функцию — интерфейс
# ContextSummarizer не изменится ни на строчку.
SummarizeFn = Callable[[str], str]


# Порог "чанк достаточно крупный, чтобы его вообще стоило суммаризировать".
# Суммаризация не бесплатна ни по деньгам (вызов LLM), ни по риску (см.
# инцидент этого урока) — применять её к каждому найденному фрагменту
# подряд неоправданно, если фрагмент и так короткий.
DEFAULT_CHUNK_TOKEN_THRESHOLD: int = 400

# Общий бюджет токенов для всего контекстного блока, который в итоге уйдёт
# в ContextBuilder (курс 2). Это НЕ полное окно модели — системный промпт и
# история диалога тоже занимают часть бюджета (см. текст урока, раздел 2).
DEFAULT_MAX_CONTEXT_TOKENS: int = 2000


def estimate_tokens(text: str) -> int:
    """
    Грубая оценка числа токенов без обращения к токенизатору модели.

    TODO: вернуть len(text) // 4 — эвристика "~4 символа на токен",
    достаточная для бюджетной прикидки на этом уровне, но НЕ для точного
    биллинга (точный подсчёт токенов — тема модуля M6 "Длинный контекст").
    Для пустой строки вернуть 0.
    """
    ...


@dataclass
class SummarizationResult:
    """
    Итог одного вызова ContextSummarizer.compress().

    chunks — финальный список чанков (после точечной суммаризации и/или
    отбрасывания части чанков), готовый к передаче в ContextBuilder.build()
    (app/rag/context_builder.py, курс 2).

    summarized_indices и dropped_indices ссылаются на позиции чанков в
    ИСХОДНОМ списке, переданном в compress(), а не в финальном chunks — это
    нужно, чтобы при разборе инцидента (постфактум, как в уроке 3.5) можно
    было точно сказать, какой из исходных фрагментов был сокращён, а какой
    отброшен целиком.
    """

    chunks: list[ContextChunk]
    original_token_count: int
    final_token_count: int
    summarized_indices: list[int] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)


class ContextSummarizer:
    """
    Первый и самый простой слой компрессии контекста (модуль M4, урок
    4.1): укладывает уже найденные RetrievalPlanner'ом (M3) фрагменты в
    бюджет токенов модели ДО того, как ContextBuilder (курс 2, RAG)
    соберёт из них финальный промпт.

    Точка входа в конвейер компрессии — ровно между "retrieval вернул
    фрагменты" и "ContextBuilder собрал промпт" (см. диаграмму
    diagram_module_placement урока 4.1). ContextSummarizer не выбирает
    источники и не решает, какие фрагменты искать, — эти решения уже
    приняты модулем M3; его единственная ответственность — уместить УЖЕ
    найденное в бюджет, минимально жертвуя информацией, нужной для ответа
    на query.

    Суммаризация в этом классе намеренно однопроходная и query-aware (см.
    build_prompt): каждый крупный фрагмент сокращается ОДИН раз, с явным
    учётом текста исходного запроса — а не "в среднем", как в инциденте с
    query-agnostic суммаризацией (см. текст урока, раздел 1). Многоуровневая
    (иерархическая) компрессия из нескольких раундов — предмет уроков
    4.3–4.4, а не этого класса.
    """

    def __init__(
        self,
        summarize_fn: SummarizeFn,
        token_counter: Callable[[str], int] = estimate_tokens,
        chunk_token_threshold: int = DEFAULT_CHUNK_TOKEN_THRESHOLD,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self._summarize_fn = summarize_fn
        self._token_counter = token_counter
        self._chunk_token_threshold = chunk_token_threshold
        self._max_context_tokens = max_context_tokens

    def build_prompt(self, query: str, chunk: ContextChunk) -> str:
        """
        Построить query-aware промпт для суммаризации ОДНОГО фрагмента.

        Ключевое отличие от query-agnostic суммаризации (см. текст урока,
        инцидент с Артёмом): промпт ЯВНО содержит текст исходного запроса
        пользователя, а не только текст фрагмента — модель, выполняющая
        суммаризацию, должна знать, какая деталь важна, а не сокращать
        документ "в среднем", без понятия о том, зачем он вообще нужен.

        TODO: вернуть строку вида:
            "Сократи следующий фрагмент, сохранив ВСЮ информацию,
            необходимую для ответа на вопрос: '{query}'.\\n\\n"
            "Источник: {chunk.source}\\n{chunk.text}"

        Точный текст шаблона не зафиксирован тестами дословно, но
        результирующая строка должна содержать И query, И chunk.text
        целиком (оба проверяются оператором `in`).
        """
        ...

    def summarize_chunk(self, query: str, chunk: ContextChunk) -> ContextChunk:
        """
        Сократить ОДИН фрагмент с помощью self._summarize_fn, сохранив
        метаданные источника и релевантности.

        TODO:
        1. prompt = self.build_prompt(query, chunk)
        2. summary_text = self._summarize_fn(prompt)
        3. new_metadata = dict(chunk.metadata); добавить в него
           "summarized": True и "original_char_length": len(chunk.text)
           (нужно для диагностики степени сжатия постфактум — тот же
           принцип, что и у RetrievalTrace.as_dict(), урок 3.5)
        4. Вернуть новый ContextChunk(text=summary_text, score=chunk.score,
           source=chunk.source, metadata=new_metadata).

        score и source НЕ меняются: суммаризация не переоценивает
        релевантность фрагмента и не маскирует его происхождение.
        """
        ...

    def compress(self, query: str, chunks: list[ContextChunk]) -> SummarizationResult:
        """
        Уместить chunks в self._max_context_tokens, применяя суммаризацию
        точечно — только там, где это действительно необходимо.

        TODO:
        1. token_counts = [self._token_counter(c.text) for c in chunks]
        2. original_total = sum(token_counts)
        3. Если original_total <= self._max_context_tokens: компрессия не
           нужна вообще — вернуть SummarizationResult(chunks=list(chunks),
           original_token_count=original_total,
           final_token_count=original_total, summarized_indices=[],
           dropped_indices=[]) БЕЗ единого вызова self._summarize_fn (не
           тратить вызов LLM и не рисковать потерей информации там, где
           фрагменты и так помещаются в бюджет).
        4. Иначе: working = list(chunks); summarized_indices = []
           Для каждого индекса i: если token_counts[i] >
           self._chunk_token_threshold — working[i] =
           self.summarize_chunk(query, working[i]);
           summarized_indices.append(i). Короткие фрагменты (не
           превышающие порог) оставить без изменений — это и есть
           "token-budget-aware dispatch", описанный в тексте урока.
        5. Пересчитать total = sum(self._token_counter(c.text) for c in
           working). Если total <= self._max_context_tokens — вернуть
           результат на этом шаге, dropped_indices=[].
        6. Иначе (после точечной суммаризации всё ещё не влезает):
           крайняя мера — truncation. Среди оставшихся чанков working
           найти чанк с наименьшим score, удалить его из working, добавить
           его исходный индекс в dropped_indices, пересчитать total.
           Повторять, пока total <= self._max_context_tokens или working
           не станет пустым. Это единственный шаг truncation в этом
           классе — применяется только тогда, когда точечной суммаризации
           оказалось недостаточно.
        7. Вернуть SummarizationResult(chunks=working,
           original_token_count=original_total, final_token_count=total,
           summarized_indices=summarized_indices,
           dropped_indices=dropped_indices).
        """
        ...
