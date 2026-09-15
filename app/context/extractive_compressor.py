from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.context.context_summarizer import estimate_tokens
from app.rag.context_builder import ContextChunk


__all__ = [
    "SentenceScorer",
    "ExtractionResult",
    "keyword_overlap_score",
    "split_sentences",
    "DEFAULT_CHUNK_TOKEN_THRESHOLD",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_POSITION_BONUS",
    "ExtractiveCompressor",
]


# Функция, которая оценивает релевантность ОДНОГО предложения запросу и
# возвращает число (обычно 0.0–1.0, но диапазон формально не ограничен —
# сравниваются только относительные величины внутри одного фрагмента). В
# production это может быть косинусное сходство эмбеддингов query и
# предложения (см. текст урока 4.2, раздел про embedding-based scoring);
# дефолт ниже — дешёвая keyword-overlap эвристика без единого обращения к
# сети или GPU, что и делает ядро ExtractiveCompressor юнит-тестируемым без
# внешних сервисов, в отличие от summarize_fn в ContextSummarizer (4.1).
SentenceScorer = Callable[[str, str], float]


# Порог "фрагмент достаточно крупный, чтобы его вообще стоило сокращать" —
# та же роль, что у DEFAULT_CHUNK_TOKEN_THRESHOLD в context_summarizer.py
# (4.1). Оба класса используют один и тот же принцип: не трогать то, что и
# так помещается в бюджет.
DEFAULT_CHUNK_TOKEN_THRESHOLD: int = 400

# Общий бюджет токенов для всего контекстного блока — см. context_summarizer.py
# (4.1) для полного объяснения, откуда берётся это число.
DEFAULT_MAX_CONTEXT_TOKENS: int = 2000

# Фиксированная надбавка к score первого предложения фрагмента —
# позиционная эвристика "первое предложение абзаца часто является
# тематическим" (topic sentence), см. текст урока, раздел 4.
DEFAULT_POSITION_BONUS: float = 0.05


def keyword_overlap_score(query: str, sentence: str) -> float:
    """
    Дефолтный SentenceScorer: доля уникальных токенов запроса, которые
    дословно встречаются среди токенов предложения (без учёта регистра).

    ВНИМАНИЕ: это НЕ семантическое сходство. Из-за словоформ ("плату" vs
    "плата") пересечение может не сработать даже при полном совпадении
    смысла — см. предупреждение в тексте урока, раздел 4. Для CORE-уровня
    этого достаточно; лемматизация и эмбеддинги — тема production-версии,
    подключаемой через параметр sentence_scorer конструктора.

    TODO:
    1. query_tokens = множество токенов re.findall(r"\\w+", query.lower())
    2. Если query_tokens пусто — вернуть 0.0
    3. sentence_tokens = множество токенов re.findall(r"\\w+", sentence.lower())
    4. Вернуть len(query_tokens & sentence_tokens) / len(query_tokens)
       (пересечение МНОЖЕСТВ, а не списков, — повторы слов не должны
       завышать итоговый score)
    """
    ...


def split_sentences(text: str) -> list[str]:
    """
    Детерминированно разбить текст фрагмента на предложения.

    TODO: разбить по регулярному выражению на границы предложений — точка,
    восклицательный или вопросительный знак, за которым следует пробел или
    конец строки (например, re.split(r"(?<=[.!?])\\s+", text.strip())).
    Отбросить пустые строки после разбиения.

    Пример:
        split_sentences("Первое предложение. Второе! Третье?")
        -> ["Первое предложение.", "Второе!", "Третье?"]

    Если во фрагменте нет ни одного разделителя предложений — вернуть
    список из одного элемента (весь текст целиком).
    """
    ...


@dataclass
class ExtractionResult:
    """
    Итог одного вызова ExtractiveCompressor.compress().

    Структура намеренно зеркалит SummarizationResult (4.1,
    app/context/context_summarizer.py) — extracted_indices играет ту же
    роль, что summarized_indices: индексы в ИСХОДНОМ списке chunks,
    переданном в compress(), а не в финальном chunks.
    """

    chunks: list[ContextChunk]
    original_token_count: int
    final_token_count: int
    extracted_indices: list[int] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)


class ExtractiveCompressor:
    """
    Второй слой компрессии контекста (модуль M4, урок 4.2) — альтернатива
    ContextSummarizer (4.1, тот же app/context/) с противоположным
    профилем риска.

    ContextSummarizer порождает НОВЫЙ текст через LLM и поэтому не может
    физически гарантировать дословное совпадение с оригиналом (см.
    инцидент урока 4.2 — комплаенс-аудит отклонил корректный по смыслу, но
    перефразированный ответ). ExtractiveCompressor никогда не порождает ни
    одного нового слова: каждое предложение результата — байт в байт то же
    предложение, что было в источнике. Плата за эту гарантию — связность:
    вырезанные из середины документа предложения, склеенные подряд, могут
    читаться разрозненно (см. текст урока, раздел 3).

    Интерфейс сознательно зеркалит ContextSummarizer: тот же
    dispatcher-паттерн в compress() (пропустить компрессию, если бюджет и
    так не превышен; сжимать точечно только крупные фрагменты; отбрасывать
    по score, если этого недостаточно). Это позволяет использовать оба
    класса через один и тот же вызывающий код и сравнивать их напрямую —
    см. exercise_1.html урока 4.2.
    """

    def __init__(
        self,
        sentence_scorer: SentenceScorer = keyword_overlap_score,
        token_counter: Callable[[str], int] = estimate_tokens,
        chunk_token_threshold: int = DEFAULT_CHUNK_TOKEN_THRESHOLD,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        position_bonus: float = DEFAULT_POSITION_BONUS,
    ) -> None:
        self._sentence_scorer = sentence_scorer
        self._token_counter = token_counter
        self._chunk_token_threshold = chunk_token_threshold
        self._max_context_tokens = max_context_tokens
        self._position_bonus = position_bonus

    def score_sentences(self, query: str, sentences: list[str]) -> list[float]:
        """
        Оценить релевантность каждого предложения запросу.

        TODO:
        1. Для каждого sentence вычислить
           base = self._sentence_scorer(query, sentence)
        2. К ПЕРВОМУ предложению списка (индекс 0) прибавить
           self._position_bonus — позиционная эвристика "первое
           предложение абзаца часто является тематическим" (см. текст
           урока, раздел 4). Остальные предложения бонус не получают.
        3. Вернуть список float той же длины и в том же порядке, что
           sentences.
        """
        ...

    def compress_chunk(self, query: str, chunk: ContextChunk) -> ContextChunk:
        """
        Сжать ОДИН фрагмент, оставив только наиболее релевантные
        предложения дословно, без единого сгенерированного слова.

        TODO:
        1. sentences = split_sentences(chunk.text). Если len(sentences) <= 1
           — вернуть chunk без изменений (сокращать нечего).
        2. scores = self.score_sentences(query, sentences)
        3. Отсортировать индексы range(len(sentences)) по scores по
           убыванию (сортировка стабильна — при равном score сохраняется
           исходный порядок).
        4. Жадно набирать индексы в этом порядке в множество kept, пока
           суммарный self._token_counter(...) уже набранных предложений
           НЕ ПРЕВЫСИТ self._chunk_token_threshold. Обязательно оставить
           хотя бы один индекс — с максимальным score, — даже если он сам
           по себе больше порога.
        5. Отсортировать kept по возрастанию (в исходном порядке
           появления в тексте, а не по score) и склеить соответствующие
           sentences через пробел — порядок важен для связности (см.
           текст урока, раздел 3), даже если полностью её не спасает.
        6. new_metadata = dict(chunk.metadata); добавить в него
           "extracted": True, "kept_sentences": len(kept),
           "original_sentences": len(sentences).
        7. Вернуть новый ContextChunk(text=<склеенный текст>,
           score=chunk.score, source=chunk.source, metadata=new_metadata).
           score и source НЕ меняются — экстракция, как и суммаризация
           (4.1), не переоценивает релевантность и не маскирует
           происхождение фрагмента.
        """
        ...

    def compress(self, query: str, chunks: list[ContextChunk]) -> ExtractionResult:
        """
        Уместить chunks в self._max_context_tokens, применяя экстракцию
        точечно — только там, где это действительно необходимо.

        Логика идентична ContextSummarizer.compress (4.1,
        app/context/context_summarizer.py) с заменой шага сокращения
        одного фрагмента на compress_chunk вместо summarize_chunk — см.
        тот файл построчно, если нужен образец.

        TODO:
        1. token_counts = [self._token_counter(c.text) for c in chunks]
        2. original_total = sum(token_counts)
        3. Если original_total <= self._max_context_tokens — вернуть
           ExtractionResult(chunks=list(chunks),
           original_token_count=original_total,
           final_token_count=original_total, extracted_indices=[],
           dropped_indices=[]) без единого вызова compress_chunk.
        4. Иначе: working = list(chunks); extracted_indices = []
           Для каждого индекса i: если token_counts[i] >
           self._chunk_token_threshold — working[i] =
           self.compress_chunk(query, working[i]);
           extracted_indices.append(i). Короткие фрагменты остаются без
           изменений.
        5. Пересчитать total = sum(self._token_counter(c.text) for c in
           working). Если total <= self._max_context_tokens — вернуть
           результат на этом шаге, dropped_indices=[].
        6. Иначе (после точечной экстракции всё ещё не влезает): крайняя
           мера — как и в 4.1, среди оставшихся чанков working найти чанк
           с наименьшим score, удалить его из working, добавить его
           ИСХОДНЫЙ индекс в dropped_indices, пересчитать total. Повторять,
           пока total <= self._max_context_tokens или working не станет
           пустым.
        7. Вернуть ExtractionResult(chunks=working,
           original_token_count=original_total, final_token_count=total,
           extracted_indices=extracted_indices,
           dropped_indices=dropped_indices).
        """
        ...
