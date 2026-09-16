from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.context.context_summarizer import estimate_tokens
from app.rag.context_builder import ContextChunk


__all__ = [
    "CompressionStrategyFn",
    "FactMatchFn",
    "CompressionTestCase",
    "CaseEvalResult",
    "CompressionEvalReport",
    "default_fact_match",
    "select_best_strategy",
    "CompressionEvalHarness",
]


# Единый интерфейс поверх ЛЮБОЙ из четырёх стратегий модуля M4
# (ContextSummarizer.compress, ExtractiveCompressor.compress,
# AbstractionLadder..., HierarchicalCompressor.compress): принимает
# (query, chunks) и возвращает УЖЕ сжатый список ContextChunk. Каждая из
# четырёх стратегий возвращает свой собственный dataclass результата
# (SummarizationResult, ExtractionResult, ...) с полем .chunks — вызывающий
# код оборачивает конкретную стратегию в CompressionStrategyFn лямбдой
# вида `lambda q, c: summarizer.compress(q, c).chunks`. Сам
# CompressionEvalHarness не знает и не должен знать, какая именно стратегия
# внутри, — тот же принцип dependency injection, что и summarize_fn в
# ContextSummarizer (4.1) или strategy_fn в RetrievalPlanner (3.4). Для
# unit-тестов сюда подставляется простая детерминированная функция без
# единого обращения к LLM.
CompressionStrategyFn = Callable[[str, list[ContextChunk]], list[ContextChunk]]

# Проверяет, "пережил" ли ground_truth_fact компрессию — присутствует ли он
# (дословно или по значимой доле слов) в итоговом сжатом тексте.
FactMatchFn = Callable[[str, str], bool]


def default_fact_match(ground_truth_fact: str, compressed_text: str, threshold: float = 0.6) -> bool:
    """
    Дефолтный FactMatchFn: доля уникальных слов ground_truth_fact,
    дословно встречающихся в compressed_text (без учёта регистра).

    ВНИМАНИЕ: как и keyword_overlap_score в ExtractiveCompressor (4.2), это
    НЕ семантическое сравнение. После суммаризации через LLM факт мог быть
    перефразирован ("неустойка снижается до 1%" -> "льготная ставка для
    спецклиентов, всего один процент") и не пройти это дословное сравнение,
    даже если по смыслу он полностью сохранён. Для production-версии
    harness'а сюда подставляется LLM-as-judge функция того же типа
    FactMatchFn — интерфейс CompressionEvalHarness не изменится ни на
    строчку, ровно как и summarize_fn у ContextSummarizer.

    TODO:
    1. Если ground_truth_fact.strip() пуст — вернуть True (нечего
       проверять, факта не задано).
    2. fact_words = множество уникальных слов ground_truth_fact.lower().split()
    3. text_words = множество уникальных слов compressed_text.lower().split()
    4. Если fact_words пусто (только пробелы) — вернуть True.
    5. overlap = len(fact_words & text_words) / len(fact_words)
    6. Вернуть overlap >= threshold.
    """
    ...


@dataclass
class CompressionTestCase:
    """
    Один тест-кейс eval harness'а: конкретный запрос, факт, который ОБЯЗАН
    остаться доступным в ответе, и набор найденных retrieval'ом фрагментов
    (тот же вход, что получают ContextSummarizer.compress /
    ExtractiveCompressor.compress / HierarchicalCompressor.compress).

    ground_truth_fact — не весь ожидаемый ответ, а минимальная опорная
    фраза, по которой можно детерминированно проверить, "выжил" ли факт
    (см. default_fact_match). Для инцидента этого урока это была бы фраза
    вроде "спецклиент неустойка 1%" — а НЕ вся структура полного ответа.
    """

    case_id: str
    query: str
    ground_truth_fact: str
    chunks: list[ContextChunk]


@dataclass
class CaseEvalResult:
    """
    Итог прогона ОДНОГО CompressionTestCase через ОДНУ стратегию.

    matched_chunk_sources — source тех чанков ИТОГОВОГО (сжатого) набора,
    которые сами по себе содержат факт (по тому же FactMatchFn) — тот же
    принцип диагностики постфактум, что summarized_indices у
    SummarizationResult (4.1) или ChunkGroup у HierarchicalCompressionResult
    (4.4): позволяет не просто узнать "факт выжил/не выжил", а увидеть, В
    КАКОМ ИМЕННО чанке он уцелел.
    """

    case_id: str
    fact_survived: bool
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float  # compressed_token_count / original_token_count
    matched_chunk_sources: list[str] = field(default_factory=list)


@dataclass
class CompressionEvalReport:
    """
    Агрегат по всем CaseEvalResult одного прогона ОДНОЙ стратегии
    (возможно, с конкретным значением token-бюджета — strategy_name
    отвечает и за то, и за другое, например "hierarchical@1000").

    recall — доля кейсов, где fact_survived == True. Это НЕ recall в
    классическом ML-смысле (доля релевантных документов), а метрика этого
    модуля: доля тест-кейсов, для которых опорный факт остался доступным
    после сжатия.

    mean_compression_ratio — среднее (невзвешенное) compression_ratio по
    всем кейсам; чем меньше, тем агрессивнее было сжатие в среднем.
    """

    strategy_name: str
    results: list[CaseEvalResult]
    recall: float
    mean_compression_ratio: float


class CompressionEvalHarness:
    """
    Модуль M4, урок 4.5: измерительный слой поверх ЛЮБОЙ из четырёх
    стратегий компрессии (4.1–4.4), закрывающий модуль "Компрессия
    контекста" вопросом, который до этого урока не задавался ни разу —
    "а как мы вообще УЗНАЁМ, что конкретная степень сжатия ещё безопасна?"

    Инцидент этого урока («Печора Телеком») — не баг одного конкретного
    компрессора, а системная ошибка процесса: команда наращивала
    агрессивность сжатия НА ГЛАЗ, чтобы уместить больше источников на
    запрос, полагаясь на выборочный ручной просмотр ответов ("выглядит
    нормально"). CompressionEvalHarness — способ заменить "выглядит
    нормально" на измеримое число: набор пар (запрос, опорный факт),
    прогнанный через стратегию, либо подтверждает, что факт пережил
    сжатие, либо нет — для КАЖДОГО кейса, а не для случайно выбранных
    вручную примеров.

    Как и ContextSummarizer (4.1) со своим summarize_fn, harness работает
    с ЛЮБОЙ функцией сигнатуры CompressionStrategyFn через dependency
    injection — включая простую детерминированную функцию без единого
    обращения к LLM, что делает ЭТОТ класс полностью unit-тестируемым.
    """

    def __init__(
        self,
        strategy_fn: CompressionStrategyFn,
        fact_match_fn: FactMatchFn = default_fact_match,
        token_counter: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self._strategy_fn = strategy_fn
        self._fact_match_fn = fact_match_fn
        self._token_counter = token_counter

    def run_case(self, case: CompressionTestCase) -> CaseEvalResult:
        """
        Прогнать ОДИН тест-кейс через self._strategy_fn и измерить,
        пережил ли compression_eval.CompressionTestCase.ground_truth_fact
        сжатие.

        TODO:
        1. original_tokens = sum(self._token_counter(c.text) for c in case.chunks)
        2. compressed_chunks = self._strategy_fn(case.query, case.chunks)
        3. compressed_text = "\\n".join(c.text for c in compressed_chunks)
        4. compressed_tokens = sum(self._token_counter(c.text) for c in compressed_chunks)
        5. fact_survived = self._fact_match_fn(case.ground_truth_fact, compressed_text)
        6. matched_chunk_sources = [c.source for c in compressed_chunks
           if self._fact_match_fn(case.ground_truth_fact, c.text)]
        7. compression_ratio: если original_tokens == 0 — вернуть 1.0 (нет
           входных токенов — считать, что "сжатие" не производилось);
           иначе compressed_tokens / original_tokens.
        8. Вернуть CaseEvalResult(case_id=case.case_id,
           fact_survived=fact_survived, original_token_count=original_tokens,
           compressed_token_count=compressed_tokens,
           compression_ratio=compression_ratio,
           matched_chunk_sources=matched_chunk_sources).
        """
        ...

    def run(self, cases: list[CompressionTestCase], strategy_name: str = "") -> CompressionEvalReport:
        """
        Прогнать ВЕСЬ набор тест-кейсов через self.run_case и агрегировать
        результат в CompressionEvalReport.

        TODO:
        1. Если cases пуст — вызвать ValueError("cases не может быть
           пустым: eval harness измеряет деградацию НА НАБОРЕ кейсов, а не
           в вакууме").
        2. results = [self.run_case(case) for case in cases]
        3. recall = доля results с fact_survived == True
           (sum(r.fact_survived for r in results) / len(results))
        4. mean_compression_ratio = среднее compression_ratio по results
        5. Вернуть CompressionEvalReport(strategy_name=strategy_name,
           results=results, recall=recall,
           mean_compression_ratio=mean_compression_ratio).
        """
        ...


def select_best_strategy(
    reports: list[CompressionEvalReport], min_recall: float = 1.0
) -> Optional[CompressionEvalReport]:
    """
    Практический слой принятия решения поверх нескольких CompressionEvalReport
    (например, одна и та же стратегия при разных max_context_tokens, или
    несколько разных стратегий из 4.1–4.4 на ОДНОМ наборе кейсов) — замена
    интуиции "наверное, этого сжатия достаточно" на явный, измеримый порог.

    min_recall — минимально допустимый recall для КОНКРЕТНОЙ ситуации
    (см. текст урока, раздел про decision framework): для сценариев с
    редкими, но критичными фактами (тарифные льготы, юридические условия —
    инцидент 4.4 и этого урока) min_recall следует держать близким к 1.0;
    для менее критичных сценариев (черновой пересказ длинного треда) можно
    сознательно допустить min_recall ниже.

    TODO:
    1. candidates = [r for r in reports if r.recall >= min_recall]
    2. Если candidates пуст — вернуть None (сигнал: ни одна из
       предложенных стратегий/бюджетов не безопасна для этого набора
       кейсов при заданном min_recall — нужно либо снизить агрессивность
       сжатия, либо расширить бюджет, а НЕ поднимать порог задним числом).
    3. Иначе вернуть элемент candidates с наименьшим
       mean_compression_ratio (максимальное сжатие среди безопасных по
       recall вариантов). При равенстве mean_compression_ratio у
       нескольких кандидатов вернуть ПЕРВЫЙ из них по порядку в списке
       reports (детерминированность, а не произвольный выбор).
    """
    ...
