from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.context.context_summarizer import ContextSummarizer, estimate_tokens
from app.rag.context_builder import ContextChunk


__all__ = [
    "GroupKeyFn",
    "ChunkGroup",
    "HierarchicalCompressionResult",
    "default_group_key",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "HierarchicalCompressor",
]


# Функция, определяющая ключ группы для ОДНОГО чанка. Дефолт
# (default_group_key, ниже) — детерминированная, без единого обращения к
# LLM или к сети функция, тот же принцип, что и у keyword_overlap_score в
# ExtractiveCompressor (4.2): логику группировки можно и нужно
# протестировать без внешних сервисов.
GroupKeyFn = Callable[[ContextChunk], str]

# Общий бюджет токенов для всего контекстного блока — та же роль, что у
# DEFAULT_MAX_CONTEXT_TOKENS в context_summarizer.py (4.1) и
# extractive_compressor.py (4.2).
DEFAULT_MAX_CONTEXT_TOKENS: int = 2000


def default_group_key(chunk: ContextChunk) -> str:
    """
    Дефолтный GroupKeyFn.

    TODO: вернуть chunk.metadata.get("topic_key"), если оно задано и не
    является пустой строкой; иначе — вернуть chunk.source как запасной,
    более грубый ключ.

    topic_key — это метка, которую в production проставляет апстрим
    (например, KnowledgeHierarchyResolver, урок 2.5, или отдельный
    topic-классификатор на этапе ingestion, курс 3) для явного обозначения
    "эти чанки — про одно и то же", даже если они пришли из разных
    source. Группировка ТОЛЬКО по source (без topic_key) не поймает
    кросс-источниковые дубликаты вроде инцидента этого урока —
    Confluence, Jira и Slack там были ТРЕМЯ разными source, но одной
    темой. Слишком широкий или ошибочный topic_key, наоборот, рискует
    объединить чанки, которые лишь ПОКАЗАЛИСЬ похожими (см. текст урока,
    failure mode "ошибки группировки") — то, что считать одной темой,
    всегда остаётся ответственностью вызывающего кода (апстрима), а не
    этой функции.
    """
    ...


@dataclass
class ChunkGroup:
    """
    Один результат стадии group(): общий ключ группы и все чанки,
    отнесённые к нему. Хранится отдельно от финального результата — нужен
    для диагностики постфактум (тот же принцип, что summarized_indices в
    SummarizationResult, 4.1, или RetrievalTrace, 3.5): по группам видно,
    какие именно фрагменты были признаны дублирующими друг друга.
    """

    key: str
    chunks: list[ContextChunk] = field(default_factory=list)


@dataclass
class HierarchicalCompressionResult:
    """
    Итог одного вызова HierarchicalCompressor.compress().

    chunks — финальный список чанков (по одному на выжившую после Reduce
    группу), готовый к передаче в ContextBuilder.build() (app/rag/
    context_builder.py, курс 2).

    groups — результат стадии group() ДО Map/Reduce, сохраняется для
    постфактум-диагностики (см. ChunkGroup).

    merged — True, если Reduce пришлось объединить результаты нескольких
    групп ещё раз поверх Map, потому что бюджет не был выполнен сразу
    после Map. False означает, что Map-результатов уже было достаточно, и
    Reduce не пришлось трогать данные лишний раз — то самое "не сливать
    сильнее, чем нужно" из текста урока.
    """

    chunks: list[ContextChunk]
    original_token_count: int
    final_token_count: int
    groups: list[ChunkGroup] = field(default_factory=list)
    merged: bool = False


class HierarchicalCompressor:
    """
    Модуль M4, урок 4.4: компрессия результата retrieval КАК ЕДИНОГО
    МНОЖЕСТВА, а не как списка независимых фрагментов, — в отличие от
    ContextSummarizer (4.1), ExtractiveCompressor (4.2) и
    AbstractionLadder (4.3), каждый из которых сжимает ОДИН фрагмент,
    опираясь только на его собственный score и размер.

    Схема — классический map-reduce, применённый к небольшому набору
    retrieval-фрагментов вместо терабайтов логов:

        group()         — отнести чанки, вероятно говорящие об одном и
                           том же документе/теме/факте, к общим группам.
        map_group()      — свести ОДНУ группу к ОДНОМУ обобщающему чанку,
                           переиспользуя ContextSummarizer.summarize_chunk
                           (4.1) над склеенным текстом группы, а не
                           изобретая новый способ суммаризации.
        reduce_groups()  — если после Map результаты групп всё ещё не
                           помещаются в бюджет, объединить (тем же
                           приёмом) summary нескольких групп ещё раз.

    Важное отличие от AbstractionLadder (4.3): та лестница варьирует
    ГЛУБИНУ представления ОДНОГО фрагмента (FULL/PARAGRAPH/SENTENCE/GIST),
    не меняя количество и состав фрагментов результата. Этот класс,
    наоборот, варьирует СТРУКТУРУ целого набора — сколько фрагментов
    останется и как они сгруппированы, — не решая, насколько подробно
    показать содержимое внутри одной группы (это по-прежнему работа
    summarize_chunk). Инструменты дополняют, а не заменяют друг друга.

    ВНИМАНИЕ: у схемы есть два специфичных для неё failure mode (см.
    текст урока, раздел 5) — чрезмерно агрессивный Reduce, растворяющий
    редкий, но важный сигнал меньшинства в общем пересказе большинства, и
    ошибки группировки, объединяющие чанки о разных сущностях по
    поверхностному совпадению ключевых слов. reduce_groups() и
    default_group_key() ниже спроектированы так, чтобы явно ограничивать
    оба риска, а не полагаться на удачу.
    """

    def __init__(
        self,
        summarizer: ContextSummarizer,
        group_key_fn: GroupKeyFn = default_group_key,
        token_counter: Callable[[str], int] = estimate_tokens,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self._summarizer = summarizer
        self._group_key_fn = group_key_fn
        self._token_counter = token_counter
        self._max_context_tokens = max_context_tokens

    def group(self, chunks: list[ContextChunk]) -> list[ChunkGroup]:
        """
        Разбить chunks на группы по self._group_key_fn — детерминированно,
        без единого вызова summarize_fn.

        TODO:
        1. Пройти по chunks по порядку. Для каждого chunk вычислить
           key = self._group_key_fn(chunk).
        2. Если такой key уже встречался — добавить chunk в существующую
           ChunkGroup с этим key. Если нет — создать новую ChunkGroup(key=key)
           и сразу добавить в неё chunk.
        3. Порядок групп в результирующем списке — порядок ПЕРВОГО
           появления каждого key во входном списке (а не алфавитный и не
           по score). Это важно для предсказуемости: два прогона с одним
           и тем же входом всегда должны вернуть группы в одном порядке.
        4. Вернуть list[ChunkGroup].

        Пустой входной список -> пустой список групп.
        """
        ...

    def map_group(self, query: str, group: ChunkGroup) -> ContextChunk:
        """
        Свести ОДНУ группу чанков к ОДНОМУ обобщающему чанку — "Map"-шаг.

        TODO:
        1. Если len(group.chunks) == 1 — вернуть этот единственный чанк
           БЕЗ ИЗМЕНЕНИЙ и без единого вызова summarize_fn (сводить
           нечего: группа из одного элемента не содержит дублирования).
        2. Иначе:
           a. Склеить тексты всех чанков группы в один текст с явной
              пометкой источника каждой части, например:
              "\\n\\n---\\n\\n".join(f"[{c.source}] {c.text}" for c in group.chunks)
              — так summarize_chunk (4.1) видит происхождение каждого
              фрагмента внутри склеенного блока, а не теряет его.
           b. merged_score = max(c.score for c in group.chunks) — группа
              заслуживает как минимум того внимания, что и самый
              релевантный чанк внутри неё; иначе группа с одним
              высокорелевантным и несколькими маргинальными чанками
              незаслуженно потеряет приоритет при дальнейшей обработке.
           c. Создать промежуточный ContextChunk(text=<склеенный текст>,
              score=merged_score, source=group.key, metadata={}).
           d. Вызвать self._summarizer.summarize_chunk(query, <этот
              чанк>) и вернуть результат.

        Именно на этом шаге "почти одинаковые по смыслу чанки из
        Confluence, Jira и Slack" (инцидент урока) сводятся к ОДНОМУ
        представлению группы вместо трёх независимо конкурирующих за
        бюджет копий одного факта.
        """
        ...

    def reduce_groups(
        self, query: str, group_summaries: list[ContextChunk]
    ) -> list[ContextChunk]:
        """
        "Reduce"-шаг: объединить результаты Map ещё раз, если они всё ещё
        не помещаются в self._max_context_tokens.

        TODO:
        1. total = sum(self._token_counter(c.text) for c in group_summaries)
        2. Если total <= self._max_context_tokens — вернуть
           group_summaries БЕЗ ИЗМЕНЕНИЙ и без единого дополнительного
           вызова summarize_fn. Сливать дальше, когда бюджет уже
           выполнен, — это и есть чрезмерно агрессивный Reduce из текста
           урока: он ничего не выигрывает по месту, но рискует растворить
           редкий сигнал меньшинства в общем пересказе.
        3. Иначе, пока total > self._max_context_tokens и
           len(group_summaries) > 1:
           a. Найти ДВА чанка с наименьшей суммой score среди оставшихся
              — наименее важные по совокупности сливаются в первую
              очередь, а не самые важные.
           b. Слить их тем же приёмом, что и в map_group (склеить тексты
              с пометкой source, score = max() двух исходных, source =
              "{source1}+{source2}", один вызов summarize_chunk).
           c. Заменить эти два чанка одним результирующим в списке,
              пересчитать total.
        4. Вернуть финальный список чанков.

        Каждое слияние на этом шаге — последний шанс не потерять
        информацию из чанка с низким score, прежде чем он исчезнет в
        общем пересказе, — поэтому цикл обязан останавливаться, как
        только бюджет выполнен, а не продолжать сокращать "про запас".
        """
        ...

    def compress(
        self, query: str, chunks: list[ContextChunk]
    ) -> HierarchicalCompressionResult:
        """
        Главный метод: group -> map каждую группу -> reduce, если нужно.

        TODO:
        1. original_total = sum(self._token_counter(c.text) for c in chunks)
        2. groups = self.group(chunks)
        3. mapped = [self.map_group(query, g) for g in groups]
        4. reduced = self.reduce_groups(query, mapped)
        5. final_total = sum(self._token_counter(c.text) for c in reduced)
        6. merged = len(reduced) < len(mapped)
        7. Вернуть HierarchicalCompressionResult(chunks=reduced,
           original_token_count=original_total,
           final_token_count=final_total, groups=groups, merged=merged)
        """
        ...
