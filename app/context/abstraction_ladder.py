from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

from app.context.context_summarizer import ContextSummarizer, estimate_tokens
from app.rag.context_builder import ContextChunk


__all__ = [
    "LadderLevel",
    "LevelSpec",
    "LadderResult",
    "DEFAULT_LEVELS",
    "AbstractionLadder",
]


class LadderLevel(IntEnum):
    """
    Уровни лестницы абстракции для ОДНОГО фрагмента — от самого
    подробного к самому грубому. Значения IntEnum используются только для
    удобного сравнения порядка ("FULL детальнее, чем PARAGRAPH"), а не как
    магические числа где-либо ещё в этом модуле.

    Это НЕ то же самое, что иерархическая компрессия урока 4.4: там
    несколько РАЗНЫХ фрагментов/документов складываются в одно дерево.
    Здесь речь об одном и том же фрагменте, представленном на разных
    уровнях детализации, — см. текст урока, раздел 3.
    """

    FULL = 0
    PARAGRAPH = 1
    SENTENCE = 2
    GIST = 3


@dataclass(frozen=True)
class LevelSpec:
    """
    Описание одного уровня лестницы.

    max_tokens — потолок токенов, который этот уровень занимает у ОДНОГО
    фрагмента; None только для FULL — у него нет собственного потолка,
    вместо него используется фактический размер оригинального текста
    (см. AbstractionLadder.select_level: FULL "стоит" ровно столько,
    сколько весит сам фрагмент, а не какое-то условное число).

    min_score — минимальный relevance score (тот же score, что вычислили
    SourceSelector/KnowledgeHierarchyResolver в M3), при котором фрагмент
    вообще ЗАСЛУЖИВАЕТ такой уровень детализации.

    Оба условия проверяются НЕЗАВИСИМО: min_score отвечает на вопрос
    "насколько подробно этот фрагмент того стоит", max_tokens (или размер
    оригинала для FULL) — на вопрос "можем ли мы вообще себе это позволить
    при текущем бюджете". Итоговый уровень — пересечение обоих условий, а
    не только одного из них (см. текст урока, раздел 2, инцидент про
    плоское сжатие).
    """

    level: LadderLevel
    max_tokens: Optional[int]
    min_score: float


# Дефолтная лестница из четырёх уровней. Пороги подобраны так, чтобы:
# - фрагмент с высоким score (>= 0.75) получает право на полный текст,
#   ЕСЛИ фактический размер этого текста умещается в выделенный ему
#   бюджет — высокий score сам по себе не отменяет ограничение по месту;
# - фрагмент с низким score (< 0.2) никогда не получит больше, чем
#   GIST — одно-два предложения, — сколько бы свободного бюджета ни было,
#   потому что тратить много токенов на маргинально релевантный контент
#   не оправдано, даже когда есть место (см. инцидент этого урока: именно
#   так маргинальный Slack-тред отъедал непропорциональную долю бюджета).
#
# Уровни ОБЯЗАНЫ идти от самого детального (FULL) к самому грубому
# (GIST) — весь класс ниже полагается на этот порядок при переборе.
DEFAULT_LEVELS: tuple[LevelSpec, ...] = (
    LevelSpec(LadderLevel.FULL, max_tokens=None, min_score=0.75),
    LevelSpec(LadderLevel.PARAGRAPH, max_tokens=150, min_score=0.45),
    LevelSpec(LadderLevel.SENTENCE, max_tokens=40, min_score=0.2),
    LevelSpec(LadderLevel.GIST, max_tokens=12, min_score=0.0),
)


@dataclass
class LadderResult:
    """
    Итог одного вызова AbstractionLadder.pick() для ОДНОГО фрагмента.

    level — уровень, который в итоге был выбран (для диагностики
    постфактум, тот же принцип, что и summarized_indices в
    SummarizationResult, 4.1). score и token_budget сохраняются рядом с
    результатом, чтобы при разборе инцидента было видно, ПОЧЕМУ был выбран
    именно этот уровень, а не другой.
    """

    chunk: ContextChunk
    level: LadderLevel
    score: float
    token_budget: int


class AbstractionLadder:
    """
    Модуль M4, урок 4.3: вместо ОДНОЙ фиксированной степени сжатия для
    всех превышающих порог фрагментов (как в ContextSummarizer, 4.1, и
    ExtractiveCompressor, 4.2) — несколько заранее определённых уровней
    абстракции ОДНОГО И ТОГО ЖЕ фрагмента, и явный выбор нужного уровня по
    ДВУМ независимым сигналам: relevance score фрагмента (M3) и доступный
    ему бюджет токенов.

    AbstractionLadder не заменяет ContextSummarizer — он его КОМПОЗИРУЕТ:
    каждый уровень абстракции, кроме FULL, строится вызовом
    self._summarizer.summarize_chunk(query, ...) — тем же методом,
    реализованным в уроке 4.1. Разница в том, ЧТО именно передаётся в этот
    метод: не оригинальный текст фрагмента заново на каждом уровне, а
    текст ПРЕДЫДУЩЕГО, уже более грубого уровня — каскадная лестница, а не
    независимые попытки сжать один и тот же оригинал по-разному (см. текст
    урока, раздел 3).

    ВАЖНО: у каскадной схемы есть неочевидная цена. Получить фрагмент на
    уровне GIST "с нуля" стоит len(levels) - 1 вызовов summarize_fn (нужно
    последовательно пройти через PARAGRAPH и SENTENCE), а не один дешёвый
    вызов, как можно было бы наивно предположить, глядя на то, что GIST —
    самый короткий результат. Именно поэтому render_level и pick
    принимают необязательный precomputed-кэш — целиком посчитанный заранее
    (например, на этапе ingestion, курс 3) результат generate_full_ladder,
    который превращает on-demand пересчёт в бесплатный lookup по словарю.
    Это и есть компромисс "посчитать один раз заранее" против "считать
    каждый раз заново", разобранный в тексте урока, раздел 4.
    """

    def __init__(
        self,
        summarizer: ContextSummarizer,
        levels: tuple[LevelSpec, ...] = DEFAULT_LEVELS,
        token_counter: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self._summarizer = summarizer
        self._levels = levels
        self._token_counter = token_counter

    def select_level(
        self, score: float, token_budget: int, full_token_count: int
    ) -> LadderLevel:
        """
        Выбрать уровень лестницы, полагаясь ТОЛЬКО на score фрагмента,
        доступный ему token_budget и фактический размер полного текста
        фрагмента (full_token_count) — без единого вызова summarize_fn.
        Это и есть логика, которую можно и нужно полностью протестировать
        без LLM (см. exercise_1.html).

        TODO:
        1. Перебрать self._levels В ЗАДАННОМ ПОРЯДКЕ (от FULL к GIST).
        2. Для каждого level_spec вычислить эффективный потолок токенов:
           cap = full_token_count, если level_spec.max_tokens is None
           (это касается только FULL — его "стоимость" равна фактическому
           размеру оригинала, а не произвольному числу), иначе
           cap = level_spec.max_tokens.
        3. Проверить ОБА условия:
           - eligible_by_score = score >= level_spec.min_score
           - eligible_by_budget = cap <= token_budget
           Если оба условия истинны — вернуть level_spec.level. Это первый
           (то есть самый детальный из подходящих) уровень, прошедший обе
           проверки.
        4. Если ни один уровень так и не подошёл (даже самый грубый
           уровень в self._levels не проходит по бюджету) — вернуть level
           последнего элемента self._levels как крайний случай: лучшее,
           что можно предложить этому фрагменту. Решение "выделенного
           бюджета этому чанку вообще недостаточно" — забота вызывающего
           кода (например, диспетчера компрессии), а не этой функции.
        """
        ...

    def generate_full_ladder(
        self, query: str, chunk: ContextChunk
    ) -> dict[LadderLevel, ContextChunk]:
        """
        Заранее построить ВСЕ уровни лестницы каскадно — сценарий
        "посчитать один раз на этапе ingestion" (курс 3), результат
        которого можно закэшировать и передать позже в render_level/pick
        как precomputed, чтобы на этапе запроса не делать ни одного
        вызова summarize_fn.

        TODO:
        1. ladder: dict[LadderLevel, ContextChunk] = {LadderLevel.FULL: chunk}
        2. current = chunk
        3. Для каждого level_spec в self._levels[1:] (все уровни, кроме
           FULL, по порядку от детального к грубому):
           current = self._summarizer.summarize_chunk(query, current)
           ladder[level_spec.level] = current
        4. Вернуть ladder.

        Обратите внимание: этот метод ВСЕГДА делает len(self._levels) - 1
        вызовов summarize_fn, независимо от того, понадобится ли в итоге
        хоть один уровень грубее PARAGRAPH, — это и есть цена
        предвычисления, которую стоит платить только тогда, когда лестница
        действительно будет переиспользована для многих будущих запросов
        к одному и тому же документу (см. текст урока, раздел 4).
        """
        ...

    def render_level(
        self,
        query: str,
        chunk: ContextChunk,
        level: LadderLevel,
        precomputed: Optional[dict[LadderLevel, ContextChunk]] = None,
    ) -> ContextChunk:
        """
        Получить фрагмент РОВНО на уровне level — по возможности
        переиспользуя precomputed (результат generate_full_ladder,
        посчитанный заранее), и только при его отсутствии считая каскад
        с нуля прямо сейчас (on-demand, сценарий "считаем в момент
        запроса", см. текст урока, раздел 4).

        TODO:
        1. Если level is LadderLevel.FULL — вернуть chunk без изменений.
           Уровень FULL никогда не требует вызова summarize_fn, посчитан
           он заранее или нет.
        2. Если precomputed передан и level в нём уже есть — вернуть
           precomputed[level] без единого вызова summarize_fn (кэш-хит,
           бесплатный lookup).
        3. Иначе — посчитать каскад с нуля: current = chunk; пройти по
           self._levels[1:] по порядку (от детального к грубому),
           каждый раз current = self._summarizer.summarize_chunk(query,
           current), пока не будет обработан level_spec с
           level_spec.level == level. Вернуть current на этом шаге, не
           продолжая каскад дальше запрошенного уровня.
        """
        ...

    def pick(
        self,
        query: str,
        chunk: ContextChunk,
        token_budget: int,
        precomputed: Optional[dict[LadderLevel, ContextChunk]] = None,
    ) -> LadderResult:
        """
        Главный метод: выбрать уровень для chunk по его score, доступному
        token_budget и фактическому размеру chunk.text, получить фрагмент
        на этом уровне и вернуть результат вместе с диагностикой.

        TODO:
        1. full_token_count = self._token_counter(chunk.text)
        2. level = self.select_level(chunk.score, token_budget,
           full_token_count)
        3. rendered = self.render_level(query, chunk, level,
           precomputed=precomputed)
        4. Вернуть LadderResult(chunk=rendered, level=level,
           score=chunk.score, token_budget=token_budget)
        """
        ...
