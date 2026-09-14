from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


__all__ = [
    "StopReason",
    "HopRecord",
    "MultiHopResult",
    "RetrieveFn",
    "ExtractFn",
    "ReformulateFn",
    "AnswerCheckFn",
    "MultiHopRetriever",
]


class StopReason(Enum):
    """
    Основание, по которому цикл multi-hop retrieval был остановлен.
    Ровно одно из трёх значений — не более и не менее.
    """

    ANSWER_FOUND = "answer_found"  # извлечённый факт закрывает вопрос целиком
    MAX_HOPS_REACHED = "max_hops_reached"  # бюджет hop'ов исчерпан
    NO_NEW_INFORMATION = "no_new_information"  # hop N не отличается от hop N-1


@dataclass
class HopRecord:
    """
    Один шаг multi-hop цепочки: какой запрос был отправлен на этом hop,
    что вернул retrieval и какая сущность/факт была из этого извлечена.

    extracted_entity может быть None — extract_fn не обязан находить
    сущность на каждом hop'е (например, если retrieved_text не содержит
    ничего полезного для продолжения цепочки).
    """

    hop_index: int
    query: str
    retrieved_text: str
    extracted_entity: Optional[str]


@dataclass
class MultiHopResult:
    """
    Полный результат работы MultiHopRetriever.run() — вся цепочка hop'ов
    плюс причина, по которой цикл остановился.
    """

    hops: list[HopRecord] = field(default_factory=list)
    stop_reason: StopReason = StopReason.MAX_HOPS_REACHED
    final_entity: Optional[str] = None


# Колбэки, которые MultiHopRetriever вызывает, но не реализует сам.
# Это осознанная граница ответственности: класс не знает ничего о
# конкретной векторной базе, embedding-модели или способе извлечения
# сущностей (regex, NER-модель, LLM со structured output — курс 6).

# query -> текст найденного релевантного фрагмента (топ-результат retrieval)
RetrieveFn = Callable[[str], str]

# (текущий запрос hop'а, найденный текст) -> извлечённая сущность/факт
# или None, если извлечь нечего
ExtractFn = Callable[[str, str], Optional[str]]

# (исходный запрос пользователя, извлечённая сущность) -> текст запроса
# для следующего hop'а
ReformulateFn = Callable[[str, str], str]

# (исходный запрос пользователя, найденный на текущем hop текст,
# извлечённая сущность) -> True, если найденного достаточно, чтобы
# полностью закрыть исходную информационную потребность
AnswerCheckFn = Callable[[str, str, Optional[str]], bool]


class MultiHopRetriever:
    """
    Оркестратор multi-hop retrieval: последовательный цикл
    retrieve -> extract -> reformulate -> retrieve, в котором каждый
    следующий запрос формулируется только после того, как получен и
    разобран результат предыдущего hop'а.

    Независим от QueryDecomposer (урок 3.1, app/context/query_decomposer.py)
    и ничего о нём не знает: декомпозиция режет ОДИН исходный запрос на
    НЕЗАВИСИМЫЕ подзапросы (retrieval которых можно выполнить параллельно),
    тогда как этот класс решает противоположную по контракту задачу —
    ПОСЛЕДОВАТЕЛЬНУЮ цепочку запросов, где текст запроса N+1 физически
    не может быть написан заранее, потому что зависит от факта,
    извлечённого на hop'е N. Оба класса составимы снаружи: планировщик
    retrieval может сначала декомпозировать запрос на независимые ветки,
    а затем прогнать через MultiHopRetriever только ту ветку, где
    обнаружена такая зависимость.

    Сам класс не выполняет ни retrieval, ни извлечение сущностей — это
    ответственность внешних колбэков (retrieve_fn, extract_fn,
    reformulate_fn, is_answer_fn), переданных в конструктор. Единственная
    ответственность MultiHopRetriever — оркестрация цикла и трёх
    стоп-условий (StopReason).
    """

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        extract_fn: ExtractFn,
        reformulate_fn: ReformulateFn,
        is_answer_fn: AnswerCheckFn,
        max_hops: int = 4,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._extract_fn = extract_fn
        self._reformulate_fn = reformulate_fn
        self._is_answer_fn = is_answer_fn
        self._max_hops = max_hops

    def run(self, query: str) -> MultiHopResult:
        """
        Точка входа: выполнить цепочку hop'ов начиная с исходного query
        и вернуть полный MultiHopResult.

        TODO:
        1. result = MultiHopResult(hops=[])
        2. current_query = query; previous_record: HopRecord | None = None
        3. Цикл по hop_index от 0 (пока не сработает стоп-условие):
           a. retrieved = self._retrieve_fn(current_query)
           b. entity = self._extract_fn(current_query, retrieved)
           c. record = HopRecord(hop_index=hop_index, query=current_query,
              retrieved_text=retrieved, extracted_entity=entity)
           d. result.hops.append(record)
           e. Проверить стоп-условия СТРОГО В ЭТОМ ПОРЯДКЕ:
              - Если self._is_answer_fn(query, retrieved, entity) is True:
                result.stop_reason = StopReason.ANSWER_FOUND
                result.final_entity = entity
                вернуть result
              - Иначе если self._has_new_information(previous_record,
                record) is False:
                result.stop_reason = StopReason.NO_NEW_INFORMATION
                result.final_entity = entity
                вернуть result
              - Иначе если hop_index + 1 >= self._max_hops:
                result.stop_reason = StopReason.MAX_HOPS_REACHED
                result.final_entity = entity
                вернуть result
           f. Иначе (ни одно условие не сработало):
              current_query = self._reformulate_fn(query, entity)
              previous_record = record
              hop_index += 1, следующая итерация
        """
        ...

    def _has_new_information(
        self, previous: Optional[HopRecord], current: HopRecord
    ) -> bool:
        """
        Проверить, продвинулся ли текущий hop дальше предыдущего, или
        цепочка "вращается на месте" (стоп-условие NO_NEW_INFORMATION).

        TODO:
        1. Если previous is None — это первый hop, прогресс есть по
           определению, вернуть True.
        2. Если current.extracted_entity == previous.extracted_entity
           (и оба не None) — прогресса нет, вернуть False.
        3. Если current.retrieved_text == previous.retrieved_text —
           retrieval второй раз подряд вернул тот же фрагмент, прогресса
           нет, вернуть False.
        4. Иначе вернуть True.
        """
        ...
