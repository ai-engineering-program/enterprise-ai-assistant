from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


__all__ = [
    "DecompositionPattern",
    "SubQuery",
    "DecompositionResult",
    "QueryDecomposer",
]


class DecompositionPattern(Enum):
    """
    Тип склейки, которая объединяет несколько самостоятельных
    информационных потребностей в одном пользовательском запросе.
    """

    ATOMIC = "atomic"  # одна потребность — декомпозиция не нужна
    CONJUNCTIVE = "conjunctive"  # "X и Y" — два независимых вопроса подряд
    COMPARISON = "comparison"  # "что лучше X или Y" / "чем отличается X от Y"
    IMPLICIT_SUBQUESTION = "implicit_subquestion"  # явный вопрос ссылается на
    # неявное базовое состояние, которое нужно выяснить отдельно


# Маркеры конъюнктивной склейки. Намеренно НЕ голое слово " и " — оно
# слишком частое в русском языке и почти всегда даёт ложные срабатывания
# (сравните "тариф и подключение" — не два вопроса, а одна именная группа).
# Вместо этого маркер — это связка "и" + вопросительное слово, которая
# перезапускает вопрос внутри той же фразы, как в инциденте этого урока:
# "какой тариф у клиента... И ПОЧЕМУ списание больше обычного".
_DEFAULT_CONJUNCTIVE_MARKERS: tuple[str, ...] = (
    " и почему",
    " и какой",
    " и какая",
    " и какое",
    " и что",
    " и как",
    " и когда",
    ", а также",
    ", а еще",
)

# Маркеры сравнения: запрос просит сопоставить две сущности, а не узнать
# факт об одной.
_DEFAULT_COMPARISON_MARKERS: tuple[str, ...] = (
    "чем отличается",
    "в чём разница",
    "в чем разница",
    "что лучше",
    "по сравнению с",
)

# Маркеры неявного подвопроса: явный вопрос ссылается на некое базовое
# состояние ("обычно", "как всегда"), которое сначала нужно узнать
# отдельным запросом, прежде чем можно оценить "необычность" явной части.
_DEFAULT_IMPLICIT_MARKERS: tuple[str, ...] = (
    "больше обычного",
    "меньше обычного",
    "не как обычно",
    "не так, как всегда",
    "иначе, чем всегда",
    "необычно",
)


@dataclass
class SubQuery:
    """
    Один независимо извлекаемый подзапрос — результат декомпозиции.

    order задаёт порядок выполнения retrieval (важно для воспроизводимости
    логов и для урока 3.2, где часть подзапросов может стать входом для
    следующего hop), а НЕ скрытую зависимость между подзапросами: на уровне
    этого урока каждый подзапрос сам по себе — самостоятельный, валидный
    embedding-запрос, который можно выполнить независимо от остальных.
    """

    text: str
    order: int
    pattern: DecompositionPattern


@dataclass
class DecompositionResult:
    """Полный результат декомпозиции одного пользовательского запроса."""

    original_query: str
    pattern: DecompositionPattern
    sub_queries: list[SubQuery] = field(default_factory=list)


class QueryDecomposer:
    """
    Разбивает один пользовательский запрос, объединяющий несколько
    самостоятельных информационных потребностей, на упорядоченный список
    независимо извлекаемых подзапросов — ДО того, как retrieval-слой
    выполнит хоть один embedding-поиск.

    Это rule-based решение (тот же принцип, что и у TruthAxisRouter,
    урок 2.4) — набор маркеров вместо LLM-вызова. Осознанный компромисс
    курса: детерминированность, нулевая стоимость и объяснимость важнее
    полноты покрытия всех возможных формулировок. В производственной
    системе тот же интерфейс decompose() может быть реализован через
    вызов LLM со structured output (курс 6) для более высокого recall на
    нестандартных формулировках — контракт (DecompositionResult) при этом
    не меняется, меняется только реализация внутри.

    Декомпозиция НЕ выполняет retrieval и НЕ решает, из какого источника
    брать каждый подзапрос — это задачи следующих ступеней конвейера
    (уроки 3.2 "Multi-hop retrieval" и 3.3 "Логика выбора источника").
    Единственная ответственность этого класса — превратить один
    "смешанный" запрос в список простых подзапросов, каждый из которых
    сам по себе даёт осмысленный embedding.
    """

    def __init__(
        self,
        conjunctive_markers: Iterable[str] | None = None,
        comparison_markers: Iterable[str] | None = None,
        implicit_markers: Iterable[str] | None = None,
    ) -> None:
        self._conjunctive_markers: tuple[str, ...] = tuple(
            conjunctive_markers
            if conjunctive_markers is not None
            else _DEFAULT_CONJUNCTIVE_MARKERS
        )
        self._comparison_markers: tuple[str, ...] = tuple(
            comparison_markers
            if comparison_markers is not None
            else _DEFAULT_COMPARISON_MARKERS
        )
        self._implicit_markers: tuple[str, ...] = tuple(
            implicit_markers if implicit_markers is not None else _DEFAULT_IMPLICIT_MARKERS
        )

    def classify_pattern(self, query: str) -> DecompositionPattern:
        """
        Определить тип склейки запроса. Порядок проверки принципиален и
        закреплён этим методом, а не случаен: сравнение и конъюнкция —
        более сильные, однозначные синтаксические сигналы, чем неявный
        подвопрос, поэтому проверяются раньше. Запрос может формально
        содержать маркеры нескольких типов одновременно (например,
        "и почему" конъюнкции и "больше обычного" неявного подвопроса в
        одной фразе) — побеждает тип с более высоким приоритетом.

        TODO:
        1. lower = query.lower() — сама query не изменяется, поиск
           маркеров регистронезависим, но текст подзапросов далее всегда
           берётся из ОРИГИНАЛЬНОГО query.
        2. Если хотя бы один marker из self._comparison_markers
           встречается в lower — вернуть DecompositionPattern.COMPARISON.
        3. Иначе если хотя бы один marker из self._conjunctive_markers
           встречается в lower — вернуть DecompositionPattern.CONJUNCTIVE.
        4. Иначе если хотя бы один marker из self._implicit_markers
           встречается в lower — вернуть
           DecompositionPattern.IMPLICIT_SUBQUESTION.
        5. Иначе — вернуть DecompositionPattern.ATOMIC (декомпозиция не
           нужна, запрос уже несёт одну информационную потребность).
        """
        ...

    def split_conjunctive(self, query: str) -> list[str]:
        """
        Разбить конъюнктивный запрос на два подзапроса по САМОМУ РАННЕМУ
        (по позиции в тексте) из совпавших маркеров self._conjunctive_markers.

        TODO:
        1. lower = query.lower(). Для каждого marker из
           self._conjunctive_markers найти lower.find(marker); собрать
           только найденные вхождения (find(...) != -1).
        2. Если вхождений нет — вернуть [query] одним элементом (не
           должно происходить, если classify_pattern уже вернул
           CONJUNCTIVE, но метод обязан быть безопасным сам по себе).
        3. Выбрать marker с МИНИМАЛЬНОЙ позицией idx среди найденных.
        4. connector = " и " (3 символа), если matched marker начинается
           с " и ", иначе connector = сам matched marker целиком (для
           маркеров вида ", а также" / ", а еще" отдельного
           вопросительного слова после связки нет — весь маркер и есть
           связка).
        5. first = query[:idx].strip()
        6. second = query[idx + len(connector):].strip()
        7. Вернуть [first, second].
        """
        ...

    def split_comparison(self, query: str) -> list[str]:
        """
        Разбить сравнительный запрос на два подзапроса — по одному на
        каждую сравниваемую сущность, используя маркер " или " как
        разделитель.

        Упрощение, осознанно принятое для CORE-уровня этого урока: если
        в запросе нет буквального " или " (например, вопрос сформулирован
        как "чем отличается X от Y"), метод не пытается разобрать
        грамматику — такие формулировки в этом уроке не разбиваются
        (см. границы применимости в тексте урока).

        TODO:
        1. lower = query.lower(); idx = lower.find(" или ").
        2. Если idx == -1 — вернуть [query] одним элементом.
        3. left = query[:idx].strip().
        4. right_start = idx + len(" или ") (5 символов).
        5. right_raw = query[right_start:].
        6. Найти самую раннюю позицию любого из символов
           {",", ".", "!", "?", "—"} в right_raw. Если найдена — right =
           right_raw[:позиция].strip(), иначе right = right_raw.strip().
        7. Если left и right оба непустые — вернуть [left, right],
           иначе вернуть [query] (не удалось выделить обе сущности).
        """
        ...

    def split_implicit(self, query: str) -> list[str]:
        """
        Разбить запрос с неявным подвопросом на базовый подзапрос
        (выясняющий, каким должно быть "обычное" состояние) и явную
        часть запроса, которая ссылается на это состояние.

        TODO:
        1. lower = query.lower(). Для каждого marker из
           self._implicit_markers найти lower.find(marker); собрать
           только найденные вхождения.
        2. Если вхождений нет — вернуть [query] одним элементом.
        3. idx = минимальная позиция среди найденных.
        4. base = ("текущее состояние: " + query[:idx].strip()).strip()
           — префикс осознанно грубый (в проде здесь была бы точная
           предметная переформулировка через LLM, см. класс-докстринг);
           для этого упражнения достаточно префикса.
        5. explicit = query.strip() — явная часть вопроса всё ещё
           нуждается в retrieval сама по себе, целиком.
        6. Вернуть [base, explicit]. Порядок важен: базовая потребность
           идёт первой, потому что без неё нельзя проверить "обычность"
           явной части.
        """
        ...

    def decompose(self, query: str) -> DecompositionResult:
        """
        Точка входа: классифицировать запрос и вернуть полный
        DecompositionResult с упорядоченным списком подзапросов.

        TODO:
        1. pattern = self.classify_pattern(query)
        2. Если pattern is DecompositionPattern.ATOMIC — parts = [query].
        3. Если pattern is DecompositionPattern.CONJUNCTIVE — parts =
           self.split_conjunctive(query).
        4. Если pattern is DecompositionPattern.COMPARISON — parts =
           self.split_comparison(query).
        5. Если pattern is DecompositionPattern.IMPLICIT_SUBQUESTION —
           parts = self.split_implicit(query).
        6. sub_queries = [SubQuery(text=part, order=i, pattern=pattern)
           for i, part in enumerate(parts)]
        7. Вернуть DecompositionResult(original_query=query,
           pattern=pattern, sub_queries=sub_queries).
        """
        ...
