from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


__all__ = [
    "Episode",
    "ClockFn",
    "OUTCOME_RESOLVED",
    "OUTCOME_UNRESOLVED",
    "OUTCOME_ESCALATED",
    "HIGH_SALIENCE_KEYWORDS",
    "DEFAULT_RECENCY_HALF_LIFE_DAYS",
    "EpisodicMemoryStore",
]


# Функция, возвращающая текущее время. Тот же принцип dependency injection,
# что и ClockFn у SessionMemory (5.1) и LongTermMemoryStore (5.2) — без
# инъекции часов протестировать "эпизод трёхмесячной давности проигрывает
# вчерашнему по весу свежести" можно было бы только реальным ожиданием.
ClockFn = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


OUTCOME_RESOLVED = "resolved"
OUTCOME_UNRESOLVED = "unresolved"
OUTCOME_ESCALATED = "escalated"

# Ключевые слова, повышающие salience эпизода — сигналы неудачной попытки
# или явного недовольства клиента. Тот же rule-based принцип, что и
# DURABLE_FACT_KEYWORDS в LongTermMemoryStore (5.2), только назначение
# другое: там фильтровали "стоит ли вообще запоминать", здесь эпизод УЖЕ
# решено запомнить, и вопрос в том, насколько высоко он должен всплывать
# при будущем поиске (см. текст урока, раздел про salience).
HIGH_SALIENCE_KEYWORDS: tuple[str, ...] = (
    "недовол",
    "жалоб",
    "не помогло",
    "повторно",
    "эскалац",
    "без решения",
)

# Период полураспада recency-компонента скоринга, в днях: эпизод недельной
# давности почти так же значим по свежести, как вчерашний; эпизод годовой
# давности — почти не значим по свежести (но высокая salience может это
# компенсировать, см. _recency_weight и текст урока).
DEFAULT_RECENCY_HALF_LIFE_DAYS: float = 30.0


@dataclass
class Episode:
    """
    Одна запись эпизодической памяти — сжатое описание ЗАВЕРШИВШЕГОСЯ
    взаимодействия с клиентом (что произошло, когда, чем закончилось), а не
    отдельная реплика (SessionMemory, 5.1) и не атомарный факт
    (LongTermMemoryStore, 5.2, Fact).

    Эпизод НИКОГДА не редактируется после создания (append-only, см. текст
    урока, раздел про event sourcing) — если проблема развивается дальше
    (например, обращение переоткрыли), это НОВЫЙ эпизод, а не правка
    старого.
    """

    user_id: str
    summary: str
    outcome: str
    occurred_at: datetime = field(default_factory=_utc_now)
    salience: float = 0.0
    topic: Optional[str] = None
    ticket_id: Optional[str] = None


class EpisodicMemoryStore:
    """
    Модуль M5, урок 5.3: эпизодическая память — журнал ЗАВЕРШИВШИХСЯ
    взаимодействий с клиентом, в отличие от LongTermMemoryStore (5.2),
    который хранит текущий срез устойчивых атомарных фактов ("что известно
    о клиенте прямо сейчас").

    Append-only по конструкции: record_episode() только добавляет записи —
    метода "обновить эпизод" в этом классе намеренно нет (см. текст урока).

    retrieve_relevant_episodes() решает retrieval-задачу, симметричную
    SourceSelector (3.3) и RetrievalPlanner (3.4) из модуля M3, только
    источником являются не внешние документы, а собственная история
    взаимодействий с клиентом.

    Эта реализация — in-process backend (см. урок 5.1, раздел про
    backend'ы): для production с несколькими инстансами нужен внешний
    key-value или документный store — публичный интерфейс класса при этом
    не меняется.
    """

    def __init__(
        self,
        clock: ClockFn = _utc_now,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    ) -> None:
        self._clock = clock
        self._recency_half_life_days = recency_half_life_days
        # user_id -> список Episode в порядке добавления. Append-only:
        # record_episode() никогда не удаляет и не изменяет существующие
        # записи, только добавляет новые.
        self._episodes: dict[str, list[Episode]] = {}

    def score_salience(self, summary: str, outcome: str) -> float:
        """
        Детерминированная эвристика значимости эпизода для будущего
        retrieval (см. текст урока, раздел "salience").

        TODO:
        1. base = 0.3
        2. Если outcome == OUTCOME_UNRESOLVED: base += 0.3.
           Если outcome == OUTCOME_ESCALATED: base += 0.4.
           (OUTCOME_RESOLVED ничего не добавляет — благополучно закрытые
           обращения по умолчанию менее приоритетны для будущего поиска,
           чем нерешённые или эскалированные.)
        3. text = summary.strip().lower()
        4. За КАЖДОЕ РАЗНОЕ ключевое слово из HIGH_SALIENCE_KEYWORDS,
           найденное как подстрока в text, добавить 0.1 (несколько разных
           слов суммируются; одно и то же слово, встретившееся в тексте
           дважды, учитывается только один раз — считайте МНОЖЕСТВО
           совпавших ключевых слов, а не общее число вхождений).
        5. Вернуть min(base, 1.0) — итоговое значение не должно превышать
           1.0.
        """
        ...

    def record_episode(
        self,
        user_id: str,
        summary: str,
        outcome: str,
        topic: Optional[str] = None,
        ticket_id: Optional[str] = None,
    ) -> Episode:
        """
        Добавить новый эпизод в append-only журнал клиента. salience
        вычисляется автоматически через self.score_salience — вызывающий
        код его не передаёт явно.

        TODO:
        1. salience = self.score_salience(summary, outcome)
        2. Создать episode = Episode(user_id=user_id, summary=summary,
           outcome=outcome, occurred_at=self._clock(), salience=salience,
           topic=topic, ticket_id=ticket_id).
        3. Добавить episode в конец self._episodes.setdefault(user_id, [])
           (append — НИКОГДА не изменять и не удалять существующие
           записи).
        4. Вернуть episode.
        """
        ...

    def _recency_weight(self, occurred_at: datetime) -> float:
        """
        Вес свежести эпизода: экспоненциальное затухание с периодом
        полураспада self._recency_half_life_days. Уже реализовано —
        используйте эту функцию из retrieve_relevant_episodes, не
        переизобретайте.
        """
        age_days = max(
            0.0, (self._clock() - occurred_at).total_seconds() / 86400.0
        )
        return 0.5 ** (age_days / self._recency_half_life_days)

    def retrieve_relevant_episodes(
        self,
        user_id: str,
        query_topic: Optional[str] = None,
        top_k: int = 3,
    ) -> list[Episode]:
        """
        Вернуть до top_k наиболее релевантных эпизодов клиента для
        текущего обращения — retrieval-задача (см. текст урока, раздел про
        связь с модулем M3): не "все эпизоды", а самые значимые для ЭТОГО
        обращения.

        TODO:
        1. candidates = self._episodes.get(user_id, []). Если пусто ->
           вернуть [].
        2. Если query_topic задан (не None) — оставить в candidates
           только эпизоды с episode.topic == query_topic (точное
           совпадение). Если query_topic задан, но ни один эпизод не
           совпал — вернуть [] (не откатываться молча на "все эпизоды":
           лучше вернуть пусто, чем подсунуть нерелевантный эпизод).
        3. Для каждого кандидата вычислить итоговый score =
           episode.salience * self._recency_weight(episode.occurred_at).
        4. Отсортировать candidates по score по убыванию (при равенстве
           score — более свежий эпизод, то есть больший occurred_at,
           раньше).
        5. Вернуть первые top_k эпизодов из отсортированного списка.
        """
        ...
