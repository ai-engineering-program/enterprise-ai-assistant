from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from app.context.context_summarizer import estimate_tokens


__all__ = [
    "Turn",
    "ClockFn",
    "DEFAULT_MAX_HISTORY_TOKENS",
    "DEFAULT_TTL_SECONDS",
    "SessionMemory",
]


# Функция, возвращающая текущее время. В production это просто
# datetime.now(timezone.utc), но для unit-тестов TTL нужен полностью
# детерминированный "фейковый" источник времени — тот же принцип
# dependency injection, что и у summarize_fn в ContextSummarizer (4.1) или
# strategy_fn в CompressionEvalHarness (4.5): без инъекции часов
# протестировать "сессия истекла через 30 минут бездействия" можно было бы
# только реальным sleep(1800) в тесте, что неприемлемо.
ClockFn = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Бюджет по умолчанию для истории ОДНОЙ сессии, отдаваемой get_history() в
# один вызов LLM. Это отдельный бюджет от DEFAULT_MAX_CONTEXT_TOKENS у
# ContextSummarizer (4.1) — история разговора и найденные RAG-фрагменты
# конкурируют за разные, но пересекающиеся в итоговом промпте бюджеты (см.
# текст урока, раздел про токеновое давление по оси времени).
DEFAULT_MAX_HISTORY_TOKENS: int = 1500

# Сколько секунд бездействия (с момента последней реплики) считается
# "разговор ещё продолжается". 30 минут — типичное значение для
# чат-поддержки: если клиент вернулся быстрее, это тот же разговор; если
# позже — уже новая, не связанная сессия (см. текст урока, раздел про TTL).
DEFAULT_TTL_SECONDS: int = 30 * 60


@dataclass
class Turn:
    """
    Одна реплика внутри сессии.

    role — "user" или "assistant" (роль в терминах chat-completion API,
    та же семантика, что и в системном промпте курса 1). created_at
    используется ТОЛЬКО для диагностики постфактум (тот же принцип, что
    original_char_length у SummarizationResult, 4.1) — решение об
    истечении TTL принимается по last_activity_at самой сессии, а не по
    created_at отдельных реплик.
    """

    role: str
    content: str
    created_at: datetime = field(default_factory=_utc_now)


class SessionMemory:
    """
    Модуль M5, урок 5.1: сессионная память — компонент, без которого
    КАЖДЫЙ вызов LLM (см. курс 1, паттерн request-response) представляет
    собой разговор с полной амнезией между репликами (инцидент «Печора
    Телеком», где Атлас переспросил номер счёта, названный двумя
    сообщениями раньше).

    SessionMemory решает ровно ту же по духу задачу, что RetrievalPlanner
    (3.4) решает для внешних источников: не ХРАНИТ решение о том, что
    ответить, а решает, какая часть уже произошедшего (в данном случае —
    реплик текущего разговора, а не документов) должна попасть в
    СЛЕДУЮЩИЙ промпт.

    Эта реализация — in-process backend (обычный dict в памяти процесса,
    см. текст урока, раздел про backend'ы): подходит для одного инстанса
    сервиса и для unit-тестирования логики без единого внешнего сервиса.
    Production-версия с несколькими инстансами за балансировщиком требует
    вынести self._sessions/self._last_activity во внешнее хранилище
    (Redis и т.п.) — сам публичный интерфейс класса (append_turn,
    get_history, is_expired, clear_session) при этом не меняется, ровно
    как ContextSummarizer не меняется при подстановке другого summarize_fn.
    """

    def __init__(
        self,
        max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        token_counter: Callable[[str], int] = estimate_tokens,
        clock: ClockFn = _utc_now,
    ) -> None:
        self._max_history_tokens = max_history_tokens
        self._ttl_seconds = ttl_seconds
        self._token_counter = token_counter
        self._clock = clock

        self._sessions: dict[str, list[Turn]] = {}
        self._last_activity: dict[str, datetime] = {}

    def is_expired(self, session_id: str) -> bool:
        """
        Проверить, истёк ли TTL сессии по времени бездействия.

        TODO:
        1. Если session_id отсутствует в self._last_activity — вернуть
           True (нечего продолжать: с точки зрения вызывающего кода
           отсутствующая сессия ведёт себя так же, как истёкшая —
           get_history() должен вернуть пустую историю в обоих случаях).
        2. elapsed = (self._clock() - self._last_activity[session_id])
        3. Вернуть elapsed.total_seconds() > self._ttl_seconds (строго
           больше: секунда в секунду — ещё не истекла).
        """
        ...

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Добавить одну реплику в историю сессии.

        TODO:
        1. Если self.is_expired(session_id) вернул True (в том числе для
           совершенно новой сессии, для которой is_expired всегда True) —
           начать сессию заново: self._sessions[session_id] = [] (не
           переиспользовать историю истёкшей сессии — см. текст урока,
           риск "подмешать в новый разговор контекст старого").
        2. self._sessions[session_id].append(Turn(role=role, content=content,
           created_at=self._clock()))
        3. self._last_activity[session_id] = self._clock() (обновить
           отметку активности ПОСЛЕ добавления реплики).
        """
        ...

    def get_history(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> list[Turn]:
        """
        Вернуть историю сессии, пригодную для подстановки в промпт
        следующего вызова LLM — в пределах бюджета токенов и/или числа
        реплик, начиная с САМЫХ СВЕЖИХ (см. текст урока, раздел про
        токеновое давление по оси времени: если всё не помещается,
        жертвовать нужно старыми репликами, а не свежими).

        TODO:
        1. Если self.is_expired(session_id) — вернуть [] (пустая история:
           либо сессии никогда не было, либо она истекла по TTL).
        2. turns = self._sessions.get(session_id, [])
        3. budget = max_tokens if max_tokens is not None else
           self._max_history_tokens
        4. Идти по turns С КОНЦА (от самой свежей реплики к самой старой),
           накапливая self._token_counter(turn.content); включать реплику
           в результат, пока накопленная сумма (вместе с этой репликой) не
           превышает budget. Как только очередная реплика превысила бы
           бюджет — остановиться, дальше (более старые) реплики не
           включать.
        5. Если max_turns задан — после шага 4 оставить не более
           max_turns последних (по времени) реплик из уже отобранных на
           шаге 4 (то есть ограничение по числу реплик применяется
           ДОПОЛНИТЕЛЬНО к ограничению по токенам, а не вместо него).
        6. Вернуть отобранные реплики в ИСХОДНОМ хронологическом порядке
           (от старой к свежей) — промпту нужен разговор в порядке, в
           котором он происходил, а не в обратном.
        """
        ...

    def total_tokens(self, session_id: str) -> int:
        """
        Суммарное число токенов ВСЕЙ сохранённой истории сессии, без
        применения бюджета get_history() — используется, чтобы решить,
        не пора ли применить к старой части истории один из инструментов
        компрессии модуля M4 (см. текст урока, раздел про сжатие по оси
        времени). Сама компрессия здесь не реализуется — только измерение,
        нужное для принятия решения.

        TODO: вернуть sum(self._token_counter(t.content) for t in
        self._sessions.get(session_id, [])).
        """
        ...

    def clear_session(self, session_id: str) -> None:
        """
        Полностью удалить сессию (историю и отметку активности).

        TODO: удалить session_id из self._sessions и self._last_activity,
        если они там есть (без исключения, если session_id отсутствует —
        повторный вызов clear_session на уже очищенной сессии должен
        быть безопасным no-op).
        """
        ...
