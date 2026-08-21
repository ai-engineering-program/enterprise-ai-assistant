from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


__all__ = [
    "TransientIngestionError",
    "PermanentIngestionError",
    "classify_exception",
    "compute_backoff_delay",
    "DeadLetterEntry",
    "DeadLetterQueue",
    "RetryingProcessor",
]


class TransientIngestionError(Exception):
    """Сбой, при котором повтор попытки имеет разумный шанс на успех:
    сетевой timeout, обрыв соединения с провайдером эмбеддингов или
    векторной БД, HTTP 429 (rate limit). Причина — временное состояние
    внешней системы, а не сам документ."""


class PermanentIngestionError(Exception):
    """Сбой, при котором повтор НИКОГДА не поможет: документ повреждён,
    формат не поддерживается, данные не проходят валидацию схемы.
    Причина — сам документ, а не временное состояние внешней системы."""


# Известные встроенные типы исключений, которые часто прилетают из
# реальных клиентов embedding API / векторных БД и классифицируются
# однозначно без явного оборачивания в TransientIngestionError.
TRANSIENT_EXCEPTION_TYPES = (TimeoutError, ConnectionError, TransientIngestionError)
PERMANENT_EXCEPTION_TYPES = (ValueError, UnicodeDecodeError, PermanentIngestionError)


def classify_exception(exc: Exception) -> str:
    """
    Классифицировать исключение, полученное при обработке документа, как
    "transient" (повтор поможет) или "permanent" (повтор не поможет,
    документ нужно откладывать в dead-letter queue немедленно).

    Правило по умолчанию для НЕИЗВЕСТНОГО типа исключения — "permanent".
    Это осознанный консервативный выбор: неизвестная ошибка — сигнал о
    баге или непредусмотренном формате данных, а не гарантированно
    временная сетевая проблема. Бесконечно retry-ить незнакомую ошибку
    молча — верный способ повторить инцидент «Гранит-Инвест» в новой
    форме: теперь документы не пропадают, а зависают в retry-цикле
    навечно, не попадая ни в индекс, ни в dead-letter queue.

    TODO:
    1. Если isinstance(exc, TRANSIENT_EXCEPTION_TYPES) — вернуть "transient".
    2. Если isinstance(exc, PERMANENT_EXCEPTION_TYPES) — вернуть "permanent".
    3. Иначе (неизвестный тип) — вернуть "permanent" (см. объяснение выше).
    """
    ...


def compute_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    rng: Optional[Callable[[], float]] = None,
) -> float:
    """
    Вычислить задержку перед следующей попыткой — экспоненциально
    нарастающая задержка (exponential backoff) с "равным джиттером"
    (equal jitter), ограниченная сверху max_delay.

    attempt — номер ПРОШЕДШЕЙ неудачной попытки, начиная с 1.

    Без jitter все документы, упавшие в одну и ту же секунду из-за
    одного и того же временного сбоя провайдера (например, кратковременный
    503 у embedding API под нагрузкой), повторят попытку СИНХРОННО в один
    и тот же момент — и своим согласованным всплеском нагрузки продлят
    или усугубят тот самый rate limit, из-за которого они изначально
    упали (retry storm). Jitter размазывает повторные попытки по времени.

    rng — внедряемая функция генерации случайного числа в [0.0, 1.0)
    (по умолчанию random.random). В тестах позволяет получать
    детерминированный результат без реального разброса.

    TODO:
    1. capped = min(max_delay, base_delay * (2 ** (attempt - 1)))
    2. Если jitter is False — вернуть capped.
    3. rng_fn = rng or random.random
    4. "Равный джиттер": вернуть capped * (0.5 + rng_fn() * 0.5)
       — то есть результат всегда лежит в [capped * 0.5, capped), сохраняя
       расходящуюся экспоненциальную тенденцию, но убирая точную
       синхронность между документами.
    """
    ...


@dataclass
class DeadLetterEntry:
    """Одна запись в dead-letter queue — документ, который не удалось
    обработать после исчерпания лимита попыток (или сразу — при
    постоянном сбое)."""

    doc_id: str
    event: dict
    error_message: str
    category: str  # "transient" (лимит попыток исчерпан) | "permanent"
    attempt_count: int
    last_attempt_at: float
    moved_to_dlq_at: float = field(default_factory=time.time)


class DeadLetterQueue:
    """
    Ограниченное по смыслу (не безграничное "мусорное ведро", а
    наблюдаемая рабочая очередь — см. урок 5.4 про bounded-структуры)
    хранилище документов, обработка которых не удалась.

    Важно: попадание в DLQ — это НЕ "документ потерян навсегда". Это
    явный, наблюдаемый сигнал "требуется внимание человека или дебаг",
    в отличие от инцидента «Гранит-Инвест», где документы пропадали, не
    оставляя вообще никакого следа.
    """

    def __init__(self) -> None:
        self._entries: list[DeadLetterEntry] = []

    def add(self, entry: DeadLetterEntry) -> None:
        """
        Добавить запись в DLQ.

        TODO: добавить entry в self._entries.
        """
        ...

    def size(self) -> int:
        """
        Количество записей, сейчас находящихся в DLQ.

        TODO: вернуть len(self._entries).
        """
        ...

    def list_entries(self) -> list[DeadLetterEntry]:
        """
        Вернуть КОПИЮ списка записей DLQ (list(self._entries)), а не
        ссылку на внутренний список — вызывающий код не должен мочь
        мутировать состояние DLQ напрямую, минуя requeue().

        TODO: вернуть list(self._entries).
        """
        ...

    def requeue(self, doc_id: str) -> Optional[DeadLetterEntry]:
        """
        Забрать одну запись из DLQ по doc_id для повторной ручной
        обработки (например, после того как инженер пофиксил баг, или
        источник прислал исправленный документ) — и удалить её из DLQ.

        Returns:
            Найденная и удалённая запись, либо None, если записи с
            таким doc_id в DLQ нет.

        TODO:
        1. Найти первый элемент self._entries с entry.doc_id == doc_id.
        2. Если не найден — вернуть None.
        3. Если найден — удалить его из self._entries и вернуть найденную
           запись.
        """
        ...


class RetryingProcessor:
    """
    Оборачивает функцию обработки одного документа (process_fn) retry с
    экспоненциальным backoff для transient-сбоев и dead-letter queue для
    документов, не обработанных после max_attempts попыток — а также для
    любого permanent-сбоя немедленно, без единой лишней попытки.

    process_fn(event) должна поднимать исключение (в частности, одно из
    TRANSIENT_EXCEPTION_TYPES / PERMANENT_EXCEPTION_TYPES) при сбое и
    ничего не возвращать при успехе.

    sleep_fn и clock — внедряемые зависимости (по умолчанию time.sleep и
    time.time) — позволяют тестировать retry с backoff без реального
    ожидания в тестах.
    """

    def __init__(
        self,
        process_fn: Callable[[dict], None],
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep_fn: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
        rng: Optional[Callable[[], float]] = None,
    ) -> None:
        # TODO: сохранить process_fn, max_attempts, base_delay, max_delay
        #       в self._process_fn / self._max_attempts / self._base_delay /
        #       self._max_delay
        # TODO: self._sleep = sleep_fn or time.sleep
        # TODO: self._clock = clock or time.time
        # TODO: self._rng = rng
        # TODO: self._dlq = dead_letter_queue if dead_letter_queue is not
        #       None else DeadLetterQueue()
        # TODO: self._retry_counts: dict[str, int] = {}
        # TODO: self._last_attempt_at: dict[str, float] = {}
        # TODO: self._succeeded_count: int = 0
        ...

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        return self._dlq

    def process_with_retry(self, doc_id: str, event: dict) -> str:
        """
        Обработать один документ с retry + dead-letter queue.

        Returns:
            "success"     — process_fn() выполнилась без исключения.
            "dead_letter" — сбой постоянный, либо временный, но лимит
                            попыток (max_attempts) исчерпан; событие
                            добавлено в self._dlq.

        TODO:
        1. attempt = 0
        2. Бесконечный цикл:
           a. attempt += 1
           b. self._last_attempt_at[doc_id] = self._clock()
           c. try: self._process_fn(event)
              При успехе: self._retry_counts[doc_id] = attempt;
              self._succeeded_count += 1; вернуть "success".
           d. except Exception as exc:
              category = classify_exception(exc)
              self._retry_counts[doc_id] = attempt
              Если category == "permanent" — сразу вызвать
              self._move_to_dead_letter(doc_id, event, exc, attempt,
              category) и вернуть "dead_letter" (НИКАКОГО backoff и
              следующей попытки — повтор гарантированно не поможет).
              Если category == "transient" и attempt >= self._max_attempts
              — вызвать self._move_to_dead_letter(doc_id, event, exc,
              attempt, category) и вернуть "dead_letter" (лимит попыток
              исчерпан).
              Иначе (transient, попытки ещё остались) — вычислить
              delay = compute_backoff_delay(attempt, self._base_delay,
              self._max_delay, rng=self._rng), вызвать self._sleep(delay)
              и перейти на следующую итерацию цикла (новая попытка).
        """
        ...

    def _move_to_dead_letter(
        self,
        doc_id: str,
        event: dict,
        exc: Exception,
        attempt: int,
        category: str,
    ) -> None:
        """
        Построить DeadLetterEntry по данным неудачной обработки и
        добавить его в self._dlq.

        TODO: DeadLetterEntry(doc_id=doc_id, event=event,
        error_message=str(exc), category=category, attempt_count=attempt,
        last_attempt_at=self._last_attempt_at[doc_id]); передать в
        self._dlq.add(...).
        """
        ...

    def retry_count(self, doc_id: str) -> int:
        """
        Сколько попыток всего было потрачено на doc_id (успешных или
        нет).

        TODO: вернуть self._retry_counts.get(doc_id, 0).
        """
        ...

    def last_attempt_at(self, doc_id: str) -> Optional[float]:
        """
        Метка времени последней попытки для doc_id, или None, если
        попыток не было.

        TODO: вернуть self._last_attempt_at.get(doc_id).
        """
        ...

    def stats(self) -> dict:
        """
        Сводная статистика работы processor'а.

        TODO: вернуть dict с ключами:
            succeeded_count   — self._succeeded_count
            dead_letter_count — self._dlq.size()
            tracked_documents — len(self._retry_counts)
        """
        ...
