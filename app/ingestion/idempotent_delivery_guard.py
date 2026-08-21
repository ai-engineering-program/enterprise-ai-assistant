from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable, Optional

from app.ingestion.event_handler import EventDrivenIngestionHandler


__all__ = [
    "compute_idempotency_key",
    "IdempotencyKeyStore",
    "IdempotentDeliveryGuard",
]


def compute_idempotency_key(event: dict) -> str:
    """
    Вычислить ключ идемпотентности для одной доставки события.

    Приоритет:
    1. Если event["event_id"] присутствует и не пустая строка — вернуть
       f"evt:{event['event_id']}". Большинство брокеров (Kafka,
       RabbitMQ) и webhook-провайдеров сохраняют event_id стабильным
       при повторной доставке одного и того же сообщения — это самый
       дешёвый и надёжный вариант ключа.
    2. Иначе (event_id отсутствует или пуст — редкий, но встречающийся
       в production случай, когда источник не проставляет стабильный
       идентификатор) — построить ключ из хеша содержимого:
       f"content:{sha256(doc_id + '::' + content)}" в hex, используя
       event["doc_id"] и event.get("content", "").

    TODO:
    1. event_id = event.get("event_id")
    2. Если event_id (непустая строка) — вернуть f"evt:{event_id}"
    3. Иначе:
       doc_id = event["doc_id"]
       content = event.get("content", "")
       raw = f"{doc_id}::{content}".encode("utf-8")
       вернуть f"content:{hashlib.sha256(raw).hexdigest()}"
    """
    ...


class IdempotencyKeyStore:
    """
    Durable-хранилище ключей идемпотентности с TTL — учебная модель
    того, что в production реализуется через Redis (SETNX + PX) или
    аналогичный атомарный durable-стор, ОБЩИЙ для всех реплик consumer
    pool (а не память одного процесса, как self._processed_ids у
    EventDrivenIngestionHandler).

    clock — внедряемая функция времени (по умолчанию time.monotonic).
    В тестах позволяет управлять "истечением" TTL без реального sleep.
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic
        # key -> момент времени (по self._clock), ПОСЛЕ которого claim
        # считается истёкшим и ключ можно захватить повторно.
        self._claims: dict[str, float] = {}
        self._lock = threading.Lock()

    def try_claim(self, key: str) -> bool:
        """
        Атомарно проверить и захватить ключ идемпотентности — аналог
        команды Redis SETNX key value PX ttl_ms в проде.

        КРИТИЧНО: проверка "ключ уже захвачен и не истёк?" и захват
        ключа должны выполняться как ОДНА неделимая операция под
        self._lock. Если разбить это на два отдельных вызова
        (сначала проверить, потом отдельно захватить) — между ними два
        конкурентных воркера, обрабатывающих одну и ту же повторную
        доставку, оба увидят "ключ свободен" и оба начнут обработку —
        именно эта гонка вызвала инцидент из текста урока.

        Returns:
            True  — ключ был свободен (не захвачен или истёк), захват
                    выполнен, вызывающий код ДОЛЖЕН выполнить обработку.
            False — ключ уже захвачен другой (более ранней или
                    конкурентной) доставкой и ещё не истёк — вызывающий
                    код обязан пропустить обработку, не читая и не
                    изменяя индекс.

        TODO:
        1. with self._lock:
        2.     now = self._clock()
        3.     expires_at = self._claims.get(key)
        4.     если expires_at is not None и expires_at > now:
                   вернуть False   # ключ занят и не истёк
        5.     self._claims[key] = now + self._ttl_seconds
        6.     вернуть True
        """
        ...

    def size(self) -> int:
        """Количество ключей, для которых сейчас есть claim (включая
        потенциально уже истёкшие — истечение проверяется лениво в
        try_claim, а не фоновой очисткой)."""
        return len(self._claims)


class IdempotentDeliveryGuard:
    """
    Обёртка над EventDrivenIngestionHandler (урок 6.1,
    app/ingestion/event_handler.py), добавляющая идемпотентность к
    ПОВТОРНОЙ ДОСТАВКЕ одного и того же события брокером сообщений
    (at-least-once delivery) — задача, которая НЕ решается:

    - чекпоинтом IdempotentIngestionPipeline (урок 1.1,
      checkpoint_tracker.py) — тот защищает от повтора работы внутри
      ОДНОГО процесса, восстанавливающегося после СВОЕГО собственного
      сбоя, а не от повторной доставки одного сообщения РАЗНЫМ
      репликам consumer pool или тому же процессу после перезапуска;
    - sequence guard StreamEventProcessor (урок 5.3,
      stream_event_processor.py) — тот отбрасывает событие с МЕНЬШИМ
      sequence, чем уже применённый для doc_id, то есть решает
      проблему НЕПРАВИЛЬНОГО ПОРЯДКА РАЗНЫХ событий. Повторная
      доставка ТОГО ЖЕ события имеет ТОТ ЖЕ sequence — она не
      "устарела" и пройдёт sequence guard как обычное новое событие;
    - собственным processed_ids у EventDrivenIngestionHandler — это
      обычный Python set в памяти одного процесса: не переживает
      перезапуск процесса, не общий между репликами consumer pool при
      горизонтальном масштабировании, и главное — проверка "уже
      обработано?" и пометка "теперь обработано" в нём выполняются
      как ДВА отдельных шага, а не одна атомарная операция (см.
      IdempotencyKeyStore.try_claim).

    НЕ изменяет и не переопределяет EventDrivenIngestionHandler —
    использует его как внутренний компонент через self._handler.
    """

    def __init__(
        self,
        handler: Optional[EventDrivenIngestionHandler] = None,
        ttl_seconds: float = 3600.0,
        store: Optional[IdempotencyKeyStore] = None,
    ) -> None:
        self._handler = handler or EventDrivenIngestionHandler()
        self._store = store or IdempotencyKeyStore(ttl_seconds=ttl_seconds)
        self._applied_count: int = 0
        self._duplicate_count: int = 0

    def handle_delivery(self, event: dict) -> str:
        """
        Обработать одну доставку события с защитой от повторной
        доставки брокером (at-least-once).

        Returns:
            "applied"   — ключ был свободен, событие передано во
                          внутренний EventDrivenIngestionHandler.
            "duplicate" — ключ уже занят другой доставкой (более
                          ранней или конкурентной) — событие ОТБРОШЕНО
                          без вызова self._handler.

        TODO:
        1. key = compute_idempotency_key(event)
        2. Если self._store.try_claim(key) is False:
               self._duplicate_count += 1
               вернуть "duplicate"
        3. Иначе:
               forwarded = dict(event)
               forwarded.setdefault("event_id", key)
               # событиям без собственного event_id нужен хоть какой-то
               # стабильный идентификатор для вложенного
               # EventDrivenIngestionHandler — используем сам ключ
               # идемпотентности, он уже детерминирован.
               self._handler.handle_event(forwarded)
               self._applied_count += 1
               вернуть "applied"

        Обратите внимание на порядок: ключ захватывается ДО вызова
        self._handler.handle_event, а не после. Если поменять порядок,
        вернётся та же гонка, которую этот класс должен устранить: два
        конкурентных вызова handle_delivery с одинаковым ключом оба
        увидят "ещё не обработано" и оба выполнят handle_event.
        """
        ...

    def stats(self) -> dict:
        """
        Вернуть статистику работы guard'а.

        Returns:
            dict с ключами:
                applied_count   — self._applied_count
                duplicate_count — self._duplicate_count
                tracked_keys    — self._store.size()

        TODO: соберите и верните словарь с этими тремя ключами.
        """
        ...
