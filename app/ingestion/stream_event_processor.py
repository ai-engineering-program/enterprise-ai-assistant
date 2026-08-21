from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.ingestion.event_handler import EventDrivenIngestionHandler


__all__ = ["StreamEvent", "StreamEventProcessor"]


@dataclass
class StreamEvent:
    """Одно событие изменения документа, пришедшее из потокового источника
    (webhook, брокер сообщений).

    sequence — монотонно растущий номер публикации события, присвоенный
    источником ОТДЕЛЬНО для каждого doc_id. Используется для обнаружения
    доставки события "не в том порядке" (out-of-order delivery), которую
    дедупликация по event_id не покрывает — см. урок 5.3.
    """

    event_id: str
    event_type: str  # "created" | "updated" | "deleted"
    doc_id: str
    content: str
    sequence: int


class StreamEventProcessor:
    """
    Обёртка над EventDrivenIngestionHandler (курс 1, урок 6.1), добавляющая
    защиту от неупорядоченной (out-of-order) доставки событий поверх уже
    готовой идемпотентности по event_id.

    EventDrivenIngestionHandler гарантирует, что повторная доставка ОДНОГО
    И ТОГО ЖЕ event_id безопасна. Но at-least-once доставка от вебхука или
    брокера сообщений не гарантирует ПОРЯДОК: два РАЗНЫХ события об одном
    документе (разные event_id) могут прийти в порядке, отличном от
    порядка их публикации — например, если доставка более раннего события
    задержалась на ретрае брокера.

    StreamEventProcessor хранит для каждого doc_id наибольший sequence,
    который был успешно применён, и отбрасывает события с меньшим
    sequence как устаревшие (stale) — даже если они не являются
    дубликатами в смысле event_id.

    НЕ изменяет и не переопределяет EventDrivenIngestionHandler — использует
    его как внутренний компонент через self._handler.
    """

    def __init__(self, handler: Optional[EventDrivenIngestionHandler] = None):
        self._handler = handler or EventDrivenIngestionHandler()
        # doc_id -> наибольший sequence, который был успешно применён к индексу
        self._last_applied_sequence: dict[str, int] = {}
        self._stale_count: int = 0

    def process(self, event: StreamEvent) -> str:
        """
        Обработать одно потоковое событие с защитой от out-of-order доставки.

        Возвращает одну из трёх строк:
            "applied"   — событие применено к индексу
            "duplicate" — событие отброшено как повтор event_id
                          (EventDrivenIngestionHandler.handle_event вернул False)
            "stale"     — событие отброшено как устаревшее: его sequence
                          меньше, чем уже применённый sequence для этого doc_id

        TODO:
        1. last_seq = self._last_applied_sequence.get(event.doc_id, -1)
        2. Если event.sequence < last_seq:
           - увеличить self._stale_count на 1
           - вернуть "stale"
           (НЕ передавать событие в self._handler — оно не должно повлиять
           на индекс, даже если у него уникальный event_id)
        3. Иначе передать событие в self._handler.handle_event(...) как dict
           с ключами event_id / event_type / doc_id / content:
               applied = self._handler.handle_event({
                   "event_id": event.event_id,
                   "event_type": event.event_type,
                   "doc_id": event.doc_id,
                   "content": event.content,
               })
        4. Если applied is False (дубликат event_id) — вернуть "duplicate"
           (self._last_applied_sequence НЕ обновляется).
        5. Если applied is True — обновить
           self._last_applied_sequence[event.doc_id] = event.sequence
           и вернуть "applied".
        """
        # TODO: реализуйте логику из docstring выше
        ...

    def stats(self) -> dict:
        """
        Вернуть статистику работы процессора.

        Returns:
            dict с ключами:
                applied_count     — self._handler.get_processed_count()
                stale_count       — self._stale_count
                tracked_documents — количество doc_id в self._last_applied_sequence

        TODO: соберите и верните словарь с этими тремя ключами.
        """
        # TODO: реализуйте сбор статистики
        ...
