from __future__ import annotations

import asyncio
from typing import Any


__all__ = ["DocumentBatcher", "BoundedIngestionQueue"]


class DocumentBatcher:
    """Копит документы во внутреннем буфере и отдаёт их плотными батчами
    фиксированного размера — амортизация накладных расходов одного вызова
    embedding API на несколько документов сразу (см. урок 5.4).

    Использование (внутри воркера-consumer'а):

        batcher = DocumentBatcher(batch_size=32)
        for item in stream_of_documents:
            batch = batcher.add(item)
            if batch is not None:
                await embed_batch(batch)          # один вызов API на 32 документа
        remainder = batcher.flush()
        if remainder:
            await embed_batch(remainder)          # неполный "хвостовой" батч
    """

    def __init__(self, batch_size: int) -> None:
        # TODO: сохранить self.batch_size = batch_size (batch_size >= 1)
        # TODO: self._buffer: list[Any] = []
        ...

    def add(self, item: Any) -> list[Any] | None:
        """Добавить один документ в буфер.

        TODO:
        1. Добавить item в self._buffer.
        2. Если len(self._buffer) достиг self.batch_size — сохранить
           текущий self._buffer в локальную переменную, завести новый
           пустой self._buffer = [] и вернуть сохранённый полный батч.
           (Важно вернуть именно ГОТОВЫЙ батч, а не ссылку на буфер,
           который переиспользуется дальше — иначе следующие add()
           будут молча мутировать уже "отданный" батч.)
        3. Иначе (батч ещё не набрался) — вернуть None.
        """
        ...

    def flush(self) -> list[Any]:
        """Забрать остаток буфера — неполный батч на конце потока документов.

        TODO: вернуть все документы, оставшиеся в self._buffer, и очистить
        буфер (self._buffer = []). Если буфер пуст — вернуть [].
        """
        ...

    def pending_count(self) -> int:
        """Сколько документов сейчас накоплено в буфере, но ещё не отданы
        ни через add(), ни через flush().

        TODO: вернуть len(self._buffer)
        """
        ...


class BoundedIngestionQueue:
    """Очередь поглощения с ограниченным размером (ёмкость max_size) и
    двумя уровнями backpressure (обратного давления) — см. урок 5.4.

    1. "Мягкое" (soft) торможение — throttle_delay() возвращает задержку
       перед постановкой в очередь, которая линейно растёт при
       приближении заполненности к high_watermark. Producer, вызывающий
       put(), сам замедляется заранее, ДО того как очередь заполнится
       полностью.
    2. "Жёсткое" (hard) торможение — если очередь всё же заполнена до
       max_size, put() блокируется на await self._queue.put() до тех пор,
       пока consumer не освободит место через get(). Это гарантирует
       жёсткий потолок памяти независимо от того, сработало ли мягкое
       торможение.

    Ниже low_watermark утилизация задержки не добавляет вовсе — очередь
    почти пуста, торможение производителя не нужно.

    Использование:

        queue = BoundedIngestionQueue(max_size=2000, low_watermark=0.5,
                                       high_watermark=0.85, max_delay=2.0)

        # producer
        await queue.put(document)   # может и сон перед постановкой, и блок

        # consumer
        document = await queue.get()
    """

    def __init__(
        self,
        max_size: int,
        low_watermark: float = 0.5,
        high_watermark: float = 0.85,
        max_delay: float = 2.0,
    ) -> None:
        # TODO: валидация: 0 <= low_watermark < high_watermark <= 1.0
        #       (при нарушении — raise ValueError)
        # TODO: self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        # TODO: сохранить self._max_size = max_size
        # TODO: сохранить self._low = low_watermark
        # TODO: сохранить self._high = high_watermark
        # TODO: сохранить self._max_delay = max_delay
        ...

    def depth(self) -> int:
        """Сколько элементов сейчас физически лежит в очереди.

        TODO: вернуть self._queue.qsize()
        """
        ...

    def utilization(self) -> float:
        """Доля заполненности очереди в диапазоне [0.0, 1.0].

        TODO: вернуть self.depth() / self._max_size
        """
        ...

    def throttle_delay(self) -> float:
        """Вычислить рекомендуемую задержку (в секундах) перед следующим put().

        TODO:
        u = self.utilization()
        1. Если u <= self._low: вернуть 0.0
        2. Если u >= self._high: вернуть self._max_delay
        3. Иначе — линейная интерполяция между low и high:
           self._max_delay * (u - self._low) / (self._high - self._low)
        """
        ...

    async def put(self, item: Any) -> None:
        """Поставить документ в очередь с двумя уровнями backpressure.

        TODO:
        1. delay = self.throttle_delay()
        2. Если delay > 0: await asyncio.sleep(delay) — мягкое торможение
           ДО попытки поставить элемент в очередь.
        3. await self._queue.put(item) — жёсткая блокировка: если очередь
           уже заполнена до max_size, корутина не вернёт управление, пока
           consumer не вызовет get() и не освободит место.
        """
        ...

    async def get(self) -> Any:
        """Забрать следующий документ из очереди (освобождает место для put()).

        TODO: вернуть await self._queue.get()
        """
        ...
