import functools
import time
from collections import defaultdict
from typing import Callable


__all__ = ["ComponentProfiler", "timed"]


class ComponentProfiler:
    """Профилировщик компонентов RAG-конвейера.

    Накапливает замеры времени выполнения по именованным компонентам
    и вычисляет percentile-статистику (mean, p50, p95).

    Использование через методы start/stop:

        profiler = ComponentProfiler()
        profiler.start("encode_query")
        vector = encoder.encode(text)
        profiler.stop("encode_query")
        print(profiler.report())

    Использование через декоратор timed():

        @timed(profiler, "encode_query")
        def encode_query(text: str) -> list[float]:
            ...
    """

    def __init__(self):
        # TODO: инициализируйте self._timings как defaultdict(list)
        #       для хранения списков замеров (float, мс) по каждому компоненту
        # TODO: инициализируйте self._active_start как dict[str, int]
        #       для хранения начального timestamp (perf_counter_ns) незакрытых замеров
        ...

    def start(self, component: str) -> None:
        """Начать замер для компонента component.

        Args:
            component: имя компонента (например, 'encode_query', 'search', 'rerank')
        """
        # TODO: сохраните time.perf_counter_ns() в self._active_start[component]
        ...

    def stop(self, component: str) -> float:
        """Завершить замер и вернуть длительность в миллисекундах.

        Args:
            component: имя компонента — должно совпадать с переданным в start()

        Returns:
            Длительность в миллисекундах (float)

        Raises:
            RuntimeError: если start() не был вызван для этого компонента
        """
        # TODO: получите start_ns из self._active_start.pop(component, None)
        # TODO: если start_ns is None — поднимите RuntimeError с понятным сообщением
        # TODO: вычислите elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000
        # TODO: добавьте elapsed_ms в self._timings[component]
        # TODO: верните elapsed_ms
        ...

    def report(self) -> dict[str, dict]:
        """Сформировать отчёт со статистикой по всем компонентам.

        Returns:
            Словарь вида:
            {
                "encode_query": {
                    "count": 10,
                    "mean_ms": 48.3,
                    "p50_ms": 46.1,
                    "p95_ms": 71.2,
                },
                ...
            }

        Примечание:
            p95 вычисляется как sorted_samples[int(N * 0.95)].
            При N=1 p95 совпадает с единственным значением.
        """
        # TODO: для каждого компонента в self._timings:
        #   - count = len(samples)
        #   - mean_ms = sum(samples) / count
        #   - p50_ms = sorted_samples[int(count * 0.50)]
        #   - p95_ms = sorted_samples[min(int(count * 0.95), count - 1)]
        #   - округлите все значения до 1 знака после запятой
        ...


def timed(profiler: ComponentProfiler, component: str) -> Callable:
    """Фабрика декораторов: оборачивает функцию в profiler.start()/stop().

    Args:
        profiler: экземпляр ComponentProfiler для записи замеров
        component: имя компонента для отображения в отчёте

    Returns:
        Декоратор, сохраняющий имя и документацию исходной функции.

    Пример:
        profiler = ComponentProfiler()

        @timed(profiler, "search")
        def search(vector: list[float]) -> list[dict]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # TODO: реализуйте обёртку через @functools.wraps(func)
        # TODO: вызывайте profiler.start(component) перед func()
        # TODO: вызывайте profiler.stop(component) в блоке finally
        # TODO: верните результат func()
        ...
    return decorator
