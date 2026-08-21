from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from app.ingestion.pipeline import IngestionReport
from app.rag.embedding_utils import get_embedding


__all__ = ["StageTiming", "IngestionStageProfiler"]


T = TypeVar("T")


@dataclass
class StageTiming:
    """Один замер одной стадии обработки одного файла."""

    stage: str
    file: str
    duration_ms: float


class IngestionStageProfiler:
    """Профилирует стадии MinimalIngestionPipeline с детализацией по файлам.

    В отличие от одного суммарного time.time() вокруг всего run(),
    этот инструмент измеряет каждую стадию (чтение, чанкинг, embedding,
    запись в Qdrant) отдельно для каждого файла — именно такой уровень
    детализации нужен, чтобы найти реальное узкое место, а не гадать
    (см. урок 5.1).

    Принимает уже настроенный pipeline — предполагается duck-typing:
    объект должен иметь атрибуты .chunker (с методом .split(text)),
    .vector_store (с методом .upsert_documents(docs)) и методы
    .load_text(path), .discover_files(). Это тот же интерфейс, что у
    MinimalIngestionPipeline из app/ingestion/pipeline.py (урок 1.4).

    Использование:

        pipeline = MinimalIngestionPipeline(source_dir="docs/")
        profiler = IngestionStageProfiler(pipeline)
        report = profiler.profile_run()
        print(profiler.stage_breakdown())
        print("Узкое место:", profiler.bottleneck_stage())
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._timings: list[StageTiming] = []

    # ------------------------------------------------------------------
    # Обёртка замера одной стадии
    # ------------------------------------------------------------------
    def _timed(self, stage: str, file_name: str, fn: Callable[[], T]) -> T:
        """Выполнить fn(), замерить длительность в миллисекундах и
        сохранить результат в self._timings как StageTiming.

        TODO:
        1. Засечь start = time.perf_counter().
        2. Вызвать result = fn().
        3. Вычислить duration_ms = (time.perf_counter() - start) * 1000.
           Используйте try/finally, чтобы замер сохранился даже если
           fn() поднимет исключение (например, UnicodeDecodeError
           при чтении файла с плохой кодировкой).
        4. Добавить в self._timings запись StageTiming(stage, file_name,
           duration_ms).
        5. Вернуть result (или заново поднять исключение из finally).
        """
        ...

    # ------------------------------------------------------------------
    # Профилирование одного файла
    # ------------------------------------------------------------------
    def profile_file(self, path: Path) -> int:
        """Обработать один файл, измеряя каждую стадию отдельно.

        Повторяет логику MinimalIngestionPipeline.process_file(), но
        каждый шаг оборачивается в self._timed(...) с именем стадии:

        TODO:
        1. text = self._timed("read", path.name,
               lambda: self.pipeline.load_text(path))
        2. Если text.strip() == "" — вернуть 0 (пустые файлы не считаем).
        3. chunks = self._timed("chunk", path.name,
               lambda: self.pipeline.chunker.split(text))
        4. Если chunks пуст — вернуть 0.
        5. Для каждого чанка получить вектор через self._timed("embed",
           path.name, lambda: get_embedding(chunk_text)) — отдельный
           замер на каждый чанк, не один общий на весь файл.
        6. Собрать документы для upsert (id, vector, metadata с source
           и chunk_index — как в process_file()).
        7. self._timed("upsert", path.name,
               lambda: self.pipeline.vector_store.upsert_documents(docs))
        8. Вернуть len(chunks).

        Обратите внимание: id точки для Qdrant в этом инструменте не
        обязателен для целей профилирования — можно использовать
        простой индекс f"{path.name}::{i}" вместо полноценного
        make_point_id(), так как профилировщик не отвечает за
        идемпотентность реальной записи.
        """
        ...

    # ------------------------------------------------------------------
    # Профилирование всего прогона
    # ------------------------------------------------------------------
    def profile_run(self) -> IngestionReport:
        """Пройти по всем файлам self.pipeline.discover_files() и
        профилировать каждый через profile_file().

        TODO:
        1. report = IngestionReport()
        2. Для каждого path из self.pipeline.discover_files():
           try:
               n = self.profile_file(path)
               report.files_processed.append(str(path))
               report.chunks_indexed += n
           except UnicodeDecodeError:
               report.files_skipped.append(str(path))
        3. Вернуть report.

        Логика ветвления идентична MinimalIngestionPipeline.run() —
        профилировщик не меняет поведение пайплайна, только измеряет его.
        """
        ...

    # ------------------------------------------------------------------
    # Агрегация результатов
    # ------------------------------------------------------------------
    def stage_breakdown(self) -> dict[str, dict]:
        """Агрегировать self._timings по стадиям.

        Returns:
            Словарь вида:
            {
                "embed": {
                    "count": 120,
                    "total_ms": 8340.5,
                    "mean_ms": 69.5,
                    "share_pct": 62.3,
                },
                ...
            }

        TODO:
        1. Сгруппировать self._timings по .stage.
        2. Для каждой стадии: count = число замеров, total_ms = сумма
           duration_ms, mean_ms = total_ms / count.
        3. share_pct — доля total_ms этой стадии от суммы total_ms по
           ВСЕМ стадиям (не только этой), в процентах, округлённая до
           1 знака после запятой.
        4. Если self._timings пуст — вернуть {}.
        """
        ...

    def bottleneck_stage(self) -> str | None:
        """Вернуть имя стадии с наибольшим суммарным total_ms.

        TODO: использовать stage_breakdown() и вернуть ключ с максимальным
        значением "total_ms". Если замеров нет — вернуть None.
        """
        ...
