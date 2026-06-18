from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextChunk:
    text: str
    score: float          # score релевантности от поиска (0.0–1.0)
    source: str           # идентификатор источника (имя файла, doc_id и т.п.)
    metadata: dict = field(default_factory=dict)


def reorder_context_for_attention(chunks: list[ContextChunk]) -> list[ContextChunk]:
    """
    Переупорядочивает фрагменты по принципу «лучшие — на края».

    Алгоритм:
    1. Отсортировать фрагменты по score по убыванию.
    2. Распределить по позициям: чередовать начало и конец результата.
       - Фрагмент с rank 0 (лучший) → позиция 0 (начало)
       - Фрагмент с rank 1 → последняя позиция
       - Фрагмент с rank 2 → позиция 1
       - Фрагмент с rank 3 → предпоследняя позиция
       - ...и так далее

    Пример для 5 фрагментов [A, B, C, D, E] по убыванию score:
    Результат: [A, C, E, D, B]
    Позиции:    0  1  2  3  4

    При пустом входном списке возвращает пустой список.
    """
    # TODO: отсортировать chunks по score (убывание)
    # TODO: распределить по позициям: лучшие → края, худшие → середина
    ...


class ContextBuilder:
    """
    Собирает контекстный блок для передачи в языковую модель.

    Параметры:
        max_chunks: максимальное число фрагментов (по умолчанию 5)
        reorder: применять ли переупорядочивание для минимизации
                 эффекта «потери в середине» (по умолчанию True)
    """

    def __init__(self, max_chunks: int = 5, reorder: bool = True) -> None:
        self.max_chunks = max_chunks
        self.reorder = reorder

    def build(self, chunks: list[ContextChunk]) -> str:
        """
        Формирует строку контекста для подстановки в промпт.

        1. Обрезает список до max_chunks (берёт первые max_chunks по убыванию score).
        2. Если reorder=True — применяет reorder_context_for_attention.
        3. Форматирует каждый фрагмент как:

               [Документ N] Источник: {source}
               {text}
               <пустая строка>

        Возвращает строку, готовую для вставки в системный промпт.
        При пустом входном списке возвращает пустую строку.
        """
        # TODO: обрезать до max_chunks (по убыванию score)
        # TODO: применить reorder если self.reorder == True
        # TODO: отформатировать в строку с метками [Документ N]
        ...


__all__ = [
    "ContextChunk",
    "reorder_context_for_attention",
    "ContextBuilder",
]
