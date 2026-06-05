from __future__ import annotations


__all__ = ["FixedSizeChunker", "RecursiveChunker"]


class FixedSizeChunker:
    """Разбивка текста на фрагменты фиксированного размера с перекрытием.

    Использует скользящее окно: каждый следующий фрагмент начинается
    на (chunk_size - chunk_overlap) символов правее предыдущего.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        # TODO: сохранить chunk_size и chunk_overlap как атрибуты экземпляра
        # TODO: проверить, что chunk_overlap строго меньше chunk_size
        #       Если нет — поднять ValueError:
        #       f"chunk_overlap ({chunk_overlap}) должно быть строго меньше chunk_size ({chunk_size})"
        ...

    def split(self, text: str) -> list[str]:
        """Разбить text на фрагменты размером chunk_size символов.

        Шаг окна = chunk_size - chunk_overlap.
        Пустые фрагменты (только пробельные символы) не включаются.

        Returns:
            Список непустых строк-фрагментов.
        """
        # TODO: вычислить шаг: step = self.chunk_size - self.chunk_overlap
        # TODO: пройти по тексту с шагом step, нарезать окна [i : i + chunk_size]
        # TODO: не включать фрагменты, у которых chunk.strip() == ""
        # TODO: вернуть список фрагментов
        ...


class RecursiveChunker:
    """Рекурсивное разделение текста по иерархии разделителей.

    Алгоритм перебирает разделители от крупных (абзац) к мелким (символ).
    Первый разделитель, дающий фрагменты нужного размера, используется.
    Если фрагмент по-прежнему превышает chunk_size — рекурсивно применяется
    следующий разделитель.
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        # TODO: сохранить chunk_size и chunk_overlap
        # TODO: если separators передан — использовать его,
        #       иначе использовать DEFAULT_SEPARATORS
        ...

    def split(self, text: str) -> list[str]:
        """Рекурсивно разделить text с учётом иерархии разделителей.

        Returns:
            Список непустых строк-фрагментов.
        """
        # TODO: вызвать _split_recursive(text, self.separators)
        ...

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Внутренний рекурсивный метод разделения.

        Алгоритм:
        1. Взять первый разделитель из списка.
        2. Разбить текст по нему: parts = text.split(separator).
        3. Жадно объединять части в буфер, пока буфер <= chunk_size.
        4. Когда буфер переполняется — сохранить его, начать новый
           с перекрытием (последние chunk_overlap символов из буфера).
        5. Если отдельная часть сама > chunk_size — рекурсивно
           разбить её оставшимися разделителями (separators[1:]).
        6. Если разделителей не осталось — разбить посимвольно.

        Returns:
            Список непустых строк-фрагментов.
        """
        # TODO: реализовать рекурсивную логику
        ...
