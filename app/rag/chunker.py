from __future__ import annotations


__all__ = ["FixedSizeChunker", "RecursiveChunker", "SemanticChunker"]


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


class SemanticChunker:
    """Семантическая разбивка текста по embedding-расстоянию между предложениями.

    Алгоритм:
    1. Разбить текст на предложения по знакам препинания ('. ', '! ', '? ').
    2. Закодировать каждое предложение через SentenceTransformer.
    3. Вычислить косинусное расстояние между соседними эмбеддингами.
    4. Отметить точки разрыва там, где расстояние превышает threshold.
    5. Объединить предложения между точками разрыва в чанки.

    Зависимость: sentence-transformers (pip install sentence-transformers)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.4) -> None:
        # TODO: загрузить SentenceTransformer(model_name) и сохранить как self.model
        #       from sentence_transformers import SentenceTransformer
        # TODO: сохранить threshold как self.threshold
        # TODO: проверить, что threshold находится в диапазоне (0, 1)
        #       Если нет — поднять ValueError:
        #       f"threshold должен быть в диапазоне (0, 1), получено: {threshold}"
        ...

    def split(self, text: str) -> list[str]:
        """Разбить текст на семантически связные фрагменты.

        Returns:
            Список непустых строк-фрагментов.
        """
        # TODO: Шаг 1 — разбить text на предложения по '. ', '! ', '? '
        #        (можно использовать re.split или последовательные str.split)
        # TODO: Шаг 2 — отфильтровать предложения, у которых sentence.strip() == ""
        # TODO: Шаг 3 — если предложений 0 — вернуть []
        #                если предложение одно — вернуть [text.strip()]
        # TODO: Шаг 4 — закодировать все предложения: embeddings = self.model.encode(sentences)
        # TODO: Шаг 5 — вычислить косинусное расстояние между соседними эмбеддингами
        #        cosine_distance(a, b) = 1 - dot(a, b) / (norm(a) * norm(b))
        #        Используйте numpy — не нужен sklearn
        # TODO: Шаг 6 — отметить точки разрыва: индексы i, где distances[i] > self.threshold
        # TODO: Шаг 7 — сгруппировать предложения между точками разрыва в чанки
        #        (объединить через " ".join или "". join в зависимости от исходного разделения)
        # TODO: Шаг 8 — вернуть список непустых строк (chunk.strip() != "")
        ...
