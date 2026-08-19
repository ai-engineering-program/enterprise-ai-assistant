from __future__ import annotations

from dataclasses import dataclass


__all__ = ["Word", "ColumnLayoutAnalyzer"]


@dataclass
class Word:
    """Одно распознанное слово с координатами прямоугольника на странице.

    Соответствует одной строке вывода pytesseract.image_to_data(): x, y —
    координаты левого верхнего угла в пикселях, width/height — размеры
    прямоугольника. Ось Y растёт вниз по странице (как в изображениях).
    """

    text: str
    x: float
    y: float
    width: float
    height: float


class ColumnLayoutAnalyzer:
    """
    Упрощённый layout-aware пайплайн для двухколоночных документов (см.
    историю урока — институт «Волга-Маш» и спецификация с двумя колонками
    и таблицей допусков).

    Работает поверх координат слов, которые уже вернул OCR-движок (или
    моковые данные в тестах) — без повторного распознавания изображения.
    Три шага:

    1. find_column_gap — найти X-координату разрыва между колонками
       методом проекционного профиля (см. текст урока).
    2. split_into_columns — разделить слова страницы на группы по
       найденному разрыву (или вернуть одну группу, если разрыва нет).
    3. group_words_into_lines — сгруппировать слова внутри одной колонки
       в строки по координате Y.

    reconstruct_reading_order собирает все три шага в порядок чтения:
    колонки слева направо, внутри каждой колонки строки сверху вниз.
    """

    def __init__(
        self,
        line_tolerance: float = 12.0,
        min_gap_ratio: float = 0.03,
        center_tolerance_ratio: float = 0.25,
    ) -> None:
        # TODO: сохранить три параметра как атрибуты экземпляра
        # (self.line_tolerance, self.min_gap_ratio,
        # self.center_tolerance_ratio)
        ...

    def find_column_gap(self, words: list[Word], page_width: float) -> float | None:
        """
        Найти X-координату центра разрыва между двумя колонками методом
        проекционного профиля.

        TODO:
        1. Если words пуст — вернуть None.
        2. Построить список интервалов (w.x, w.x + w.width) для всех
           слов, отсортировать по началу интервала.
        3. Слить пересекающиеся/соприкасающиеся интервалы в список
           merged: [(start, end), ...] — если следующий интервал
           начинается раньше или ровно там, где кончается последний
           слитый (start <= merged[-1][1]), расширить последний слитый
           интервал до max(merged[-1][1], end); иначе добавить новый.
        4. Если после слияния осталось меньше двух интервалов — вернуть
           None (нет разрыва вообще, скорее всего одна колонка).
        5. Для каждой пары соседних слитых интервалов (merged[i],
           merged[i+1]) вычислить:
               gap_start = merged[i][1]
               gap_end = merged[i+1][0]
               gap_width = gap_end - gap_start
               gap_center = (gap_start + gap_end) / 2
        6. Отобрать кандидатов, где ОБА условия верны:
               gap_width >= self.min_gap_ratio * page_width
               page_width * (0.5 - self.center_tolerance_ratio)
                   <= gap_center <=
               page_width * (0.5 + self.center_tolerance_ratio)
        7. Если кандидатов нет — вернуть None.
        8. Среди кандидатов выбрать тот, у которого gap_width максимален,
           и вернуть его gap_center.
        """
        ...

    def split_into_columns(
        self, words: list[Word], page_width: float
    ) -> list[list[Word]]:
        """
        Разделить слова страницы на колонки по найденному разрыву.

        TODO:
        1. boundary = self.find_column_gap(words, page_width)
        2. Если boundary is None — вернуть [list(words)] (одна колонка,
           весь список слов без изменений).
        3. Иначе разбить words на левую и правую группы по тому, куда
           попадает центр слова (w.x + w.width / 2): меньше boundary —
           левая колонка, иначе — правая.
        4. Вернуть [левая_колонка, правая_колонка] — именно в этом
           порядке (слева направо), даже если одна из групп пуста.
        """
        ...

    def group_words_into_lines(self, words: list[Word]) -> list[list[Word]]:
        """
        Сгруппировать слова одной колонки (или всей страницы) в строки
        по координате Y.

        TODO:
        1. Если words пуст — вернуть [].
        2. Отсортировать слова по y (ordered = sorted(words, key=lambda
           w: w.y)).
        3. Завести current_line = [ordered[0]], current_top = ordered[0].y.
        4. Для каждого следующего слова w в ordered[1:]:
           - если (w.y - current_top) <= self.line_tolerance: добавить w
             в current_line;
           - иначе: зафиксировать текущую строку — добавить в результат
             current_line, отсортированный по x (sorted(current_line,
             key=lambda w: w.x)); начать новую строку current_line = [w],
             current_top = w.y.
        5. После цикла добавить в результат последнюю current_line,
           тоже отсортированную по x.
        6. Вернуть список строк (каждая строка — list[Word]) в порядке
           сверху вниз.
        """
        ...

    def reconstruct_reading_order(self, words: list[Word], page_width: float) -> str:
        """
        Собрать полный текст страницы в правильном порядке чтения.

        TODO:
        1. columns = self.split_into_columns(words, page_width)
        2. Для каждой колонки column_words в columns (в порядке слева
           направо):
           - lines = self.group_words_into_lines(column_words)
           - line_texts = [" ".join(w.text for w in line) for line in
             lines]
           - column_text = "\\n".join(line_texts)
        3. Собрать список column_text для непустых колонок (колонку с
           lines == [] пропустить — иначе в тексте появится лишний
           разделитель).
        4. Вернуть "\\n\\n".join(...) по этому списку — колонки
           разделены пустой строкой, как отдельные блоки страницы.
        """
        ...
