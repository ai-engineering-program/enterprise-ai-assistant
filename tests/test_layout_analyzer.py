import pytest

from app.ingestion.layout_analyzer import ColumnLayoutAnalyzer, Word


PAGE_WIDTH = 1000.0


def _two_column_words() -> list[Word]:
    return [
        # Левая колонка, три строки
        Word("Диаметр", 50, 100, 100, 20),
        Word("вала", 160, 100, 60, 20),
        Word("40", 50, 140, 30, 20),
        Word("мм", 90, 140, 40, 20),
        Word("Допуск", 50, 180, 90, 20),
        Word("поля", 150, 180, 60, 20),
        # Правая колонка, три строки
        Word("Материал", 600, 100, 110, 20),
        Word("сталь", 720, 100, 70, 20),
        Word("45", 600, 140, 30, 20),
        Word("ХГ", 640, 140, 50, 20),
        Word("ГОСТ", 600, 180, 80, 20),
        Word("4543", 690, 180, 70, 20),
    ]


@pytest.mark.unit
class TestFindColumnGap:
    def test_two_column_layout_finds_centered_gap(self):
        analyzer = ColumnLayoutAnalyzer()
        gap = analyzer.find_column_gap(_two_column_words(), PAGE_WIDTH)
        assert gap is not None
        assert 220.0 <= gap <= 600.0

    def test_empty_words_returns_none(self):
        analyzer = ColumnLayoutAnalyzer()
        assert analyzer.find_column_gap([], PAGE_WIDTH) is None

    def test_single_row_full_width_has_no_wide_gap(self):
        # Одна строка обычного текста без колонок: промежутки между
        # словами — это межсловные пробелы (~10px), а не разрыв колонки.
        analyzer = ColumnLayoutAnalyzer()
        words = [
            Word("Полный", 50, 100, 100, 20),
            Word("текст", 160, 100, 100, 20),
            Word("без", 270, 100, 60, 20),
            Word("колонок", 340, 100, 150, 20),
            Word("чтобы", 500, 100, 90, 20),
            Word("проверить", 600, 100, 150, 20),
            Word("разрыв", 760, 100, 140, 20),
        ]
        assert analyzer.find_column_gap(words, PAGE_WIDTH) is None

    def test_gap_from_one_row_filled_by_words_of_another_row(self):
        # В первой строке между 300 и 700 выглядит как разрыв колонки,
        # но во второй строке ровно этот промежуток занят словом —
        # значит, разрыва на уровне всей страницы нет.
        analyzer = ColumnLayoutAnalyzer()
        words = [
            Word("левое", 50, 100, 250, 20),      # 50-300
            Word("правое", 700, 100, 200, 20),     # 700-900
            Word("заполнитель", 290, 140, 420, 20),  # 290-710, закрывает разрыв
        ]
        assert analyzer.find_column_gap(words, PAGE_WIDTH) is None

    def test_edge_gap_near_margin_is_ignored(self):
        # Широкий промежуток есть, но он не в центре страницы (у левого
        # края) — это не граница колонок, а просто широкое левое поле.
        analyzer = ColumnLayoutAnalyzer()
        words = [
            Word("маргинальный", 950, 100, 40, 20),
            Word("остальной", 100, 100, 800, 20),  # 100-900, единая колонка
        ]
        assert analyzer.find_column_gap(words, PAGE_WIDTH) is None


@pytest.mark.unit
class TestSplitIntoColumns:
    def test_two_column_layout_splits_correctly(self):
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer.split_into_columns(_two_column_words(), PAGE_WIDTH)
        assert len(columns) == 2
        left_texts = {w.text for w in columns[0]}
        right_texts = {w.text for w in columns[1]}
        assert left_texts == {"Диаметр", "вала", "40", "мм", "Допуск", "поля"}
        assert right_texts == {
            "Материал", "сталь", "45", "ХГ", "ГОСТ", "4543",
        }

    def test_single_column_returns_one_group(self):
        analyzer = ColumnLayoutAnalyzer()
        words = [
            Word("обычный", 50, 100, 200, 20),
            Word("текст", 260, 100, 200, 20),
            Word("без", 470, 100, 100, 20),
            Word("колонок", 580, 100, 300, 20),
        ]
        columns = analyzer.split_into_columns(words, PAGE_WIDTH)
        assert len(columns) == 1
        assert {w.text for w in columns[0]} == {
            "обычный", "текст", "без", "колонок",
        }

    def test_empty_words(self):
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer.split_into_columns([], PAGE_WIDTH)
        assert columns == [[]]


@pytest.mark.unit
class TestGroupWordsIntoLines:
    def test_groups_by_y_within_tolerance(self):
        analyzer = ColumnLayoutAnalyzer(line_tolerance=12.0)
        words = [
            Word("a", 50, 100, 40, 20),
            Word("b", 60, 108, 40, 20),   # diff 8 <= 12 -> та же строка
            Word("c", 70, 118, 40, 20),   # diff от начала строки 18 > 12 -> новая
        ]
        lines = analyzer.group_words_into_lines(words)
        assert len(lines) == 2
        assert [w.text for w in lines[0]] == ["a", "b"]
        assert [w.text for w in lines[1]] == ["c"]

    def test_words_within_line_sorted_by_x(self):
        analyzer = ColumnLayoutAnalyzer(line_tolerance=12.0)
        words = [
            Word("второе", 200, 100, 40, 20),
            Word("первое", 50, 102, 40, 20),
        ]
        lines = analyzer.group_words_into_lines(words)
        assert len(lines) == 1
        assert [w.text for w in lines[0]] == ["первое", "второе"]

    def test_empty_input(self):
        analyzer = ColumnLayoutAnalyzer()
        assert analyzer.group_words_into_lines([]) == []

    def test_three_distinct_lines(self):
        analyzer = ColumnLayoutAnalyzer(line_tolerance=12.0)
        left_column = [w for w in _two_column_words() if w.x < 300]
        lines = analyzer.group_words_into_lines(left_column)
        assert len(lines) == 3
        assert [w.text for w in lines[0]] == ["Диаметр", "вала"]
        assert [w.text for w in lines[1]] == ["40", "мм"]
        assert [w.text for w in lines[2]] == ["Допуск", "поля"]


@pytest.mark.unit
class TestReconstructReadingOrder:
    def test_two_column_document_columns_not_interleaved(self):
        analyzer = ColumnLayoutAnalyzer()
        text = analyzer.reconstruct_reading_order(_two_column_words(), PAGE_WIDTH)
        expected = (
            "Диаметр вала\n40 мм\nДопуск поля"
            "\n\n"
            "Материал сталь\n45 ХГ\nГОСТ 4543"
        )
        assert text == expected

    def test_left_column_fully_precedes_right_column(self):
        analyzer = ColumnLayoutAnalyzer()
        text = analyzer.reconstruct_reading_order(_two_column_words(), PAGE_WIDTH)
        # Именно эта гарантия нарушалась в истории урока: значение из
        # правой колонки не может оказаться раньше конца левой колонки.
        assert text.index("Допуск поля") < text.index("Материал сталь")

    def test_single_column_document(self):
        analyzer = ColumnLayoutAnalyzer()
        words = [
            Word("первая", 50, 100, 100, 20),
            Word("строка", 160, 100, 100, 20),
            Word("вторая", 50, 140, 100, 20),
            Word("строка", 160, 140, 100, 20),
        ]
        text = analyzer.reconstruct_reading_order(words, PAGE_WIDTH)
        assert text == "первая строка\nвторая строка"

    def test_empty_page(self):
        analyzer = ColumnLayoutAnalyzer()
        assert analyzer.reconstruct_reading_order([], PAGE_WIDTH) == ""


@pytest.mark.integration
class TestColumnLayoutAnalyzerIntegration:
    """Требует реального вывода pytesseract.image_to_data на многоколоночном скане."""

    def test_with_real_ocr_word_boxes(self):
        pytest.skip(
            "Требует запущенного Tesseract на тестовых многоколоночных сканах"
            " — запускать вручную"
        )
