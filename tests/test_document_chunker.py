"""
Тесты для DocumentAwareChunker.

Урок 3.4 — Document-aware и hierarchical chunking
Курс: Проектирование RAG-систем для production
"""

import pytest
from app.ingestion.document_chunker import DocumentAwareChunker, DocumentChunk


# ---------------------------------------------------------------------------
# Тестовые данные
# ---------------------------------------------------------------------------

SIMPLE_MD = """# Регламент работы с НДС

## Общие положения

Настоящий регламент определяет порядок работы с налогом.
Действие распространяется на все юридические лица.

## Штрафные санкции

| Тип нарушения | Штраф | Минимум |
|---|---|---|
| Несвоевременная подача | 5% | 1 000 руб. |
| Неуплата | 20% | 5 000 руб. |

```python
def calculate_penalty(amount: float) -> float:
    return max(amount * 0.05, 1000.0)
```

Расчёт производится автоматически.
"""

HEADINGS_ONLY_MD = """# Раздел 1

## Подраздел 1.1

### Пункт 1.1.1
"""

CODE_BLOCK_MD = """# Документация API

Пример использования:

```python
import requests

response = requests.get("https://api.example.com/v1/data")
print(response.json())
```

Метод возвращает JSON.
"""


# ---------------------------------------------------------------------------
# Unit-тесты (не требуют внешних сервисов)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentAwareChunkerBasic:
    """Базовые тесты структуры и типов элементов."""

    def test_empty_text_returns_empty_list(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown("   \n\n\t  ")
        assert result == []

    def test_returns_list_of_document_chunks(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        assert isinstance(result, list)
        assert all(isinstance(c, DocumentChunk) for c in result)

    def test_source_stored_in_each_chunk(self):
        chunker = DocumentAwareChunker(source="reglamент.md")
        result = chunker.chunk_markdown(SIMPLE_MD)
        assert all(c.source == "reglamент.md" for c in result)

    def test_all_element_types_are_valid(self):
        valid_types = {"heading", "paragraph", "table", "code_block", "list"}
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        for chunk in result:
            assert chunk.element_type in valid_types, (
                f"Неожиданный тип: {chunk.element_type!r}"
            )


@pytest.mark.unit
class TestDocumentAwareChunkerHeadings:
    """Тесты обработки заголовков."""

    def test_heading_extracted_as_heading_type(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        headings = [c for c in result if c.element_type == "heading"]
        assert len(headings) >= 2, "Ожидается минимум 2 заголовка"

    def test_heading_content_matches_text(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        heading_contents = [c.content for c in result if c.element_type == "heading"]
        assert "Регламент работы с НДС" in heading_contents
        assert "Штрафные санкции" in heading_contents

    def test_heading_path_includes_itself(self):
        """Заголовок должен включать себя в heading_path."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        h2_headings = [c for c in result if c.element_type == "heading"
                       and "Штрафные санкции" in c.content]
        assert len(h2_headings) == 1
        path = h2_headings[0].heading_path
        assert "Штрафные санкции" in path

    def test_subheading_path_includes_parent(self):
        """Подзаголовок H2 должен иметь в пути и H1 и сам себя."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(HEADINGS_ONLY_MD)
        h3_headings = [c for c in result if c.element_type == "heading"
                       and "Пункт 1.1.1" in c.content]
        assert len(h3_headings) == 1
        path = h3_headings[0].heading_path
        assert "Раздел 1" in path
        assert "Подраздел 1.1" in path
        assert "Пункт 1.1.1" in path


@pytest.mark.unit
class TestDocumentAwareChunkerTable:
    """Тесты обработки таблиц."""

    def test_table_extracted_as_table_type(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        tables = [c for c in result if c.element_type == "table"]
        assert len(tables) == 1, f"Ожидается 1 таблица, найдено: {len(tables)}"

    def test_table_is_not_split(self):
        """Таблица должна быть в одном фрагменте, не разбита."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        tables = [c for c in result if c.element_type == "table"]
        assert len(tables) == 1
        # Все строки таблицы должны быть в одном чанке
        table_content = tables[0].content
        assert "Несвоевременная подача" in table_content
        assert "Неуплата" in table_content

    def test_table_heading_path_contains_parent_section(self):
        """Таблица должна помнить, в каком разделе она находится."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        tables = [c for c in result if c.element_type == "table"]
        assert len(tables) == 1
        path = tables[0].heading_path
        assert "Штрафные санкции" in path


@pytest.mark.unit
class TestDocumentAwareChunkerCodeBlock:
    """Тесты обработки блоков кода."""

    def test_code_block_extracted_as_code_block_type(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(CODE_BLOCK_MD)
        code_blocks = [c for c in result if c.element_type == "code_block"]
        assert len(code_blocks) == 1

    def test_code_block_not_split(self):
        """Блок кода должен быть целиком в одном фрагменте."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(CODE_BLOCK_MD)
        code_blocks = [c for c in result if c.element_type == "code_block"]
        assert len(code_blocks) == 1
        content = code_blocks[0].content
        assert "import requests" in content
        assert "response.json()" in content

    def test_code_block_heading_path(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(CODE_BLOCK_MD)
        code_blocks = [c for c in result if c.element_type == "code_block"]
        assert len(code_blocks) == 1
        path = code_blocks[0].heading_path
        assert "Документация API" in path

    def test_multiple_code_blocks_are_separate(self):
        md = """# Примеры

```python
x = 1
```

Пояснение.

```python
y = 2
```
"""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(md)
        code_blocks = [c for c in result if c.element_type == "code_block"]
        assert len(code_blocks) == 2


@pytest.mark.unit
class TestDocumentAwareChunkerParagraph:
    """Тесты обработки параграфов."""

    def test_paragraph_extracted(self):
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        paragraphs = [c for c in result if c.element_type == "paragraph"]
        assert len(paragraphs) >= 1

    def test_long_paragraph_split_by_sentences(self):
        """Параграф длиннее max_chunk_tokens разбивается по предложениям."""
        # Создаём параграф из 10 длинных предложений
        sentences = [f"Предложение номер {i}, содержащее достаточно слов для теста" for i in range(10)]
        long_para = ". ".join(sentences) + "."
        md = f"# Раздел\n\n{long_para}\n"

        chunker = DocumentAwareChunker(max_chunk_tokens=100)
        result = chunker.chunk_markdown(md)
        paragraphs = [c for c in result if c.element_type == "paragraph"]

        # С max_chunk_tokens=100 длинный параграф должен разбиться на несколько
        assert len(paragraphs) > 1, (
            "Длинный параграф должен разбиваться на несколько фрагментов"
        )
        # Каждый фрагмент должен быть <= max_chunk_tokens символов
        for p in paragraphs:
            assert len(p.content) <= 100 + 50, (
                # Допуск +50 на последнее предложение
                f"Фрагмент слишком длинный: {len(p.content)} символов"
            )

    def test_short_paragraph_not_split(self):
        md = "# Раздел\n\nКороткий текст.\n"
        chunker = DocumentAwareChunker(max_chunk_tokens=800)
        result = chunker.chunk_markdown(md)
        paragraphs = [c for c in result if c.element_type == "paragraph"]
        assert len(paragraphs) == 1

    def test_paragraph_no_empty_content(self):
        """Ни один фрагмент не должен содержать пустой контент."""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(SIMPLE_MD)
        for chunk in result:
            assert chunk.content.strip() != "", (
                f"Пустой фрагмент типа {chunk.element_type!r}"
            )


@pytest.mark.unit
class TestDocumentAwareChunkerHeadingPath:
    """Тесты корректности heading_path для всех типов элементов."""

    def test_paragraph_under_h2_has_h1_and_h2_in_path(self):
        md = """# H1 Заголовок

## H2 Подраздел

Параграф под подразделом.
"""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(md)
        paragraphs = [c for c in result if c.element_type == "paragraph"]
        assert len(paragraphs) == 1
        path = paragraphs[0].heading_path
        assert "H1 Заголовок" in path
        assert "H2 Подраздел" in path

    def test_h2_resets_deeper_headings(self):
        """После H2 предыдущий H3 не должен быть в пути."""
        md = """# Раздел 1

## Подраздел 1.1

### Пункт 1.1.1

Текст пункта.

## Подраздел 1.2

Текст нового подраздела.
"""
        chunker = DocumentAwareChunker()
        result = chunker.chunk_markdown(md)
        # Параграф "Текст нового подраздела" находится под "Подраздел 1.2"
        last_paragraph = [c for c in result if c.element_type == "paragraph"][-1]
        assert "Подраздел 1.2" in last_paragraph.heading_path
        assert "Пункт 1.1.1" not in last_paragraph.heading_path
