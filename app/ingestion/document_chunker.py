from __future__ import annotations

import re
from dataclasses import dataclass, field


__all__ = ["DocumentChunk", "DocumentAwareChunker"]


@dataclass
class DocumentChunk:
    """Структурный фрагмент документа с метаданными."""

    content: str
    element_type: str        # "heading", "paragraph", "table", "code_block", "list"
    heading_path: list[str] = field(default_factory=list)  # ["Раздел 3", "Штрафные санкции"]
    source: str = ""         # Путь к файлу или идентификатор документа


class DocumentAwareChunker:
    """Разбивка Markdown-документов на структурные фрагменты.

    В отличие от RecursiveCharacterTextSplitter, этот класс сначала
    извлекает структуру документа (заголовки, таблицы, блоки кода,
    параграфы), а затем разбивает на фрагменты, уважая границы
    структурных единиц.

    Поддерживаемые форматы: Markdown.
    Для HTML и PDF — см. курс 3 «Инжиниринг данных».
    """

    # Паттерн для заголовков: # Заголовок, ## Подзаголовок
    _HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')
    # Паттерн для строк таблицы: | col1 | col2 |
    _TABLE_ROW_RE = re.compile(r'^\|.*\|$')
    # Начало блока кода: ```python или просто ```
    _CODE_FENCE_RE = re.compile(r'^```')

    def __init__(self, max_chunk_tokens: int = 800, source: str = "") -> None:
        # TODO: сохранить max_chunk_tokens как self.max_chunk_tokens
        # TODO: сохранить source как self.source
        ...

    def chunk_markdown(self, text: str) -> list[DocumentChunk]:
        """Разобрать Markdown-текст и вернуть список структурных фрагментов.

        Правила разбиения:
        - Заголовки (# ## ###) — отдельный элемент типа "heading",
          уровень определяется количеством символов #.
        - Блоки кода (``` ... ```) — отдельный элемент типа "code_block",
          всегда целиком, не разбивать по токенам.
        - Строки таблицы (| col | col |) — собирать в один элемент
          типа "table", всегда целиком.
        - Остальной текст — элементы типа "paragraph".
        - Каждый элемент несёт heading_path: список текстов заголовков,
          под которыми он находится.
        - Если параграф длиннее max_chunk_tokens символов —
          разбить его по предложениям (". "), каждая часть <= max_chunk_tokens.

        Returns:
            Список DocumentChunk. Пустой список для пустого текста.
        """
        # TODO: вернуть [] если text.strip() == ""

        # TODO: разбить text на строки: lines = text.split('\n')

        # TODO: обойти строки и отслеживать состояние:
        #   - heading_stack: list[tuple[int, str]] — стек [(уровень, текст), ...]
        #   - in_code_block: bool — флаг, находимся ли внутри блока кода
        #   - code_lines: list[str] — накапливаем строки блока кода
        #   - in_table: bool — флаг, находимся ли внутри таблицы
        #   - table_lines: list[str] — накапливаем строки таблицы
        #   - paragraph_lines: list[str] — накапливаем строки параграфа

        # TODO: для каждой строки:
        #   1. Если _CODE_FENCE_RE.match(line):
        #      - Если in_code_block=False: завершить текущий параграф/таблицу,
        #        установить in_code_block=True, начать code_lines = [line]
        #      - Если in_code_block=True: закрыть блок, добавить code_lines + [line]
        #        как DocumentChunk(element_type="code_block", ...)
        #        установить in_code_block=False
        #   2. Иначе если in_code_block=True: добавить строку в code_lines

        #   3. Иначе если _TABLE_ROW_RE.match(line.strip()):
        #      - Если in_table=False: завершить параграф, установить in_table=True
        #      - Добавить строку в table_lines
        #   4. Иначе если in_table=True и строка пустая:
        #      - Завершить таблицу: добавить table_lines как DocumentChunk(element_type="table", ...)
        #        установить in_table=False
        #   5. Иначе если _HEADING_RE.match(line):
        #      - Завершить текущий параграф/таблицу
        #      - Обновить heading_stack: удалить заголовки с уровнем >= текущему,
        #        добавить (level, heading_text)
        #      - Добавить DocumentChunk(element_type="heading", ...)
        #   6. Иначе: добавить строку в paragraph_lines

        # TODO: по завершении цикла — flush оставшихся paragraph_lines и table_lines

        # TODO: _flush_paragraph(lines, heading_path) — вспомогательный метод:
        #   - объединить строки в текст, пропустить пустые
        #   - если текст длиннее max_chunk_tokens — разбить по ". "
        #   - вернуть список DocumentChunk(element_type="paragraph", ...)

        ...

    def _current_heading_path(self, heading_stack: list[tuple]) -> list[str]:
        """Вернуть список текстов заголовков из стека.

        Args:
            heading_stack: список кортежей (уровень: int, текст: str)

        Returns:
            Список строк-заголовков в порядке от H1 к текущему.
        """
        # TODO: вернуть [text for _, text in heading_stack]
        ...

    def _flush_paragraph(
        self,
        lines: list[str],
        heading_path: list[str],
    ) -> list[DocumentChunk]:
        """Завершить накопленный параграф и разбить при необходимости.

        Args:
            lines: накопленные строки параграфа
            heading_path: текущий путь заголовков

        Returns:
            Список DocumentChunk типа "paragraph". Пустой список, если текст пуст.
        """
        # TODO: объединить lines через '\n', вызвать strip()
        # TODO: если текст пустой — вернуть []
        # TODO: если len(text) <= self.max_chunk_tokens — вернуть [DocumentChunk(...)]
        # TODO: иначе — разбить по '. ':
        #   sentences = [s.strip() for s in text.split('. ') if s.strip()]
        #   накапливать предложения в буфер до max_chunk_tokens символов,
        #   при переполнении — сохранять фрагмент и начинать новый
        ...
