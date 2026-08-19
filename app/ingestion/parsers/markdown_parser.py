from __future__ import annotations

from pathlib import Path

from app.ingestion.format_detector import DocumentFormat
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument


__all__ = ["MarkdownParser"]


_FRONT_MATTER_DELIMITER = "---"


class MarkdownParser:
    """Третий парсер, реализующий DocumentParser — без обёртки над
    существующим классом. Для Markdown в репозитории пока нет отдельного
    "родного" парсера, извлекающего заголовок и метаданные документа:
    DocumentAwareChunker (курс 2, RAG) решает другую задачу —
    структурное разбиение уже готового текста на чанки, а не извлечение
    заголовка/метаданных из сырого файла.

    Многие внутренние wiki-платформы (в том числе экспорт, на который
    "СибЛайн" перешла в истории урока 2.4) добавляют в начало
    Markdown-файла блок метаданных, обёрнутый строками "---" — так
    называемый front matter. Полноценный разбор YAML здесь избыточен:
    формат ограничен простыми строками "ключ: значение".
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Разобрать Markdown-файл и вернуть его как ParsedDocument.

        TODO:
        1. raw = path.read_text(encoding="utf-8")
        2. front_matter, body = self._split_front_matter(raw)
        3. title:
           - если front_matter.get("title") задан — использовать его
           - иначе найти первую строку body, начинающуюся с "# ", и
             взять текст после неё (обрезав пробелы) как title
           - если ни того, ни другого нет — title остаётся None
        4. blocks = [ParsedBlock(text=body.strip(), block_type="text")]
           если body.strip() непустой, иначе []
        5. Вернуть ParsedDocument(source_format=DocumentFormat.MARKDOWN,
                                    title=title, blocks=blocks,
                                    metadata=front_matter)
        """
        ...

    def _split_front_matter(self, raw: str) -> tuple[dict[str, str], str]:
        """Отделить блок front matter от остального текста.

        TODO:
        1. Разбить raw на строки: lines = raw.split("\\n")
        2. Если lines пуст или lines[0].strip() != "---" — вернуть
           ({}, raw): front matter отсутствует.
        3. Найти индекс closing_index — следующую строку (начиная со
           второй), которая равна "---" после strip(). Если такой
           строки нет — вернуть ({}, raw) (незакрытый front matter
           лучше проигнорировать целиком, чем уронить парсер).
        4. Разобрать lines[1:closing_index] построчно: для каждой
           строки со ":" внутри — front_matter[key.strip()] = value.strip()
           (key, _, value = line.partition(":")); строки без ":"
           пропустить.
        5. body = "\\n".join(lines[closing_index + 1:])
        6. Вернуть (front_matter, body).
        """
        ...
