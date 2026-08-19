from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup


__all__ = ["HTMLBlock", "HTMLParseResult", "HTMLParser"]


# Тэги, которые почти никогда не содержат основной контент страницы и
# удаляются целиком, независимо от того, что внутри них написано.
NOISE_TAGS = {"nav", "footer", "aside", "header", "script", "style", "form", "noscript"}

# Подстроки в атрибутах class/id (в нижнем регистре), которые обычно
# помечают вспомогательные блоки разметки: навигационные меню, подвалы,
# сайдбары, баннеры, cookie-уведомления, виджеты "похожих" материалов.
NOISE_CLASS_HINTS = (
    "nav", "menu", "sidebar", "footer", "banner",
    "advert", "cookie", "widget", "promo", "related",
)

# Тэги-контейнеры, которые рассматриваются как кандидаты на "основной
# контент" после того, как явный шум уже вырезан.
CONTAINER_TAGS = ("div", "section", "article", "main")

# id корневого DOM-узла, куда типичные SPA-фреймворки (React/Vue/Angular)
# монтируют всё приложение целиком через JavaScript.
SPA_MOUNT_POINT_IDS = {"root", "app", "application"}

JS_SHELL_MAX_BODY_TEXT_CHARS = 200
JS_SHELL_MIN_SCRIPT_TAGS = 1


@dataclass
class HTMLBlock:
    """Один кандидат на «основной контент» — контейнер-тэг с посчитанной
    плотностью текста.

    text_density == 1.0 означает, что весь текст блока — не текст ссылок;
    text_density == 0.0 означает, что весь текст блока — текст ссылок
    (типичная навигация или список ссылок без обёртки .nav/.sidebar).
    """

    tag: str
    text: str
    char_count: int
    text_density: float


@dataclass
class HTMLParseResult:
    """Результат разбора одной HTML-страницы.

    blocks_considered/blocks_kept — диагностика: сколько контейнеров
    рассматривалось как кандидат на контент и сколько прошло фильтр по
    длине текста и плотности. likely_js_rendered — см. docstring
    HTMLParser._looks_like_js_shell().
    """

    title: str | None
    main_text: str
    blocks_considered: int
    blocks_kept: int
    likely_js_rendered: bool


class HTMLParser:
    """Отделяет основной контент HTML-страницы от навигационного и
    рекламного шума разметки.

    Работает в два этапа. Первый — жёсткое удаление заведомо
    вспомогательных узлов: тэги из NOISE_TAGS целиком (script/style/nav/
    footer/aside/header/form/noscript) и любые оставшиеся элементы, чей
    атрибут class или id похож на навигацию, сайдбар, баннер, cookie-
    уведомление или блок "похожие материалы" (NOISE_CLASS_HINTS). Это
    убирает шум с явной разметочной сигнатурой — ровно то, что
    переполняло контекстное окно ассистента в истории урока.

    Второй этап — оценка того, что осталось, по плотности текста
    (упрощённый аналог эвристики readability.js/Boilerpipe): доля
    "собственного" текста контейнера, не являющегося текстом ссылок.
    Список ссылок без обёртки с узнаваемым классом всё равно будет
    отфильтрован на этом этапе, потому что почти весь его текст —
    текст ссылок, то есть text_density близка к нулю.

    Отдельная эвристика _looks_like_js_shell() ловит третий случай, не
    решаемый ни удалением тэгов, ни оценкой плотности: когда HTML
    физически не содержит контента вообще, потому что он подгружается
    JavaScript-ом в браузере после первой загрузки страницы (SPA). Для
    сырого HTML, полученного простым HTTP GET-запросом без исполнения
    JavaScript, это выглядит как минимальный "шелл" — и должно быть явно
    зафиксировано как "нужен рендеринг", а не молча превращено в
    документ с пустым main_text.
    """

    def __init__(self, min_block_chars: int = 40, min_text_density: float = 0.6) -> None:
        # TODO: сохранить оба параметра как self.min_block_chars и
        # self.min_text_density — они управляют фильтром на шаге 6 parse()
        ...

    def parse(self, html: str) -> HTMLParseResult:
        """Разобрать HTML-документ, отделив основной контент от шума.

        TODO:
        1. soup = BeautifulSoup(html, "html.parser")
        2. title = soup.title.get_text(strip=True) if soup.title else None
        3. likely_js = self._looks_like_js_shell(soup)
           (обязательно ДО удаления script-тэгов на следующем шаге —
           эвристике нужно посчитать реальное количество <script>)
        4. self._strip_noise(soup) — мутирует soup, удаляя шум "на месте"
        5. candidates = self._candidate_blocks(soup)
        6. kept = [b for b in candidates
                   if b.char_count >= self.min_block_chars
                   and b.text_density >= self.min_text_density]
        7. main_text = "" если kept пуст, иначе text блока с максимальным
           char_count среди kept (max(kept, key=lambda b: b.char_count).text)
        8. Вернуть HTMLParseResult(title=title, main_text=main_text,
           blocks_considered=len(candidates), blocks_kept=len(kept),
           likely_js_rendered=likely_js)
        """
        ...

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        """Удалить из DOM-дерева узлы с заведомо вспомогательной ролью.

        TODO:
        1. Для каждого имени тэга в NOISE_TAGS: soup.find_all(tag_name),
           вызвать .decompose() на каждом найденном узле.
        2. Для каждого ОСТАВШЕГОСЯ узла с атрибутом class или id:
           - class_attr_str = " ".join(tag.get("class") or [])
           - id_attr = tag.get("id") or ""
           - signature = (class_attr_str + " " + id_attr).lower()
           - если хотя бы одна подстрока из NOISE_CLASS_HINTS входит в
             signature -> tag.decompose()
           Обходить нужно по снимку списка тэгов (например
           list(soup.find_all(True))), а не напрямую по живому дереву —
           decompose() части узлов во время итерации может привести к
           пропуску соседних узлов.
        """
        ...

    def _candidate_blocks(self, soup: BeautifulSoup) -> list[HTMLBlock]:
        """Собрать кандидатов на основной контент среди CONTAINER_TAGS.

        TODO: для каждого тэга в soup.find_all(list(CONTAINER_TAGS)):
        1. paragraphs = tag.find_all(["p", "li"])
        2. own_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs).strip()
        3. Если own_text пуст -> пропустить этот тэг (continue) — это
           просто обёртка без собственного текстового контента
        4. char_count = len(own_text)
        5. link_char_count = sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a"))
        6. text_density = 1.0, если char_count == 0, иначе
           max(0.0, (char_count - link_char_count) / char_count)
        7. Добавить HTMLBlock(tag=tag.name, text=own_text,
           char_count=char_count, text_density=text_density)
        Вернуть собранный список.

        Ограничение упрощённого курса: если один контейнер вложен в
        другой (например <article> внутри <main>), оба могут получить
        одинаковый own_text — production-алгоритмы (CETR, Boilerpipe)
        решают это выбором самого глубокого контейнера, покрывающего
        весь текст; здесь это не реализовано.
        """
        ...

    def _looks_like_js_shell(self, soup: BeautifulSoup) -> bool:
        """Эвристически определить SPA-шелл, не содержащий контента без JS.

        TODO:
        1. Если soup.body is None -> вернуть False
        2. visible_text_len = len(soup.body.get_text(strip=True))
        3. script_count = len(soup.find_all("script"))
        4. has_mount_point = bool(soup.find(id=lambda value: value in SPA_MOUNT_POINT_IDS))
        5. Вернуть True, если одновременно:
           visible_text_len < JS_SHELL_MAX_BODY_TEXT_CHARS
           и script_count >= JS_SHELL_MIN_SCRIPT_TAGS
           и has_mount_point
           иначе -> False
        """
        ...
