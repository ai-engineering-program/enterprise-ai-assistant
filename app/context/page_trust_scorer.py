from __future__ import annotations

from dataclasses import dataclass
from datetime import date


__all__ = ["ConfluencePage", "TrustSignals", "PageTrustScorer"]


@dataclass
class ConfluencePage:
    """
    Метаданные ОДНОЙ страницы вики (Confluence и аналоги) — не содержимое
    текста и не тип источника. TextStructureScorer (урок 2.1) уже отвечает
    на вопрос "насколько организован текст этой страницы"; этот класс
    работает с независимым слоем сигналов — метаданными жизненного цикла
    страницы, которые определяют, стоит ли доверять её содержимому как
    актуальному факту прямо сейчас, вне зависимости от того, насколько
    аккуратно страница оформлена и в каком пространстве лежит.

    topic_key — нормализованный ключ темы, по аналогии с ThreadNoiseFilter
    (урок 2.2): страницы с одинаковым topic_key описывают один и тот же
    вопрос и потенциально конфликтуют друг с другом.
    """

    page_id: str
    title: str
    space_key: str
    topic_key: str
    owner: str | None  # None или пустая строка -> страница-сирота
    last_reviewed_at: date | None  # None -> ни разу не проходила ревью
    last_edited_at: date
    view_count_90d: int
    is_deprecated: bool = False


@dataclass
class TrustSignals:
    """Сигналы доверия, посчитанные по метаданным одной страницы."""

    has_owner: bool
    days_since_review: int | None  # None -> ревью не было ни разу
    is_stale_by_views: bool
    is_deprecated: bool


class PageTrustScorer:
    """
    Оценивает, насколько можно доверять КОНКРЕТНОЙ странице вики как
    источнику истины — по метаданным жизненного цикла, а не по репутации
    системы, в которой она лежит ("это же Confluence — значит, надёжно").

    Это не замена и не дублирование других инструментов app/context/:
    - SourceRegistry (урок 1.1) оценивает авторитетность ИСТОЧНИКА как
      системы в целом (Confluence в среднем надёжнее Slack).
    - TextStructureScorer (урок 2.1) оценивает форму СОДЕРЖИМОГО текста.
    - PageTrustScorer оценивает конкретную СТРАНИЦУ по метаданным
      жизненного цикла — независимо от того, насколько хорошо она
      структурирована и в какой системе лежит.

    Хорошо структурированная страница в самом авторитетном пространстве
    Confluence всё равно может провалить эту оценку, если её никто не
    проверял полтора года и она никем не помечена как актуальная (см.
    инцидент урока 2.3).
    """

    def __init__(
        self,
        review_staleness_days: int = 180,
        min_views_90d: int = 5,
    ) -> None:
        self._review_staleness_days = review_staleness_days
        self._min_views_90d = min_views_90d

    def days_since_review(
        self, page: ConfluencePage, today: date
    ) -> int | None:
        """
        Число дней с последнего РЕАЛЬНОГО ревью страницы.

        TODO:
        1. Если page.last_reviewed_at is None — вернуть None (ревью не
           было ни разу; это не то же самое, что "0 дней назад").
        2. Иначе вернуть (today - page.last_reviewed_at).days.
        """
        ...

    def has_owner(self, page: ConfluencePage) -> bool:
        """
        Есть ли у страницы заполненный владелец/мейнтейнер.

        TODO: вернуть True, если page.owner не None и после .strip()
        не пустая строка. Иначе False.
        """
        ...

    def is_stale_by_views(self, page: ConfluencePage) -> bool:
        """
        Признак того, что страницу почти никто не открывает — слабый, но
        полезный намёк на осиротевшую страницу, которую забыли не только
        обновить, но и вообще посещать.

        TODO: вернуть True, если page.view_count_90d < self._min_views_90d.
        """
        ...

    def analyze(self, page: ConfluencePage, today: date) -> TrustSignals:
        """
        Собрать все сигналы доверия для одной страницы в TrustSignals.

        TODO: вызвать has_owner, days_since_review, is_stale_by_views и
        прочитать page.is_deprecated. Не дублировать их логику — только
        вызвать и собрать результат в TrustSignals(...).
        """
        ...

    def score(self, page: ConfluencePage, today: date) -> float:
        """
        Свести сигналы в композитную оценку доверия в диапазоне [0.0, 1.0].

        TODO:
        1. signals = self.analyze(page, today)
        2. Если signals.is_deprecated — вернуть 0.0 немедленно (явная
           пометка устаревания перекрывает все остальные сигналы, её
           нельзя "компенсировать" хорошим владельцем или свежим ревью).
        3. trust = 1.0
        4. Если not signals.has_owner — trust -= 0.3 (страница-сирота).
        5. Штраф за ревью:
           - signals.days_since_review is None -> trust -= 0.3
             (ни разу не проходила ревью, несмотря на возможный видимый
             "процесс" вроде статуса "Согласовано").
           - иначе, если signals.days_since_review >
             self._review_staleness_days -> trust -= 0.2 (ревью было,
             но давно просрочено).
           - иначе — без штрафа.
        6. Если signals.is_stale_by_views — trust -= 0.1.
        7. Ограничить диапазоном [0.0, 1.0]
           (max(0.0, min(1.0, trust))) и вернуть.
        """
        ...

    def rank_by_trust(
        self, pages: list[ConfluencePage], today: date
    ) -> list[ConfluencePage]:
        """
        Отсортировать страницы по убыванию score(). При равенстве score —
        по page_id по возрастанию, чтобы порядок был детерминирован.

        TODO: вернуть pages, отсортированный с ключом
        (-score(page, today), page.page_id).
        """
        ...

    def find_conflicting_topics(
        self, pages: list[ConfluencePage], today: date
    ) -> dict[str, str]:
        """
        Сгруппировать страницы по topic_key. Для каждой группы из 2+
        страниц, описывающих одну тему по-разному, определить страницу
        с НАИВЫСШИМ score как рекомендованную "истину" по этой теме.

        Это прямой инструмент против инцидента урока 2.3: наивая логика
        "чем выше authority_weight пространства/системы, тем достовернее"
        (SourceRegistry, урок 1.1) могла бы порекомендовать страницу из
        официального пространства просто по умолчанию — этот метод
        игнорирует пространство и авторитетность системы, опираясь только
        на реальные метаданные жизненного цикла конкретной страницы.

        TODO:
        1. Сгруппировать pages по topic_key.
        2. Оставить только группы из 2 и более страниц.
        3. Для каждой такой группы найти страницу с максимальным
           score(page, today); при равенстве score — с наименьшим
           page_id (для детерминированности).
        4. Вернуть {topic_key: page_id_рекомендованной_страницы}.
        Группы из одной страницы в результат не входят.
        """
        ...
