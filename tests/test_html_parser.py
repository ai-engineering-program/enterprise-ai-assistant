"""Tests for app/ingestion/html_parser.py (Lesson 2.3).

All fixtures are small hand-built HTML strings — no network access or
headless browser is required, so the whole module is covered by unit
tests only. A dedicated integration test documents the SPA-rendering
case that genuinely requires a headless browser (Playwright) and is
skipped by default.

Run:
    pytest tests/test_html_parser.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.html_parser import HTMLBlock, HTMLParseResult, HTMLParser


_DOC_PAGE = """
<html>
<head><title>Настройка окружения — Документация НеваСофт</title></head>
<body>
  <header><div class="topbar">НеваСофт Wiki</div></header>
  <nav class="breadcrumbs">Главная / Документация / Backend / Заказы</nav>
  <div class="sidebar-menu">
    <p><a href="/a">Обзор архитектуры</a></p>
    <p><a href="/b">API v1</a></p>
    <p><a href="/c">API v2</a></p>
  </div>
  <main id="content">
    <h1>Настройка окружения для сервиса заказов</h1>
    <p>Первый содержательный абзац с реальной информацией, необходимой инженеру для настройки локального окружения сервиса заказов.</p>
    <p>Второй абзац продолжает описывать шаги настройки, включая переменные окружения и docker-compose команды для локального запуска.</p>
  </main>
  <div class="related-widget">
    <p>Похожие статьи: Настройка CI, Обзор архитектуры, FAQ по деплою.</p>
  </div>
  <footer class="site-footer"><p>&copy; 2024 НеваСофт. Техподдержка: helpdesk@nevasoft-internal.ru</p></footer>
</body>
</html>
"""

_LINK_ONLY_PAGE = """
<html>
<head><title>Карта раздела</title></head>
<body>
  <div class="link-list">
    <p><a href="/a">Раздел А</a></p>
    <p><a href="/b">Раздел Б</a></p>
    <p><a href="/c">Раздел В</a></p>
  </div>
</body>
</html>
"""

_SPA_SHELL_PAGE = """
<html>
<head><title>Панель поддержки</title></head>
<body>
  <div id="root"></div>
  <script src="/static/bundle.js"></script>
  <script src="/static/vendor.js"></script>
</body>
</html>
"""


@pytest.mark.unit
class TestHTMLParserNoiseRemoval:
    def test_main_text_excludes_navigation_and_footer(self):
        result = HTMLParser().parse(_DOC_PAGE)
        assert "Главная" not in result.main_text
        assert "Похожие статьи" not in result.main_text
        assert "Техподдержка" not in result.main_text
        assert "Обзор архитектуры" not in result.main_text

    def test_main_text_contains_real_content(self):
        result = HTMLParser().parse(_DOC_PAGE)
        assert "docker-compose" in result.main_text
        assert "настройки локального окружения" in result.main_text

    def test_title_extracted_from_title_tag(self):
        result = HTMLParser().parse(_DOC_PAGE)
        assert result.title == "Настройка окружения — Документация НеваСофт"

    def test_only_main_container_survives_noise_filtering(self):
        result = HTMLParser().parse(_DOC_PAGE)
        assert result.blocks_considered == 1
        assert result.blocks_kept == 1

    def test_not_flagged_as_js_shell(self):
        result = HTMLParser().parse(_DOC_PAGE)
        assert result.likely_js_rendered is False


@pytest.mark.unit
class TestHTMLParserDensityFiltering:
    def test_link_only_block_is_discarded_as_low_density(self):
        result = HTMLParser().parse(_LINK_ONLY_PAGE)
        assert result.main_text == ""
        assert result.blocks_kept == 0

    def test_relaxed_thresholds_keep_the_same_block(self):
        result = HTMLParser(min_block_chars=1, min_text_density=0.0).parse(_LINK_ONLY_PAGE)
        assert result.main_text != ""
        assert result.blocks_kept >= 1


@pytest.mark.unit
class TestHTMLParserSPADetection:
    def test_empty_react_shell_flagged_as_js_rendered(self):
        result = HTMLParser().parse(_SPA_SHELL_PAGE)
        assert result.likely_js_rendered is True
        assert result.main_text == ""

    def test_normal_page_with_analytics_script_is_not_falsely_flagged(self):
        page_with_analytics_script = _DOC_PAGE.replace(
            "</body>", '<script src="/static/analytics.js"></script></body>'
        )
        result = HTMLParser().parse(page_with_analytics_script)
        assert result.likely_js_rendered is False


@pytest.mark.integration
class TestHTMLParserIntegration:
    """Требует headless-браузер (например Playwright) для рендеринга
    реального SPA до извлечения HTML — вне рамок unit-тестов.

    Запуск: pytest tests/test_html_parser.py -v -m integration
    """

    def test_real_spa_page_after_headless_render(self):
        pytest.skip("Требуется headless-браузер — запустить вручную")
