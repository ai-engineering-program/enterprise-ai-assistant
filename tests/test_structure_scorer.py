import pytest

from app.context.structure_scorer import (
    ContentStructureLevel,
    StructureSignals,
    TextStructureScorer,
)


@pytest.mark.unit
class TestSplitLines:
    def test_drops_empty_lines(self):
        scorer = TextStructureScorer()
        text = "Строка один\n\n   \nСтрока два\n"
        assert scorer.split_lines(text) == ["Строка один", "Строка два"]


@pytest.mark.unit
class TestHeadingDensity:
    def test_empty_text_is_zero(self):
        scorer = TextStructureScorer()
        assert scorer.heading_density("") == 0.0

    def test_mixed_headings_and_plain_lines(self):
        scorer = TextStructureScorer()
        text = (
            "# Заголовок раздела\n"
            "Обычный текст без разметки.\n"
            "h2. Ещё один\n"
            "Простая строка."
        )
        assert scorer.heading_density(text) == pytest.approx(0.5)

    def test_no_headings_is_zero(self):
        scorer = TextStructureScorer()
        text = "Просто текст.\nЕщё одна строка без заголовков."
        assert scorer.heading_density(text) == 0.0


@pytest.mark.unit
class TestListDensity:
    def test_mixed_list_and_plain_lines(self):
        scorer = TextStructureScorer()
        text = (
            "- Пункт один\n"
            "- Пункт два\n"
            "Обычный текст.\n"
            "1. Нумерованный пункт"
        )
        assert scorer.list_density(text) == pytest.approx(0.75)

    def test_no_lists_is_zero(self):
        scorer = TextStructureScorer()
        text = "Просто предложение без списков и пунктов вообще."
        assert scorer.list_density(text) == 0.0


@pytest.mark.unit
class TestFieldDensity:
    def test_mixed_field_and_plain_lines(self):
        scorer = TextStructureScorer()
        text = (
            "Root Cause: сбой конфигурации\n"
            "Вот текст без поля.\n"
            "Due date: 20.07\n"
            "Простая строка без двоеточия"
        )
        assert scorer.field_density(text) == pytest.approx(0.5)

    def test_no_fields_is_zero(self):
        scorer = TextStructureScorer()
        text = "Просто абзац прозы без единого поля ключ-значение."
        assert scorer.field_density(text) == 0.0


@pytest.mark.unit
class TestSentenceLengthVariance:
    def test_fewer_than_two_sentences_returns_zero(self):
        scorer = TextStructureScorer()
        assert scorer.sentence_length_variance("Одно предложение без точки в конце") == 0.0

    def test_uniform_sentence_lengths_low_variance(self):
        scorer = TextStructureScorer()
        text = "Раз два три. Четыре пять шесть. Семь восемь девять."
        assert scorer.sentence_length_variance(text) == pytest.approx(0.0, abs=1e-9)

    def test_varied_sentence_lengths_higher_variance(self):
        scorer = TextStructureScorer()
        uniform = "Раз два три. Четыре пять шесть. Семь восемь девять."
        varied = (
            "Да. Мы обсуждали это довольно долго и не пришли к "
            "единому мнению о переносе сроков миграции. Хорошо."
        )
        assert scorer.sentence_length_variance(varied) > scorer.sentence_length_variance(
            uniform
        )


@pytest.mark.unit
class TestAnalyze:
    def test_returns_structure_signals(self):
        scorer = TextStructureScorer()
        text = "# Заголовок\n- пункт списка\nRoot Cause: сбой"
        signals = scorer.analyze(text)
        assert isinstance(signals, StructureSignals)
        assert signals.heading_density == pytest.approx(1 / 3)
        assert signals.list_density == pytest.approx(1 / 3)
        assert signals.field_density == pytest.approx(1 / 3)


@pytest.mark.unit
class TestScoreAndClassify:
    STRUCTURED_TEXT = (
        "Root Cause: конфигурация не применилась\n"
        "Resolution: откат конфигурации, перезапуск сервиса\n"
        "- проверить логи\n"
        "- подтвердить у дежурного"
    )

    UNSTRUCTURED_TEXT = (
        "Мы обсуждали миграцию довольно долго, и решение менялось "
        "несколько раз в зависимости от того, что говорил вендор. "
        "В итоге договорились зафиксировать финальный вариант завтра утром. Ок."
    )

    SEMI_STRUCTURED_TEXT = (
        "# Обзор миграции\n"
        "Здесь описан общий план миграции хранилища на несколько месяцев "
        "вперёд, включая обсуждение рисков и открытых вопросов, которые "
        "пока не решены.\n"
        "# Следующие шаги\n"
        "Команда обсудит детали на следующей неделе и обновит план по "
        "итогам встречи."
    )

    def test_structured_text_scores_high(self):
        scorer = TextStructureScorer()
        assert scorer.score(self.STRUCTURED_TEXT) >= 0.66
        assert scorer.classify(self.STRUCTURED_TEXT) == ContentStructureLevel.STRUCTURED

    def test_unstructured_text_scores_low(self):
        scorer = TextStructureScorer()
        assert scorer.score(self.UNSTRUCTURED_TEXT) < 0.33
        assert (
            scorer.classify(self.UNSTRUCTURED_TEXT) == ContentStructureLevel.UNSTRUCTURED
        )

    def test_semi_structured_text_in_middle_band(self):
        scorer = TextStructureScorer()
        score = scorer.score(self.SEMI_STRUCTURED_TEXT)
        assert 0.33 <= score < 0.66
        assert (
            scorer.classify(self.SEMI_STRUCTURED_TEXT)
            == ContentStructureLevel.SEMI_STRUCTURED
        )

    def test_score_is_bounded(self):
        scorer = TextStructureScorer()
        for text in (self.STRUCTURED_TEXT, self.UNSTRUCTURED_TEXT, "", "# А\n# Б\n# В"):
            score = scorer.score(text)
            assert 0.0 <= score <= 1.0


@pytest.mark.integration
class TestTextStructureScorerIntegration:
    """Требует реального корпуса живых страниц Confluence и тикетов Jira,
    чтобы проверить пороги 0.66/0.33 на распределении, а не на нескольких
    сконструированных примерах. В этом уроке не используется — заготовка
    для более поздних курсов, где пороги калибруются на статистике
    реального корпуса компании, а не подбираются вручную."""

    def test_thresholds_on_live_corpus_sample(self):
        pytest.skip("Требует выгрузки реального корпуса Confluence/Jira")
