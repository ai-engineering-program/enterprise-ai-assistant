import pytest

from app.context.message_quality import ChatMessage, MessageQualityScorer


@pytest.mark.unit
class TestIsNoise:
    def test_pure_reaction_is_noise(self):
        scorer = MessageQualityScorer()
        assert scorer.is_noise(ChatMessage("Игорь", "👍", "10:05")) is True

    @pytest.mark.parametrize("phrase", ["+1", "окей", "OK", "Спасибо", "круто"])
    def test_noise_phrases_case_insensitive(self, phrase):
        scorer = MessageQualityScorer()
        assert scorer.is_noise(ChatMessage("Игорь", phrase, "10:05")) is True

    def test_empty_text_is_noise(self):
        scorer = MessageQualityScorer()
        assert scorer.is_noise(ChatMessage("Игорь", "   ", "10:05")) is True

    def test_very_short_text_is_noise(self):
        scorer = MessageQualityScorer()
        assert scorer.is_noise(ChatMessage("Игорь", "ок", "10:05")) is True

    def test_substantive_message_is_not_noise(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Игорь", "Для бэкапов ПДн предлагаю AES-128-CBC", "10:03")
        assert scorer.is_noise(msg) is False


@pytest.mark.unit
class TestHasUnresolvedReference:
    def test_deictic_marker_with_short_message_is_unresolved(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "фиксируем как договорились, го", "14:40")
        assert scorer.has_unresolved_reference(msg) is True

    def test_no_marker_is_resolved(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "фиксируем: GCM, 256", "14:40")
        assert scorer.has_unresolved_reference(msg) is False

    def test_marker_present_but_message_long_is_resolved(self):
        scorer = MessageQualityScorer()
        long_text = (
            "как договорились утром на созвоне, финально используем алгоритм "
            "AES в режиме GCM с длиной ключа 256 бит для бэкапов персональных данных"
        )
        msg = ChatMessage("Вячеслав", long_text, "14:40")
        assert scorer.has_unresolved_reference(msg) is False

    def test_case_insensitive_marker_match(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "Как Договорились, го", "14:40")
        assert scorer.has_unresolved_reference(msg) is True


@pytest.mark.unit
class TestScore:
    def test_noise_message_scores_zero(self):
        scorer = MessageQualityScorer()
        assert scorer.score(ChatMessage("Игорь", "👍", "10:05")) == 0.0

    def test_substantive_normal_length_message_scores_full(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage(
            "Игорь", "Для бэкапов ПДн предлагаю алгоритм AES-128-CBC", "10:03"
        )
        assert scorer.score(msg) == pytest.approx(1.0)

    def test_short_message_without_deixis_gets_length_penalty(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "фиксируем: GCM, 256", "14:40")
        assert scorer.score(msg) == pytest.approx(0.8)

    def test_short_deictic_message_gets_both_penalties(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "как договорились, го", "14:40")
        assert scorer.score(msg) == pytest.approx(0.4)

    def test_score_never_below_zero(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage("Вячеслав", "как", "14:40")
        assert scorer.score(msg) >= 0.0

    def test_score_never_above_one(self):
        scorer = MessageQualityScorer()
        msg = ChatMessage(
            "Игорь",
            "Для бэкапов ПДн предлагаю использовать надёжный алгоритм AES-128-CBC сегодня же",
            "10:03",
        )
        assert scorer.score(msg) <= 1.0


@pytest.mark.unit
class TestFilterUsable:
    def test_filters_out_low_score_messages(self):
        scorer = MessageQualityScorer()
        messages = [
            ChatMessage("Игорь", "Для бэкапов ПДн предлагаю алгоритм AES-128-CBC", "10:03"),
            ChatMessage("Игорь", "👍", "10:05"),
            ChatMessage("Марина", "Уточнила — точно нужен AES-256, режим GCM", "10:14"),
        ]
        result = scorer.filter_usable(messages, min_score=0.5)
        assert [m.text for m in result] == [
            "Для бэкапов ПДн предлагаю алгоритм AES-128-CBC",
            "Уточнила — точно нужен AES-256, режим GCM",
        ]

    def test_preserves_original_order(self):
        scorer = MessageQualityScorer()
        messages = [
            ChatMessage("A", "содержательное сообщение номер один здесь", "10:00"),
            ChatMessage("B", "содержательное сообщение номер два тоже здесь", "10:01"),
        ]
        result = scorer.filter_usable(messages, min_score=0.5)
        assert [m.author for m in result] == ["A", "B"]

    def test_empty_input_returns_empty_list(self):
        scorer = MessageQualityScorer()
        assert scorer.filter_usable([], min_score=0.5) == []


@pytest.mark.integration
class TestMessageQualityIntegration:
    """Требует реального Slack API для потоковой оценки живых тредов
    в проде (пагинация истории треда, разрешение упомянутых пользователей).
    В этом уроке не используется — заготовка для более поздних курсов."""

    def test_score_live_slack_thread(self):
        pytest.skip("Требует настроенного Slack-коннектора")
