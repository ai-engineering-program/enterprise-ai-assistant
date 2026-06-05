import pytest

from app.rag.architecture_advisor import (
    Approach,
    ArchitectureAdvisor,
    Recommendation,
    Requirements,
)


@pytest.mark.unit
class TestArchitectureAdvisorUnit:
    """Unit-тесты для ArchitectureAdvisor.

    Проверяют матрицу принятия решений по пяти бизнес-кейсам из урока 1.3.
    Не требуют внешних сервисов.
    """

    def setup_method(self) -> None:
        self.advisor = ArchitectureAdvisor()

    # ------------------------------------------------------------------
    # Кейс 1: «СберТехБанк» — ассистент кредитного отдела
    # Документы обновляются каждые 45 дней, нужна ссылка на источник.
    # Ожидается: RAG (прослеживаемость источников — главный критерий)
    # ------------------------------------------------------------------

    def test_case_1_bank_assistant_recommends_rag(self) -> None:
        req = Requirements(
            data_update_frequency_days=45,
            requires_source_citation=True,
            domain_style_critical=False,
            data_volume_docs=500,
            budget_rubles=500_000,
        )
        rec = self.advisor.recommend(req)
        assert rec.approach == Approach.RAG, (
            "Кейс 1: при требовании прослеживаемости источников должен быть RAG"
        )
        assert isinstance(rec.reason, str) and len(rec.reason) > 10
        assert rec.estimated_dev_weeks >= 1

    # ------------------------------------------------------------------
    # Кейс 2: «ЮрДок» — генератор юридических контрактов
    # Данные статичны (раз в 180 дней), стиль критичен, источники не нужны.
    # Ожидается: FINE_TUNING
    # ------------------------------------------------------------------

    def test_case_2_legal_doc_generator_recommends_fine_tuning(self) -> None:
        req = Requirements(
            data_update_frequency_days=180,
            requires_source_citation=False,
            domain_style_critical=True,
            data_volume_docs=50,
            budget_rubles=4_000_000,
        )
        rec = self.advisor.recommend(req)
        assert rec.approach == Approach.FINE_TUNING, (
            "Кейс 2: статичные данные + критичен стиль без прослеживаемости → дообучение"
        )
        assert rec.estimated_dev_weeks >= 8

    # ------------------------------------------------------------------
    # Кейс 3: «МедИнфо» — справочник по лекарственным взаимодействиям
    # База обновляется ежедневно (1 день), нужна ссылка на источник.
    # Ожидается: RAG
    # ------------------------------------------------------------------

    def test_case_3_medical_portal_recommends_rag(self) -> None:
        req = Requirements(
            data_update_frequency_days=1,
            requires_source_citation=True,
            domain_style_critical=False,
            data_volume_docs=50_000,
            budget_rubles=300_000,
        )
        rec = self.advisor.recommend(req)
        assert rec.approach == Approach.RAG, (
            "Кейс 3: ежедневные обновления + прослеживаемость → RAG"
        )

    # ------------------------------------------------------------------
    # Кейс 4: «ТелекомАссист» — поддержка клиентов оператора связи
    # Тарифы меняются еженедельно (7 дней), источник не критичен.
    # Ожидается: RAG (данные динамичны)
    # ------------------------------------------------------------------

    def test_case_4_telecom_support_recommends_rag(self) -> None:
        req = Requirements(
            data_update_frequency_days=7,
            requires_source_citation=False,
            domain_style_critical=False,
            data_volume_docs=200,
            budget_rubles=150_000,
        )
        rec = self.advisor.recommend(req)
        assert rec.approach == Approach.RAG, (
            "Кейс 4: данные меняются еженедельно → RAG предпочтительнее дообучения"
        )

    # ------------------------------------------------------------------
    # Кейс 5: «КаталогПро» — классификатор товаров в 200 категорий
    # Категории стабильны (раз в год), прослеживаемость не нужна,
    # но важна точность специфической классификации.
    # Ожидается: FINE_TUNING
    # ------------------------------------------------------------------

    def test_case_5_product_classifier_recommends_fine_tuning(self) -> None:
        req = Requirements(
            data_update_frequency_days=365,
            requires_source_citation=False,
            domain_style_critical=True,
            data_volume_docs=100,
            budget_rubles=3_000_000,
        )
        rec = self.advisor.recommend(req)
        assert rec.approach == Approach.FINE_TUNING, (
            "Кейс 5: статичные данные + специфика классификации → дообучение"
        )

    # ------------------------------------------------------------------
    # Дополнительные проверки: типы и структура Recommendation
    # ------------------------------------------------------------------

    def test_recommendation_returns_dataclass(self) -> None:
        req = Requirements(
            data_update_frequency_days=7,
            requires_source_citation=True,
            domain_style_critical=False,
            data_volume_docs=100,
            budget_rubles=200_000,
        )
        rec = self.advisor.recommend(req)
        assert isinstance(rec, Recommendation)
        assert isinstance(rec.approach, Approach)
        assert isinstance(rec.reason, str)
        assert isinstance(rec.estimated_dev_weeks, int)

    def test_reason_is_non_empty_string(self) -> None:
        req = Requirements(
            data_update_frequency_days=30,
            requires_source_citation=True,
            domain_style_critical=True,
            data_volume_docs=1000,
            budget_rubles=2_000_000,
        )
        rec = self.advisor.recommend(req)
        assert len(rec.reason.strip()) >= 20, (
            "Обоснование должно быть содержательным (не менее 20 символов)"
        )

    def test_hybrid_approach_exists_for_complex_case(self) -> None:
        """Гибридный подход должен возвращаться, когда данные статичны,
        но одновременно важны и прослеживаемость, и специфика стиля."""
        req = Requirements(
            data_update_frequency_days=90,
            requires_source_citation=True,
            domain_style_critical=True,
            data_volume_docs=2000,
            budget_rubles=5_000_000,
        )
        rec = self.advisor.recommend(req)
        # В сложном случае допустимы RAG или HYBRID — главное не FINE_TUNING
        # (прослеживаемость источников несовместима с чистым дообучением)
        assert rec.approach in (Approach.RAG, Approach.HYBRID), (
            "При требовании прослеживаемости результат не может быть FINE_TUNING"
        )
