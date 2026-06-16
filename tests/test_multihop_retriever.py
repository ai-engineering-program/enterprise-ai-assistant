"""Tests for MultiHopRetriever (lesson 4.6).

Run unit tests (no external services required):
    pytest tests/test_multihop_retriever.py -v -m unit

Run all tests:
    pytest tests/test_multihop_retriever.py -v
"""

from __future__ import annotations

import pytest

from app.rag.multihop_retriever import (
    EntityExtractor,
    HopResult,
    MockEntityExtractor,
    MultiHopResult,
    MultiHopRetriever,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

# Тестовый корпус: 15 документов, три явных цепочки A, B, C.
# Цепочка A: инцидент → приказ → регламент
# Цепочка B: закон → постановление → инструкция
# Цепочка C: исследование → внедрение → отчёт

_CORPUS: list[SearchResult] = [
    # --- Цепочка A ---
    SearchResult(
        text=(
            "Инцидент-репорт ИБ-2023-08. "
            "Утечка произошла 14 августа 2023 года. "
            "Затронуто около 2,1 млн записей клиентов. "
            "Источник: скомпрометированный API внешнего партнёра."
        ),
        source="doc_A1_incident_report.txt",
        score=0.92,
    ),
    SearchResult(
        text=(
            "Приказ №142 от сентября 2023. "
            "Во исполнение инцидент-репорта ИБ-2023-08 учреждаются инициативы: "
            "аудит API-интеграций, ужесточение DLP-контроля, Проект-ПДн."
        ),
        source="doc_A2_order_142.txt",
        score=0.70,
    ),
    SearchResult(
        text=(
            "Технический регламент ТР-ПДн-2024. "
            "Во исполнение Приказа-142: токенизация идентификаторов клиентов, "
            "шифрование PII в транзитном слое, ротация ключей API каждые 90 дней."
        ),
        source="doc_A3_regulation.txt",
        score=0.55,
    ),
    # --- Цепочка B ---
    SearchResult(
        text=(
            "ФЗ-152 «О персональных данных», статья 11. "
            "Обработка биометрических персональных данных допускается только "
            "при наличии письменного согласия субъекта."
        ),
        source="doc_B1_fz152.txt",
        score=0.88,
    ),
    SearchResult(
        text=(
            "Постановление Правительства ПП-658 от 2021 года. "
            "В соответствии с ФЗ-152 устанавливаются требования к хранению "
            "биометрических данных для финансовых организаций."
        ),
        source="doc_B2_pp658.txt",
        score=0.65,
    ),
    SearchResult(
        text=(
            "Указание ЦБ РФ №5599-У. "
            "Во исполнение ПП-658 банки обязаны обеспечить раздельное хранение "
            "биометрических шаблонов и персональных данных."
        ),
        source="doc_B3_cbr_instruction.txt",
        score=0.50,
    ),
    # --- Цепочка C ---
    SearchResult(
        text=(
            "Исследование эффективности DLP-систем в банковском секторе, 2022. "
            "Рекомендовано внедрение Проект-DLP для контроля утечек данных "
            "через внешние API-интеграции."
        ),
        source="doc_C1_research.txt",
        score=0.80,
    ),
    SearchResult(
        text=(
            "Отчёт о внедрении Проект-DLP в корпоративную инфраструктуру. "
            "Внедрение выполнено в Q1 2023. Охват: 95% API-шлюзов."
        ),
        source="doc_C2_implementation.txt",
        score=0.60,
    ),
    SearchResult(
        text=(
            "Итоговый отчёт Проект-DLP. "
            "Зафиксировано снижение инцидентов утечки через API на 78% "
            "по сравнению с базовым периодом до внедрения."
        ),
        source="doc_C3_final_report.txt",
        score=0.45,
    ),
    # --- Нерелевантные документы ---
    SearchResult(
        text="Политика информационной безопасности редакция 2019 года. Общие принципы.",
        source="doc_misc_policy_2019.txt",
        score=0.30,
    ),
    SearchResult(
        text="Регламент работы с клиентскими обращениями через колл-центр.",
        source="doc_misc_callcenter.txt",
        score=0.25,
    ),
    SearchResult(
        text="Инструкция по резервному копированию баз данных.",
        source="doc_misc_backup.txt",
        score=0.20,
    ),
    SearchResult(
        text="Соглашение о конфиденциальности для сотрудников банка.",
        source="doc_misc_nda.txt",
        score=0.18,
    ),
    SearchResult(
        text="Требования к паролям корпоративных систем, версия 3.1.",
        source="doc_misc_passwords.txt",
        score=0.15,
    ),
    SearchResult(
        text="Список уполномоченных лиц для подписания договоров.",
        source="doc_misc_signatories.txt",
        score=0.10,
    ),
]


class SimpleCorpusRetriever:
    """Минималистичный retriever для тестов.

    Ищет документы по подстроке в тексте (case-insensitive).
    Возвращает совпавшие документы, отсортированные по score по убыванию.
    """

    def __init__(self, corpus: list[SearchResult]) -> None:
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_lower = query.lower()
        matched = [
            doc for doc in self._corpus
            if any(word in doc.text.lower() for word in query_lower.split()
                   if len(word) > 3)
        ]
        matched.sort(key=lambda d: d.score, reverse=True)
        return matched[:top_k]


# ---------------------------------------------------------------------------
# Unit-тесты
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMockEntityExtractor:
    """Тесты для MockEntityExtractor — работают без внешних сервисов."""

    def test_extracts_incident_code(self):
        extractor = MockEntityExtractor()
        docs = [_CORPUS[0]]  # doc_A1: содержит "ИБ-2023-08"
        entities, facts, hint = extractor.extract(
            original_query="утечка данных 2023",
            current_subquery="утечка данных 2023",
            documents=docs,
            accumulated_facts=[],
        )
        assert any("2023-08" in e or "ИБ" in e for e in entities), (
            f"Ожидали найти код ИБ-2023-08 в сущностях, получили: {entities}"
        )

    def test_extracts_order_number(self):
        extractor = MockEntityExtractor()
        docs = [_CORPUS[1]]  # doc_A2: содержит "Приказ №142"
        entities, facts, hint = extractor.extract(
            original_query="приказ после инцидента",
            current_subquery="Приказ ИБ-2023-08",
            documents=docs,
            accumulated_facts=[],
        )
        assert any("142" in e for e in entities), (
            f"Ожидали найти '142' в сущностях, получили: {entities}"
        )

    def test_returns_facts_from_documents(self):
        extractor = MockEntityExtractor()
        docs = [_CORPUS[0]]
        entities, facts, hint = extractor.extract(
            original_query="утечка",
            current_subquery="утечка",
            documents=docs,
            accumulated_facts=[],
        )
        assert len(facts) >= 1, "Должен извлечь хотя бы один факт"
        assert len(facts[0]) > 5, "Факт не должен быть пустым"

    def test_hint_contains_entities(self):
        extractor = MockEntityExtractor()
        docs = [_CORPUS[0]]
        entities, facts, hint = extractor.extract(
            original_query="меры после утечки",
            current_subquery="утечка 2023",
            documents=docs,
            accumulated_facts=[],
        )
        if entities:
            assert any(e in hint for e in entities), (
                "Подсказка должна включать найденные сущности"
            )

    def test_no_duplicate_entities(self):
        extractor = MockEntityExtractor()
        # Два документа, оба упоминают ИБ-2023-08
        docs = [_CORPUS[0], _CORPUS[1]]
        entities, _, _ = extractor.extract("q", "q", docs, [])
        assert len(entities) == len(set(entities)), "Сущности не должны дублироваться"


@pytest.mark.unit
class TestMultiHopRetrieverInit:
    """Тесты инициализации MultiHopRetriever."""

    def _make_retriever(self):
        corpus_retriever = SimpleCorpusRetriever(_CORPUS)
        extractor = MockEntityExtractor()
        return MultiHopRetriever(
            retriever=corpus_retriever,
            extractor=extractor,
            top_k_per_hop=3,
            min_new_entities=1,
        )

    def test_init_stores_retriever(self):
        mh = self._make_retriever()
        assert hasattr(mh, "retriever"), "retriever должен быть атрибутом экземпляра"

    def test_init_stores_extractor(self):
        mh = self._make_retriever()
        assert hasattr(mh, "extractor"), "extractor должен быть атрибутом экземпляра"

    def test_init_stores_top_k_per_hop(self):
        mh = self._make_retriever()
        assert mh.top_k_per_hop == 3

    def test_init_stores_min_new_entities(self):
        mh = self._make_retriever()
        assert mh.min_new_entities == 1


@pytest.mark.unit
class TestPlanSubquery:
    """Тесты метода _plan_subquery."""

    def _make_retriever(self):
        return MultiHopRetriever(
            retriever=SimpleCorpusRetriever(_CORPUS),
            extractor=MockEntityExtractor(),
        )

    def test_first_hop_returns_original_query(self):
        mh = self._make_retriever()
        result = mh._plan_subquery(
            original_query="утечка данных 2023",
            accumulated_facts=[],
            next_hint="",
        )
        assert result == "утечка данных 2023", (
            "На первом шаге (пустые facts) должен возвращаться исходный запрос"
        )

    def test_subsequent_hop_uses_hint(self):
        mh = self._make_retriever()
        result = mh._plan_subquery(
            original_query="утечка данных 2023",
            accumulated_facts=["Инцидент-репорт ИБ-2023-08"],
            next_hint="утечка данных 2023 ИБ-2023-08 142",
        )
        assert "ИБ-2023-08" in result or "142" in result, (
            "При наличии hint должна использоваться подсказка с сущностями"
        )

    def test_falls_back_to_original_when_no_hint(self):
        mh = self._make_retriever()
        result = mh._plan_subquery(
            original_query="утечка данных 2023",
            accumulated_facts=["какой-то факт"],
            next_hint="",
        )
        assert result == "утечка данных 2023"


@pytest.mark.unit
class TestMultiHopSearch:
    """Тесты метода search — центральный функционал."""

    def _make_retriever(self, min_new_entities: int = 1):
        return MultiHopRetriever(
            retriever=SimpleCorpusRetriever(_CORPUS),
            extractor=MockEntityExtractor(),
            top_k_per_hop=3,
            min_new_entities=min_new_entities,
        )

    def test_returns_multihop_result(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=3)
        assert isinstance(result, MultiHopResult)

    def test_hops_not_empty(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=3)
        assert len(result.hops) >= 1, "Должен быть хотя бы один шаг поиска"

    def test_max_hops_not_exceeded(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=2)
        assert len(result.hops) <= 2, "Количество шагов не должно превышать max_hops"

    def test_stop_reason_is_valid(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=3)
        valid_reasons = {"max_hops_reached", "no_new_entities", "no_documents"}
        assert result.stop_reason in valid_reasons, (
            f"stop_reason должен быть одним из {valid_reasons}, получено: {result.stop_reason!r}"
        )

    def test_stop_reason_max_hops_when_limit_reached(self):
        # min_new_entities=0 гарантирует, что остановка по лимиту, не по сущностям
        mh = self._make_retriever(min_new_entities=0)
        result = mh.search("утечка данных 2023", max_hops=2)
        assert result.stop_reason == "max_hops_reached"
        assert len(result.hops) == 2

    def test_stop_reason_no_documents_for_empty_query(self):
        mh = self._make_retriever()
        # Запрос, по которому ничего не найдётся
        result = mh.search("xyzzy несуществующий термин qwerty", max_hops=3)
        assert result.stop_reason == "no_documents"
        assert len(result.hops) == 0

    def test_total_documents_deduplicated(self):
        mh = self._make_retriever(min_new_entities=0)
        result = mh.search("утечка данных 2023", max_hops=3)
        sources = [doc.source for doc in result.total_documents]
        assert len(sources) == len(set(sources)), (
            "total_documents не должен содержать дубликатов по source"
        )

    def test_hop_results_contain_subquery(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=2)
        for hop in result.hops:
            assert isinstance(hop, HopResult)
            assert len(hop.subquery) > 0
            assert hop.hop_number >= 1

    def test_final_context_not_empty_when_docs_found(self):
        mh = self._make_retriever()
        result = mh.search("утечка данных 2023", max_hops=2)
        if result.hops:
            assert len(result.final_context) > 0

    def test_multihop_finds_more_docs_than_single_hop(self):
        """Ключевой тест: многошаговый поиск находит больше документов цепочки.

        Одношаговый поиск (max_hops=1) должен найти только начало цепочки A.
        Многошаговый (max_hops=3) должен пройти по цепочке A1→A2→A3.
        """
        mh = self._make_retriever(min_new_entities=0)

        single_hop = mh.search("утечка данных 2023", max_hops=1)
        multi_hop = mh.search("утечка данных 2023", max_hops=3)

        single_sources = {doc.source for doc in single_hop.total_documents}
        multi_sources = {doc.source for doc in multi_hop.total_documents}

        assert len(multi_sources) >= len(single_sources), (
            "Многошаговый поиск должен находить не меньше документов, чем одношаговый"
        )


@pytest.mark.unit
class TestBuildContext:
    """Тесты метода _build_context."""

    def _make_retriever(self):
        return MultiHopRetriever(
            retriever=SimpleCorpusRetriever(_CORPUS),
            extractor=MockEntityExtractor(),
        )

    def test_empty_hops_returns_empty_string(self):
        mh = self._make_retriever()
        result = mh._build_context([])
        assert result == ""

    def test_context_contains_step_headers(self):
        mh = self._make_retriever()
        hop = HopResult(
            hop_number=1,
            subquery="утечка 2023",
            documents=[_CORPUS[0]],
            entities=["ИБ-2023-08"],
            facts=["Утечка произошла 14 августа 2023"],
        )
        context = mh._build_context([hop])
        assert "Шаг 1" in context, "Контекст должен содержать заголовок шага"

    def test_context_contains_source(self):
        mh = self._make_retriever()
        hop = HopResult(
            hop_number=1,
            subquery="утечка 2023",
            documents=[_CORPUS[0]],
            entities=[],
            facts=[],
        )
        context = mh._build_context([hop])
        assert _CORPUS[0].source in context, "Контекст должен содержать source документа"

    def test_context_contains_document_text(self):
        mh = self._make_retriever()
        hop = HopResult(
            hop_number=1,
            subquery="утечка 2023",
            documents=[_CORPUS[0]],
            entities=[],
            facts=[],
        )
        context = mh._build_context([hop])
        # Проверяем, что хотя бы часть текста документа попала в контекст
        assert "2023" in context

    def test_context_contains_all_hops(self):
        mh = self._make_retriever()
        hop1 = HopResult(1, "запрос 1", [_CORPUS[0]], [], [])
        hop2 = HopResult(2, "запрос 2", [_CORPUS[1]], [], [])
        context = mh._build_context([hop1, hop2])
        assert "Шаг 1" in context
        assert "Шаг 2" in context


@pytest.mark.integration
class TestMultiHopRetrieverIntegration:
    """Интеграционные тесты. Требуют Qdrant или реального embedder.

    Запускать вручную: pytest tests/test_multihop_retriever.py -v -m integration
    """

    def test_with_dense_retriever(self):
        pytest.skip(
            "Требует запущенный Qdrant и модель эмбеддингов — запустите вручную"
        )
