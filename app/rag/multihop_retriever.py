from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


__all__ = [
    "SearchResult",
    "HopResult",
    "MultiHopResult",
    "EntityExtractor",
    "MockEntityExtractor",
    "MultiHopRetriever",
]


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Унифицированный результат поиска.

    Attributes:
        text:   текст документа (чанка).
        source: уникальный идентификатор документа.
        score:  оценка релевантности (выше — лучше).
    """

    text: str
    source: str
    score: float


@dataclass
class HopResult:
    """Результат одного шага итеративного поиска.

    Attributes:
        hop_number: номер шага, начиная с 1.
        subquery:   подзапрос, который был использован на этом шаге.
        documents:  список найденных документов.
        entities:   сущности, извлечённые для следующего шага.
        facts:      факты, накопленные для финального контекста.
    """

    hop_number: int
    subquery: str
    documents: list[SearchResult]
    entities: list[str]
    facts: list[str]


@dataclass
class MultiHopResult:
    """Финальный результат многошагового поиска.

    Attributes:
        hops:            все шаги поиска с деталями.
        final_context:   собранный контекст для генерации ответа LLM.
        total_documents: дедуплицированный список всех найденных документов.
        stop_reason:     причина остановки итерации.
    """

    hops: list[HopResult]
    final_context: str
    total_documents: list[SearchResult]
    stop_reason: str


# ---------------------------------------------------------------------------
# Протокол и реализации извлечения сущностей
# ---------------------------------------------------------------------------

class EntityExtractor(Protocol):
    """Протокол компонента извлечения сущностей из найденных документов."""

    def extract(
        self,
        original_query: str,
        current_subquery: str,
        documents: list[SearchResult],
        accumulated_facts: list[str],
    ) -> tuple[list[str], list[str], str]:
        """Извлекает сущности и факты, предлагает подсказку для следующего шага.

        Returns:
            (entities, facts, next_subquery_hint)
            entities: идентификаторы документов, коды, имена для следующего поиска.
            facts: короткие факты для накопления контекста.
            next_subquery_hint: подсказка для формирования следующего подзапроса.
        """
        ...


class MockEntityExtractor:
    """Детерминированный экстрактор сущностей для тестов (без вызова LLM).

    Ищет в тексте документов идентификаторы вида «Приказ №NN», «ИБ-YYYY-NN»,
    «ТР-СЛОВО-YYYY», «ФЗ-N», «ПП-N», «Проект-СЛОВО».

    Используйте только для юнит-тестов.
    В production замените на LLMEntityExtractor с реальным вызовом API.
    """

    _PATTERNS = [
        r'(?:Приказ|приказ)[- ]№?\s*(\d+)',         # Приказ №142
        r'([А-Я]{2,}-\d{4}-\d+)',                    # ИБ-2023-08
        r'([А-Я]{2,}-[А-Яа-яёЁ]+-\d{4})',           # ТР-ПДн-2024
        r'(?:ФЗ)-(\d+)',                              # ФЗ-152
        r'(?:ПП)-(\d+)',                              # ПП-658
        r'(?:Проект)-([А-Яа-яёЁA-Za-z]+)',           # Проект-DLP
    ]

    def extract(
        self,
        original_query: str,
        current_subquery: str,
        documents: list[SearchResult],
        accumulated_facts: list[str],
    ) -> tuple[list[str], list[str], str]:
        entities: list[str] = []
        facts: list[str] = []

        combined_text = " ".join(doc.text for doc in documents)

        for pattern in self._PATTERNS:
            matches = re.findall(pattern, combined_text)
            entities.extend(matches)

        # Дедупликация с сохранением порядка
        seen: dict[str, None] = {}
        for e in entities:
            seen[e] = None
        entities = list(seen.keys())

        # Извлечь первое предложение каждого документа как факт
        for doc in documents[:2]:
            first = doc.text.split(".")[0].strip()
            if first and first not in accumulated_facts:
                facts.append(first)

        # Подсказка: добавить найденные идентификаторы к исходному запросу
        hint = current_subquery
        if entities:
            hint = f"{original_query} {' '.join(entities[:3])}"

        return entities, facts, hint


# ---------------------------------------------------------------------------
# MultiHopRetriever
# ---------------------------------------------------------------------------

class MultiHopRetriever:
    """Итеративный поисковик для составных вопросов с цепочками документов.

    На каждой итерации:
    1. Планировщик формулирует подзапрос на основе накопленного контекста.
    2. Базовый retriever выполняет поиск.
    3. Экстрактор извлекает сущности и факты.
    4. Условие остановки решает — продолжать или собирать ответ.

    Пример:
        from app.rag.sparse_retriever import BM25Retriever

        docs = [...]  # список SearchResult-совместимых объектов
        base = BM25Retriever()
        base.index_documents(docs)

        extractor = MockEntityExtractor()
        mh = MultiHopRetriever(retriever=base, extractor=extractor, max_hops=3)
        result = mh.search("Какие меры после утечки данных в 2023?")
        print(result.stop_reason)    # "no_new_entities" | "max_hops_reached" | "no_documents"
        print(result.final_context)  # контекст для передачи в LLM
    """

    def __init__(
        self,
        retriever,
        extractor: EntityExtractor,
        top_k_per_hop: int = 3,
        min_new_entities: int = 1,
    ) -> None:
        # TODO: сохранить retriever в self.retriever
        # TODO: сохранить extractor в self.extractor
        # TODO: сохранить top_k_per_hop в self.top_k_per_hop
        # TODO: сохранить min_new_entities в self.min_new_entities
        ...

    def _plan_subquery(
        self,
        original_query: str,
        accumulated_facts: list[str],
        next_hint: str,
    ) -> str:
        """Формирует подзапрос для текущего шага итерации.

        TODO: реализуйте этот метод.
        Логика:
        - Если accumulated_facts пустой (первый шаг) — вернуть original_query.
        - Если next_hint не пустой — вернуть next_hint.
        - Иначе — вернуть original_query.
        """
        ...

    def search(
        self,
        query: str,
        max_hops: int = 3,
    ) -> MultiHopResult:
        """Выполняет итеративный поиск с условиями остановки.

        TODO: реализуйте этот метод.

        Алгоритм:
        1. Инициализировать состояние:
               hops: list[HopResult] = []
               all_docs: list[SearchResult] = []
               accumulated_facts: list[str] = []
               all_entities: set[str] = set()
               next_hint: str = query
               stop_reason: str = "max_hops_reached"

        2. Для hop_num в range(1, max_hops + 1):
               a. subquery = self._plan_subquery(query, accumulated_facts, next_hint)
               b. docs = self.retriever.retrieve(subquery, top_k=self.top_k_per_hop)
               c. Если docs пустой:
                      stop_reason = "no_documents"
                      break
               d. entities, facts, next_hint = self.extractor.extract(
                      query, subquery, docs, accumulated_facts)
               e. new_entities = set(entities) - all_entities
               f. all_entities.update(entities)
               g. accumulated_facts.extend(facts)
               h. hops.append(HopResult(hop_num, subquery, docs, entities, facts))
               i. all_docs.extend(docs)
               j. Если len(new_entities) < self.min_new_entities:
                      stop_reason = "no_new_entities"
                      break

        3. Дедуплицировать all_docs по source, сохранив экземпляр
           с наибольшим score.

        4. final_context = self._build_context(hops)

        5. Вернуть MultiHopResult(hops, final_context, deduped_docs, stop_reason)
        """
        ...

    def _build_context(self, hops: list[HopResult]) -> str:
        """Собирает финальный контекст из всех шагов поиска.

        TODO: реализуйте этот метод.

        Формат каждого раздела:
            === Шаг N: {subquery} ===
            [Источник: {source}] {text}
            [Источник: {source}] {text}

        Разделы разделяются двойным переводом строки (\\n\\n).
        Если hops пустой — вернуть пустую строку.
        """
        ...
