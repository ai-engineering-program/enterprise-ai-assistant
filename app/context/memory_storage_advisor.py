from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


__all__ = [
    "StorageRecommendation",
    "FieldProfile",
    "FieldClassification",
    "MigrationSignals",
    "MigrationDiagnosis",
    "DEFAULT_VOLATILE_THRESHOLD_DAYS",
    "DEFAULT_SLOW_CHANGE_THRESHOLD_DAYS",
    "MemoryStorageAdvisor",
]


class StorageRecommendation(Enum):
    """
    Куда следует поместить поле знания — рекомендация верхнего уровня,
    предшествующая любой конкретной реализации хранилища (см. текст урока
    5.5, раздел "таксономия знания по волатильности"). Это НЕ выбор между
    технологиями ("векторная БД" или "SQL") — это архитектурное решение о
    МОДЕЛИ хранения.
    """

    # Стабильный справочник: регламенты, инструкции, тарифные условия как
    # текст. Хорошо ложится на семантический поиск — редко меняется и не
    # привязан к одной конкретной сущности.
    DOCUMENT_STORE = "document_store"

    # То же, что DOCUMENT_STORE, но значение меняется достаточно часто
    # (месяцы, не годы), чтобы недостаточно было проиндексировать один
    # раз — нужен отдельный сигнал свежести (см. PageTrustScorer, урок
    # 2.3) поверх обычного retrieval.
    DOCUMENT_STORE_WITH_FRESHNESS_MONITOR = "document_store_with_freshness_monitor"

    # Волатильный атрибут КОНКРЕТНОЙ сущности (клиент, сотрудник, тикет).
    # Не подлежит семантическому поиску по определению задачи — нужен
    # точный fact_key + журнал версий (FactJournal, урок 5.4), а не
    # similarity search.
    FACT_JOURNAL = "fact_journal"

    # Живой оперативный поток (телеметрия, длина очереди, статус сенсора
    # в реальном времени) — сознательно ВНЕ систем памяти ассистента (см.
    # текст урока 5.5, раздел про границу с курсом 5 "AI Routing"): такие
    # данные не "запоминаются", а запрашиваются live через отдельный
    # вызов инструмента / мониторинговую систему.
    OUT_OF_SCOPE_OPERATIONAL = "out_of_scope_operational"


@dataclass(frozen=True)
class FieldProfile:
    """
    Метаданные ОДНОГО поля знания из корпуса — то, что аудит существующего
    RAG-корпуса должен собрать перед классификацией (см. текст урока,
    раздел "чек-лист миграции", шаг 1-2).
    """

    field_name: str
    # Среднее число дней между двумя изменениями значения ЭТОГО поля.
    # None означает "в проверенной истории изменений не зафиксировано" —
    # трактуется как практически статичный справочник.
    update_frequency_days: Optional[float] = None
    # True, если у значения поля есть отдельное значение НА КАЖДУЮ
    # сущность (клиент, сотрудник, тикет), а не одно общее значение для
    # всех записей этого типа (например, тарифный план компании — общий,
    # а персональный менеджер клиента — свой у каждого).
    entity_scoped: bool = False
    # True, если поле отражает живое, ежесекундно/ежеминутно меняющееся
    # оперативное состояние (телеметрия, текущая длина очереди), а не
    # состояние, о котором имеет смысл спрашивать постфактум.
    is_real_time_operational: bool = False


@dataclass(frozen=True)
class FieldClassification:
    """Результат классификации одного поля."""

    field_name: str
    recommendation: StorageRecommendation
    rationale: str


@dataclass(frozen=True)
class MigrationSignals:
    """
    Три диагностических сигнала из текста урока (раздел "диагностика:
    признаки неверного выбора хранилища") для ОДНОГО поля/типа вопроса,
    собранные из production-метрик или ручного разбора инцидентов.
    """

    # Точность ответа заметно ниже именно на срезе вопросов об этом
    # атрибуте сущности, чем в среднем по остальным типам вопросов.
    accuracy_drop_on_entity_queries: bool = False
    # Один и тот же вопрос про одну и ту же сущность в разных вызовах
    # получает разные ответы, хотя индекс между вызовами не менялся.
    non_deterministic_same_entity: bool = False
    # Корректное значение физически присутствует в проиндексированном
    # корпусе, но система систематически выбирает не его, а другую,
    # более старую версию (инцидент урока 5.4).
    correct_version_exists_but_wrong_surfaces: bool = False


@dataclass(frozen=True)
class MigrationDiagnosis:
    """Итог диагностики: стоит ли считать поле кандидатом на миграцию в FactJournal."""

    should_migrate: bool
    matched_signals: list[str]


# Порог волатильности: если значение поля, привязанного к конкретной
# сущности, в среднем меняется чаще, чем раз в столько дней, — это
# кандидат на FACT_JOURNAL, а не на документное хранилище. 90 дней —
# отправная точка (см. текст урока), а не универсальная константа.
DEFAULT_VOLATILE_THRESHOLD_DAYS: float = 90.0

# Порог "медленно меняющейся политики": значение, не привязанное к одной
# сущности, но обновляемое чаще, чем раз в этот период, требует
# отдельного мониторинга свежести поверх обычного document store.
DEFAULT_SLOW_CHANGE_THRESHOLD_DAYS: float = 365.0


class MemoryStorageAdvisor:
    """
    Модуль M5, урок 5.5 (синтез модуля): архитектурный слой ПЕРЕД любым
    конкретным хранилищем — отвечает не "как хранить факт" (это уже
    решено в FactJournal, урок 5.4, и LongTermMemoryStore, урок 5.2), а
    "в какое из уже существующих хранилищ вообще должно попасть ЭТО поле
    корпуса".

    MemoryStorageAdvisor НЕ хранит ни одного факта и не дублирует логику
    FactJournal или LongTermMemoryStore — он классифицирует метаданные
    поля (FieldProfile) и выдаёт рекомендацию (FieldClassification), а
    также умеет диагностировать по production-сигналам (MigrationSignals),
    стоит ли уже проиндексированное как документ поле переносить в
    FactJournal (см. текст урока, раздел "чек-лист миграции").
    """

    def __init__(
        self,
        volatile_threshold_days: float = DEFAULT_VOLATILE_THRESHOLD_DAYS,
        slow_change_threshold_days: float = DEFAULT_SLOW_CHANGE_THRESHOLD_DAYS,
    ) -> None:
        self._volatile_threshold_days = volatile_threshold_days
        self._slow_change_threshold_days = slow_change_threshold_days

    def classify_field(self, profile: FieldProfile) -> FieldClassification:
        """
        Классифицировать ОДНО поле по таксономии волатильности из текста
        урока (раздел "таксономия знания по волатильности").

        TODO, проверять условия СТРОГО в этом порядке:
        1. Если profile.is_real_time_operational is True -> вернуть
           FieldClassification с рекомендацией
           StorageRecommendation.OUT_OF_SCOPE_OPERATIONAL и обоснованием,
           объясняющим, что живой оперативный поток — не задача систем
           памяти ассистента (проверяется ПЕРВОЙ, независимо от
           entity_scoped/update_frequency_days).
        2. Если profile.update_frequency_days is None -> вернуть
           StorageRecommendation.DOCUMENT_STORE (стабильный справочник —
           истории изменений не зафиксировано).
        3. Если profile.entity_scoped is True И
           profile.update_frequency_days <= self._volatile_threshold_days
           -> вернуть StorageRecommendation.FACT_JOURNAL (волатильный
           атрибут конкретной сущности — см. урок 5.4).
        4. Если profile.update_frequency_days <=
           self._slow_change_threshold_days -> вернуть
           StorageRecommendation.DOCUMENT_STORE_WITH_FRESHNESS_MONITOR
           (медленно меняющаяся политика, общая для всех сущностей).
        5. Иначе -> вернуть StorageRecommendation.DOCUMENT_STORE.

        Обоснование (rationale) должно быть содержательной строкой,
        объясняющей ИМЕННО эту рекомендацию для ИМЕННО этого профиля
        (например, упоминать частоту обновления и entity_scoped) — не
        общей фразой без опоры на входные данные.
        """
        ...

    def classify_corpus(
        self, profiles: list[FieldProfile]
    ) -> dict[str, FieldClassification]:
        """
        Пакетная версия classify_field для всего аудита корпуса сразу
        (см. текст урока, раздел "чек-лист миграции", шаг аудита).

        TODO: вернуть {profile.field_name: self.classify_field(profile)
        for profile in profiles}. Если в profiles встречаются повторяющиеся
        field_name, последний профиль в списке побеждает (обычное
        поведение словаря при построении через dict/comprehension) — это
        осознанное поведение, не баг.
        """
        ...

    def migration_candidates(
        self, profiles: list[FieldProfile]
    ) -> list[FieldClassification]:
        """
        Из полного аудита корпуса вернуть только те поля, для которых
        classify_field рекомендовал StorageRecommendation.FACT_JOURNAL —
        то есть поля, которые ПРЯМО СЕЙЧАС могут быть в document store по
        ошибке и являются кандидатами на миграцию (см. текст урока,
        раздел "чек-лист миграции").

        TODO: вернуть список FieldClassification из
        self.classify_corpus(profiles).values(), у которых recommendation
        == StorageRecommendation.FACT_JOURNAL. Порядок результата — как в
        classify_corpus (порядок словаря Python 3.7+, совпадающий с
        порядком первого появления field_name в profiles).
        """
        ...

    def diagnose_wrong_storage(
        self, signals: MigrationSignals
    ) -> MigrationDiagnosis:
        """
        Диагностировать по production-сигналам, стоит ли считать текущий
        выбор хранилища ошибочным (см. текст урока, раздел "диагностика:
        признаки неверного выбора хранилища").

        TODO:
        1. matched = список имён (str) сработавших сигналов из
           signals.accuracy_drop_on_entity_queries,
           signals.non_deterministic_same_entity,
           signals.correct_version_exists_but_wrong_surfaces —
           используйте ИМЕННО имена полей датакласса как строки
           ("accuracy_drop_on_entity_queries" и т.д.), проверяя их в
           этом порядке.
        2. should_migrate:
           - True, если signals.correct_version_exists_but_wrong_surfaces
             is True само по себе (это прямое, самодостаточное
             подтверждение — см. инцидент урока 5.4: обе версии факта
             физически в индексе, similarity не может выбрать актуальную).
           - Иначе True, если СРАЗУ ОБА
             signals.accuracy_drop_on_entity_queries И
             signals.non_deterministic_same_entity истинны (два более
             слабых по отдельности сигнала, но убедительных вместе).
           - Иначе False (одного слабого сигнала недостаточно, чтобы
             утверждать "хранилище выбрано неверно" — возможны другие
             причины, см. диагностическое дерево в тексте урока).
        3. Вернуть MigrationDiagnosis(should_migrate=...,
           matched_signals=matched).
        """
        ...
