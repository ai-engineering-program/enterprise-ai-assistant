from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


__all__ = [
    "FieldType",
    "FieldSpec",
    "DataContract",
    "ContractViolation",
    "DocumentContractValidator",
]


class FieldType(str, Enum):
    """Тип значения поля, который data contract ожидает от источника."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    ENUM = "enum"


@dataclass(frozen=True)
class FieldSpec:
    """
    Описание одного поля входного документа, которое источник обязан
    поставлять в согласованном виде.

    date_formats используется только для FieldType.DATETIME: набор
    strptime-шаблонов, допустимых ДЛЯ ЭТОЙ версии контракта. Несколько
    шаблонов одновременно — штатный механизм переходного периода при
    смене формата источником (см. текст урока, "Версионирование:
    переходное окно"), а не признак небрежно специфицированного
    контракта.

    max_future_skew_seconds — для FieldType.DATETIME: насколько далеко
    в будущее относительно момента проверки значение ещё считается
    правдоподобным. None означает "проверка на будущее не выполняется".
    """

    name: str
    field_type: FieldType
    required: bool = True
    allowed_values: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    date_formats: tuple[str, ...] = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S")
    max_future_skew_seconds: float | None = None


@dataclass(frozen=True)
class DataContract:
    """
    Версионированная схема ожидаемых входных документов одного
    источника — явное соглашение о структуре данных на границе
    пайплайна поглощения, а не молчаливое предположение "источник не
    изменится" (см. текст урока).

    version — версия контракта ЭТОГО источника (например "1.0", "2.0"),
    а не версия кода валидатора. Явно хранить version внутри самого
    контракта позволяет отличать "источник шлёт то, что мы ожидали
    год назад" от "источник шлёт то, что мы ожидаем сейчас" без
    археологии по git log (см. текст урока, "Версионирование").
    """

    source_name: str
    version: str
    fields: tuple[FieldSpec, ...]

    def field_by_name(self, name: str) -> FieldSpec | None:
        """Найти FieldSpec по имени поля.

        TODO: вернуть первый FieldSpec из self.fields, у которого
        f.name == name, либо None, если такого поля в контракте нет.
        """
        ...


@dataclass
class ContractViolation:
    """
    Одно нарушение data contract конкретным документом — источник
    данных и для явного отказа (карантин документа), и для алертов
    (см. app/ingestion/pipeline_alerting.py, урок 7.2).
    """

    document_id: str
    field: str
    violation_type: str  # "missing_required" | "wrong_type" | "out_of_range" | "unparseable_datetime"
    message: str


class DocumentContractValidator:
    """
    Проверяет входной документ (сырой dict — например, одна запись из
    JSON/CSV-экспорта источника, ДО парсинга содержимого и ДО
    нормализации) на соответствие DataContract.

    Валидатор НЕ пытается угадать намерение источника и НЕ подставляет
    значение по умолчанию при несоответствии — именно такая тихая
    подстановка (`except ValueError: published_at = datetime.utcnow()`)
    и превратила смену формата даты источником в незаметную порчу
    данных в истории этого урока. Любое нарушение возвращается как
    явный ContractViolation; что делать с документом, провалившим
    контракт (отбросить, отправить в карантин, всё же записать с
    флагом invalid), решает вызывающий код пайплайна — не сам
    валидатор.
    """

    def __init__(self, contract: DataContract) -> None:
        # TODO: self.contract = contract
        ...

    # ------------------------------------------------------------------
    # Проверка одного документа
    # ------------------------------------------------------------------
    def validate_document(
        self, document_id: str, raw_document: dict[str, Any]
    ) -> list[ContractViolation]:
        """
        Проверить один документ по всем полям self.contract.fields.

        TODO:
        1. violations: list[ContractViolation] = []
        2. Для каждого spec в self.contract.fields:
           a. Если spec.name отсутствует в raw_document ИЛИ
              raw_document[spec.name] is None:
              - если spec.required -> добавить в violations
                ContractViolation(document_id=document_id,
                field=spec.name, violation_type="missing_required",
                message=f"обязательное поле '{spec.name}' отсутствует")
              - иначе перейти к следующему полю (необязательное поле
                законно может отсутствовать)
           b. Иначе вызвать self._validate_field(document_id, spec,
              raw_document[spec.name]); если результат не None —
              добавить его в violations.
        3. Вернуть violations.
        """
        ...

    def _validate_field(
        self, document_id: str, spec: FieldSpec, value: Any
    ) -> ContractViolation | None:
        """
        Проверить одно присутствующее значение против одного FieldSpec.

        TODO — разобрать по spec.field_type:

        1. FieldType.STRING:
           если not isinstance(value, str) -> ContractViolation(...,
           violation_type="wrong_type",
           message=f"поле '{spec.name}' должно быть строкой, получено {type(value).__name__}").
           Иначе None.

        2. FieldType.INTEGER / FieldType.FLOAT:
           - bool — это подкласс int в Python, но НЕ допустимое
             числовое значение контракта: если isinstance(value, bool)
             -> сразу "wrong_type", не доходя до проверки диапазона.
           - если not isinstance(value, (int, float)) -> "wrong_type".
           - иначе: если spec.min_value задан и value < spec.min_value,
             или spec.max_value задан и value > spec.max_value ->
             violation_type="out_of_range" с message, указывающим
             фактическое значение и нарушенную границу.
           - иначе None.

        3. FieldType.ENUM:
           если spec.allowed_values задан и value not in
           spec.allowed_values -> violation_type="out_of_range"
           (значение вне допустимого множества, message должен
           перечислить allowed_values). Иначе None.

        4. FieldType.DATETIME:
           - если value не строка -> "wrong_type".
           - иначе parsed = self._parse_datetime(value, spec.date_formats).
             Если parsed is None -> violation_type="unparseable_datetime"
             (ни один шаблон из контракта не подошёл — самый частый
             симптом смены формата источником без предупреждения, см.
             текст урока).
           - если parsed не None и spec.max_future_skew_seconds задан:
             сравнить parsed с datetime.now(timezone.utc) (если parsed
             — naive, то есть parsed.tzinfo is None, сравнивать с
             datetime.now() без timezone, чтобы не поднять TypeError
             при сравнении aware/naive). Если parsed находится в
             будущем больше, чем на max_future_skew_seconds ->
             violation_type="out_of_range" (документ "из будущего" —
             типичный симптом неверно распарсенной даты).
           - иначе None.

        Вернуть None, если ни одна проверка не сработала.
        """
        ...

    @staticmethod
    def _parse_datetime(value: str, date_formats: tuple[str, ...]) -> datetime | None:
        """
        Попробовать разобрать value по каждому шаблону из date_formats
        по очереди, вернуть первый успешно распарсенный результат.

        TODO:
        1. Для fmt в date_formats:
           try:
               return datetime.strptime(value, fmt)
           except ValueError:
               продолжить со следующим шаблоном
        2. Если ни один шаблон не подошёл — вернуть None.
        """
        ...

    # ------------------------------------------------------------------
    # Проверка батча документов
    # ------------------------------------------------------------------
    def validate_batch(
        self, raw_documents: dict[str, dict[str, Any]]
    ) -> dict[str, list[ContractViolation]]:
        """
        Проверить несколько документов сразу.

        TODO:
        1. result: dict[str, list[ContractViolation]] = {}
        2. Для document_id, raw_document в raw_documents.items():
           violations = self.validate_document(document_id, raw_document)
           если violations не пуст -> result[document_id] = violations
        3. Вернуть result (документы без нарушений в результат не
           входят — вызывающему коду важны только те, что провалили
           контракт).
        """
        ...
