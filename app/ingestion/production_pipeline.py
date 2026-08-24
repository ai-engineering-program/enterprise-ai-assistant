from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.ingestion.data_contract import ContractViolation, DataContract, DocumentContractValidator
from app.ingestion.data_quality_report import (
    DataQualityMetrics,
    DataQualityReport,
    DataQualityThresholds,
    QualityViolation,
)
from app.ingestion.document_deduplicator import DocumentDeduplicator, DuplicateMatch
from app.ingestion.format_detector import DocumentFormat, detect_format
from app.ingestion.idempotent_delivery_guard import IdempotentDeliveryGuard
from app.ingestion.parsers.factory import get_parser
from app.ingestion.pii_sanitizer import PIISanitizer
from app.ingestion.pipeline import DEFAULT_VECTOR_SIZE, IngestionReport
from app.ingestion.pipeline_alerting import Alert, OperationalThresholds, PipelineAlertMonitor
from app.ingestion.retry_dead_letter import (
    DeadLetterQueue,
    PermanentIngestionError,
    RetryingProcessor,
    TransientIngestionError,
)
from app.ingestion.text_normalizer import TextNormalizer
from app.rag.chunker import FixedSizeChunker
from app.rag.embedding_utils import get_embedding
from app.rag.vector_store import VectorStore


__all__ = [
    "PipelineConfig",
    "DocumentOutcome",
    "ProductionRunReport",
    "ProductionIngestionPipeline",
]


@dataclass
class PipelineConfig:
    """Настройки одного развёртывания production-пайплайна поглощения.

    contract — data contract источника (урок 7.3). None означает
    "источник без явного контракта" — process_document() в этом случае
    пропускает шаг валидации метаданных целиком, а не молча считает
    любой документ валидным по умолчанию (см. __init__).
    """

    collection_name: str = "granit_invest_docs"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_size: int = DEFAULT_VECTOR_SIZE
    max_retry_attempts: int = 5
    min_scanned_warnings_for_ocr_flag: int = 1
    contract: Optional[DataContract] = None
    data_quality_thresholds: DataQualityThresholds = field(default_factory=DataQualityThresholds)
    operational_thresholds: OperationalThresholds = field(default_factory=OperationalThresholds)


@dataclass
class DocumentOutcome:
    """Итог process_document() для одного документа.

    status:
      "indexed"              — чанки записаны в векторное хранилище.
      "quarantined_contract" — документ не прошёл data contract (урок 7.3),
                               содержимое не парсилось вообще.
      "flagged_for_ocr"      — PDF без текстового слоя; передан на отдельный
                               OCR-конвейер раздела 3, здесь не индексируется.
      "duplicate"            — точный или приблизительный дубль уже
                               проиндексированного документа (урок 4.3).
    """

    doc_id: str
    status: str
    chunks_indexed: int = 0
    contract_violations: list[ContractViolation] = field(default_factory=list)
    duplicate_match: Optional[DuplicateMatch] = None
    pii_masked_count: int = 0


@dataclass
class ProductionRunReport:
    """Итог одного вызова ProductionIngestionPipeline.run() по батчу документов."""

    ingestion_report: IngestionReport
    quality_metrics: Optional[DataQualityMetrics]
    quality_violations: list[QualityViolation]
    dispatched_alerts: list[Alert]
    dead_letter_count: int
    contract_violation_count: int
    duplicate_count: int
    ocr_flagged_count: int


class ProductionIngestionPipeline:
    """Финальная сборка пайплайна поглощения холдинга «Гранит-Инвест».

    Синтезирует компоненты, построенные по одному в каждом разделе курса,
    в единый конвейер "документ на входе -> решение о его судьбе":

        раздел 7 (data contract)      -> DocumentContractValidator
        раздел 2 (парсинг форматов)   -> format_detector + parsers/factory
        раздел 3 (маршрутизация OCR)  -> сигнал "нет текстового слоя"
                                          из PDFParserAdapter.warnings
        раздел 4 (норм./дедуп./PII)   -> TextNormalizer, DocumentDeduplicator,
                                          PIISanitizer
        раздел 5/6 (события, retry)   -> IdempotentDeliveryGuard,
                                          RetryingProcessor + DeadLetterQueue
        раздел 7 (качество, алерты)   -> DataQualityReport, PipelineAlertMonitor

    Этот класс НЕ реализует ни одну из этих задач заново — он вызывает уже
    существующие методы существующих классов в правильном порядке и
    правильно интерпретирует их результаты. Если один из компонентов ниже
    у вас не реализован — соответствующий блок process_document() упадёт
    там же, где вы использовали бы этот компонент напрямую.

    Полный конвейер парсинга сканов (ImagePreprocessor -> pytesseract ->
    OCRConfidenceGate -> OCRQualityDiagnostics -> TextPostProcessor)
    сознательно НЕ встроен инлайн в process_document(): документы со
    статусом "flagged_for_ocr" уходят в отдельный, более медленный и
    дорогой конвейер (тот же аргумент, что и в истории урока 5.1 про
    одну медленную стадию, держащую весь пайплайн).
    """

    def __init__(
        self,
        *,
        chunker: FixedSizeChunker,
        vector_store: VectorStore,
        normalizer: TextNormalizer,
        deduplicator: DocumentDeduplicator,
        pii_sanitizer: PIISanitizer,
        quality_report: DataQualityReport,
        alert_monitor: PipelineAlertMonitor,
        delivery_guard: IdempotentDeliveryGuard,
        config: PipelineConfig,
        contract_validator: Optional[DocumentContractValidator] = None,
    ) -> None:
        # TODO: сохранить каждый именованный аргумент как атрибут экземпляра
        # с тем же именем (self.chunker = chunker, self.vector_store =
        # vector_store, ... self.contract_validator = contract_validator).
        # Ни один аргумент не создаётся здесь заново — все компоненты
        # приходят уже готовыми, как в ProductionRAGPipeline из курса 2
        # (см. app/rag/pipeline.py) — это упрощает unit-тесты: тесты
        # подставляют MagicMock вместо реальных компонентов.
        ...

    # ------------------------------------------------------------------
    # Один документ
    # ------------------------------------------------------------------
    def process_document(
        self, doc_id: str, path: Path, raw_metadata: dict[str, Any]
    ) -> DocumentOutcome:
        """Провести один документ через весь конвейер поглощения.

        TODO, строго в этом порядке:

        1. Если self.contract_validator не None:
           violations = self.contract_validator.validate_document(doc_id, raw_metadata)
           если violations не пуст -> вернуть DocumentOutcome(doc_id=doc_id,
               status="quarantined_contract", contract_violations=violations)
           (см. урок 7.3 — недоверенным метаданным источника нельзя давать
           дойти до парсинга содержимого)

        2. fmt = detect_format(path)

        3. try:
               parser = get_parser(fmt)
           except ValueError as exc:
               raise PermanentIngestionError(str(exc)) from exc
           (PARSER_REGISTRY из parsers/factory.py, урок 2.4 — источник
           истины о том, какие форматы реально поддержаны СЕЙЧАС; повтор
           попытки для неподдерживаемого формата никогда не поможет)

        4. try:
               parsed = parser.parse(path)
           except (OSError, UnicodeDecodeError) as exc:
               raise PermanentIngestionError(str(exc)) from exc
           except (TimeoutError, ConnectionError) as exc:
               raise TransientIngestionError(str(exc)) from exc

        5. Если fmt == DocumentFormat.PDF:
               scan_warnings = [w for w in parsed.warnings if "текстового слоя" in w]
               если len(scan_warnings) >= self.config.min_scanned_warnings_for_ocr_flag:
                   вернуть DocumentOutcome(doc_id=doc_id, status="flagged_for_ocr")
           (сигнал приходит из PDFParserAdapter.warnings, урок 2.2/2.4 —
           этот метод не разбирает PDF заново и не запускает OCR сам)

        6. text = self.normalizer.normalize(parsed.full_text)

        7. dup = self.deduplicator.register_document(doc_id, text)
           если dup.match_type != "unique":
               вернуть DocumentOutcome(doc_id=doc_id, status="duplicate",
                   duplicate_match=dup)

        8. clean_text, pii_matches = self.pii_sanitizer.sanitize(text)

        9. chunks = self.chunker.split(clean_text)
           если chunks пуст:
               вернуть DocumentOutcome(doc_id=doc_id, status="indexed",
                   chunks_indexed=0, duplicate_match=dup,
                   pii_masked_count=len(pii_matches))

        10. try:
                docs_to_upsert = [{
                    "id": self._make_point_id(doc_id, i),
                    "vector": get_embedding(chunk),
                    "metadata": {"text": chunk, "source": doc_id, "chunk_index": i},
                } for i, chunk in enumerate(chunks)]
                self.vector_store.upsert_documents(docs_to_upsert)
            except (TimeoutError, ConnectionError) as exc:
                raise TransientIngestionError(str(exc)) from exc

        11. вернуть DocumentOutcome(doc_id=doc_id, status="indexed",
                chunks_indexed=len(chunks), duplicate_match=dup,
                pii_masked_count=len(pii_matches))
        """
        ...

    def _make_point_id(self, doc_id: str, chunk_index: int) -> str:
        """Детерминированный id точки Qdrant для идемпотентного upsert.

        TODO: тот же приём, что MinimalIngestionPipeline.make_point_id
        (урок 1.4) и AsyncIngestionPipeline.make_point_id (урок 5.2):
        вернуть sha256(f"{doc_id}::{chunk_index}".encode("utf-8")).hexdigest()
        """
        ...

    # ------------------------------------------------------------------
    # Потоковый вход: одно событие изменения документа
    # ------------------------------------------------------------------
    def process_event(self, event: dict[str, Any], path: Path) -> Optional[DocumentOutcome]:
        """Точка входа для потокового источника (раздел 5).

        event ожидается в формате, совместимом с
        IdempotentDeliveryGuard.handle_delivery (урок 6.2): как минимум
        doc_id, опционально event_id, content, raw_metadata.

        TODO:
        1. status = self.delivery_guard.handle_delivery(event)
        2. Если status == "duplicate" -> вернуть None (эта ДОСТАВКА
           отброшена как повторная; process_document даже не вызывается)
        3. Иначе -> вернуть self.process_document(
               event["doc_id"], path, event.get("raw_metadata", {}))
        """
        ...

    # ------------------------------------------------------------------
    # Батч: полный прогон с retry, DLQ и метриками качества
    # ------------------------------------------------------------------
    def run(self, documents: list[dict[str, Any]]) -> ProductionRunReport:
        """Обработать батч документов с retry/DLQ и собрать отчёт о качестве.

        documents: список словарей {"doc_id": str, "path": Path,
        "raw_metadata": dict, "freshness_lag_seconds": float (опционально,
        по умолчанию 0.0)}.

        TODO:
        1. outcomes: dict[str, DocumentOutcome] = {}
           Это временное хранилище результатов ТЕКУЩЕГО run() — сохраните
           его как self._last_outcomes перед циклом, чтобы
           _process_for_retry() (шаг 3) мог туда писать.

        2. dlq = DeadLetterQueue()
           retrying = RetryingProcessor(process_fn=self._process_for_retry,
               max_attempts=self.config.max_retry_attempts,
               dead_letter_queue=dlq)

        3. ingestion_report = IngestionReport()
           Для каждого doc в documents:
               result = retrying.process_with_retry(doc["doc_id"], doc)
               если result == "dead_letter":
                   ingestion_report.files_skipped.append(doc["doc_id"])
                   продолжить со следующим документом (для него не будет
                   записи в self._last_outcomes — process_document так и
                   не вернул результат)
               иначе (result == "success"):
                   outcome = self._last_outcomes[doc["doc_id"]]
                   ingestion_report.files_processed.append(doc["doc_id"])
                   если outcome.status == "indexed":
                       ingestion_report.chunks_indexed += outcome.chunks_indexed
                   (документы в карантине по контракту/OCR/дубликату
                   тоже "обработаны" в смысле IngestionReport — пайплайн
                   принял по ним решение без сбоя; "skip" здесь означает
                   конкретно ТЕХНИЧЕСКИЙ провал, ушедший в DLQ)

        4. invalid_document_ids = {doc_id for doc_id, o in
               self._last_outcomes.items()
               if o.status in ("quarantined_contract", "flagged_for_ocr")}

        5. duplicate_matches = [o.duplicate_match for o in
               self._last_outcomes.values() if o.duplicate_match is not None]

        6. freshness_lag_seconds = [doc.get("freshness_lag_seconds", 0.0)
               for doc in documents]

        7. metrics = self.quality_report.build_report(
               ingestion_report=ingestion_report,
               total_documents=len(documents),
               invalid_document_ids=invalid_document_ids,
               freshness_lag_seconds=freshness_lag_seconds,
               duplicate_matches=duplicate_matches,
               dead_letter_count=dlq.size(),
           )
           quality_violations = self.quality_report.check_thresholds(metrics)

        8. all_contract_violations = [] — собрать
               o.contract_violations по всем o в self._last_outcomes.values()
               (сумма списков, а не только непустые)
           alerts = self.alert_monitor.from_quality_violations(quality_violations)
           alerts += self.alert_monitor.from_contract_violations(all_contract_violations)
           dispatched = self.alert_monitor.dispatch(alerts)

        9. вернуть ProductionRunReport(
               ingestion_report=ingestion_report,
               quality_metrics=metrics,
               quality_violations=quality_violations,
               dispatched_alerts=dispatched,
               dead_letter_count=dlq.size(),
               contract_violation_count=len([o for o in
                   self._last_outcomes.values()
                   if o.status == "quarantined_contract"]),
               duplicate_count=len([o for o in self._last_outcomes.values()
                   if o.status == "duplicate"]),
               ocr_flagged_count=len([o for o in self._last_outcomes.values()
                   if o.status == "flagged_for_ocr"]),
           )
        """
        ...

    def _process_for_retry(self, doc: dict[str, Any]) -> None:
        """Callable, передаваемый в RetryingProcessor.process_fn из run().

        TODO:
        1. outcome = self.process_document(doc["doc_id"], doc["path"],
               doc.get("raw_metadata", {}))
        2. self._last_outcomes[doc["doc_id"]] = outcome
        3. Ничего не возвращать (None). Успех для RetryingProcessor — это
           отсутствие поднятого исключения, независимо от бизнес-статуса
           outcome: "quarantined_contract" — тоже корректно обработанный
           документ, просто результат обработки — карантин, а не индекс.
           TransientIngestionError/PermanentIngestionError, поднятые
           ВНУТРИ process_document (шаги 3-4, 10), пробрасываются отсюда
           без перехвата — именно их видит RetryingProcessor снаружи.
        """
        ...

    # ------------------------------------------------------------------
    # Конструктор реального пайплайна из конфигурации
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, config: PipelineConfig) -> "ProductionIngestionPipeline":
        """Собрать пайплайн с реальными (не тестовыми) компонентами.

        TODO: создать и передать в конструктор:
        - chunker = FixedSizeChunker(chunk_size=config.chunk_size,
              chunk_overlap=config.chunk_overlap)
        - vector_store = VectorStore(host=config.qdrant_host,
              port=config.qdrant_port, collection_name=config.collection_name,
              vector_size=config.vector_size)
        - normalizer = TextNormalizer()
        - deduplicator = DocumentDeduplicator()
        - pii_sanitizer = PIISanitizer()
        - quality_report = DataQualityReport(thresholds=config.data_quality_thresholds)
        - alert_monitor = PipelineAlertMonitor(operational_thresholds=config.operational_thresholds)
        - delivery_guard = IdempotentDeliveryGuard()
        - contract_validator = (DocumentContractValidator(config.contract)
              if config.contract is not None else None)

        Вернуть cls(chunker=chunker, vector_store=vector_store,
            normalizer=normalizer, deduplicator=deduplicator,
            pii_sanitizer=pii_sanitizer, quality_report=quality_report,
            alert_monitor=alert_monitor, delivery_guard=delivery_guard,
            config=config, contract_validator=contract_validator)
        """
        ...
