from dataclasses import dataclass
from datetime import datetime


__all__ = ["StaleDocument", "StalenessDetector"]


@dataclass
class StaleDocument:
    doc_id: str
    last_modified: datetime
    staleness_seconds: float


class StalenessDetector:
    """Detects documents whose source has changed after the index was last updated.

    Used to identify which documents require re-indexing before the next
    search cycle, preventing stale retrieval results.
    """

    def __init__(self, threshold_seconds: int = 3600) -> None:
        # TODO: сохранить threshold_seconds в self.threshold_seconds
        ...

    def get_staleness_seconds(
        self,
        doc_last_modified: datetime,
        index_last_updated: datetime,
    ) -> float:
        """Return the lag in seconds between a document's modification time
        and the index's last update time.

        A positive value means the document was modified AFTER the index
        was last updated — i.e. it is a candidate for re-indexing.

        Args:
            doc_last_modified: when the source document was last changed
            index_last_updated: when the vector index was last rebuilt

        Returns:
            float — difference in seconds (doc_last_modified - index_last_updated)
        """
        # TODO: вернуть (doc_last_modified - index_last_updated).total_seconds()
        ...

    def detect_stale(
        self,
        doc_last_modified: datetime,
        index_last_updated: datetime,
    ) -> bool:
        """Return True if the document is stale (lag exceeds threshold).

        A document is stale when get_staleness_seconds() > self.threshold_seconds.
        Equality (lag == threshold) is NOT considered stale.

        Args:
            doc_last_modified: when the source document was last changed
            index_last_updated: when the vector index was last rebuilt

        Returns:
            bool — True if the document needs re-indexing
        """
        # TODO: использовать get_staleness_seconds и сравнить с self.threshold_seconds
        ...

    def get_stale_documents(
        self,
        documents: list[dict],
        index_last_updated: datetime,
    ) -> list[StaleDocument]:
        """Filter a list of documents and return only those that are stale.

        Each document dict must have keys:
            - "id": str — unique document identifier
            - "last_modified": datetime — when the source document was changed

        Args:
            documents: list of document metadata dicts
            index_last_updated: when the vector index was last rebuilt

        Returns:
            list of StaleDocument for each document whose staleness exceeds the threshold
        """
        # TODO: для каждого документа из documents:
        #   1. Прочитать doc["last_modified"] как datetime
        #   2. Вызвать detect_stale(last_modified, index_last_updated)
        #   3. Если True — создать StaleDocument(
        #          doc_id=doc["id"],
        #          last_modified=last_modified,
        #          staleness_seconds=get_staleness_seconds(last_modified, index_last_updated)
        #      ) и добавить в результирующий список
        # TODO: вернуть список StaleDocument
        ...
