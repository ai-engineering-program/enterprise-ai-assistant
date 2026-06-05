import warnings
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IndexMetadata:
    model_name: str
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    collection_name: str = ""


class ModelRegistry:
    """Tracks which embedding model was used to build each vector index collection.

    Prevents silent quality degradation when the embedding model is updated
    without rebuilding the index.
    """

    def __init__(self) -> None:
        self._registry: dict[str, IndexMetadata] = {}

    def register(self, collection_name: str, model_name: str) -> IndexMetadata:
        """Register the model used to build a collection's index.

        Args:
            collection_name: name of the Qdrant collection
            model_name: name of the embedding model (e.g. "all-MiniLM-L6-v2")

        Returns:
            IndexMetadata with the registered information.
        """
        # TODO: создать IndexMetadata(model_name=model_name, collection_name=collection_name)
        # TODO: сохранить объект в self._registry под ключом collection_name
        # TODO: вернуть созданный объект
        ...

    def check_compatibility(self, collection_name: str, current_model: str) -> bool:
        """Check whether the current model matches the registered one.

        Args:
            collection_name: name of the Qdrant collection to check
            current_model: name of the model currently used for query encoding

        Returns:
            True if compatible (or collection unknown), False if mismatch detected.
            Emits warnings.warn when a mismatch is found.
        """
        # TODO: если collection_name нет в self._registry — вернуть True
        # TODO: получить metadata = self._registry[collection_name]
        # TODO: если metadata.model_name != current_model — вызвать:
        #       warnings.warn(
        #           f"Несовместимость модели для коллекции '{collection_name}': "
        #           f"индекс построен моделью '{metadata.model_name}', "
        #           f"текущая модель '{current_model}'. Требуется переиндексация.",
        #           UserWarning,
        #           stacklevel=2,
        #       )
        #       и вернуть False
        # TODO: вернуть True если модели совпадают
        ...
