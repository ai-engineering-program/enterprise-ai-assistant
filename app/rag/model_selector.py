import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BenchmarkResult:
    model_name: str
    recall_at_5: float      # полнота@5: доля запросов, у которых правильный документ в топ-5
    avg_latency_sec: float  # средняя задержка построения векторных представлений (сек / запрос)
    model_type: str         # "openai" или "local"


class ModelBenchmark:
    """Сравнивает несколько моделей векторных представлений по полноте@5 и задержке.

    Задание: реализуйте методы evaluate() и compare().
    Описание задания: exercise_1.html в папке lessons/2_2/.
    """

    def __init__(
        self,
        test_documents: list[str],
        test_queries: list[str],
        ground_truth: list[int],
    ) -> None:
        """
        Args:
            test_documents: список текстовых документов для индексации
            test_queries: список поисковых запросов
            ground_truth: ground_truth[i] — индекс правильного документа для queries[i]
        """
        # TODO: сохранить аргументы в self._documents, self._queries, self._ground_truth
        ...

    def _build_vectors_openai(self, texts: list[str], model_name: str) -> np.ndarray:
        """Строит векторные представления через OpenAI API (openai>=1.0).

        Используйте:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(input=texts, model=model_name)
            return np.array([item.embedding for item in response.data])
        """
        # TODO: реализовать вызов OpenAI API
        ...

    def _build_vectors_local(self, texts: list[str], model_name: str) -> np.ndarray:
        """Строит векторные представления через sentence-transformers.

        Используйте:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            return model.encode(texts, normalize_embeddings=True)
        """
        # TODO: реализовать загрузку и инференс локальной модели
        ...

    def evaluate(self, model_config: dict) -> BenchmarkResult:
        """Оценивает одну конфигурацию модели.

        Args:
            model_config: словарь вида {"name": "BAAI/bge-m3", "type": "local"}
                          или {"name": "text-embedding-3-small", "type": "openai"}

        Returns:
            BenchmarkResult с полнотой@5 и средней задержкой.
            При ошибке API возвращает BenchmarkResult с recall_at_5=0.0.

        Алгоритм:
            1. Засечь время t0
            2. Построить векторные представления для self._documents и self._queries
            3. Вычислить elapsed = time.perf_counter() - t0
            4. avg_latency = elapsed / len(self._queries)
            5. Вычислить матрицу косинусного сходства query × docs
            6. Для каждого запроса i: если self._ground_truth[i] входит в
               топ-5 индексов по сходству — это «попадание»
            7. recall_at_5 = hits / len(self._queries)
        """
        # TODO: реализовать evaluate()
        ...

    def compare(self, model_configs: list[dict]) -> list[BenchmarkResult]:
        """Запускает evaluate() для каждой конфигурации.

        Returns:
            Список BenchmarkResult, отсортированный по убыванию recall_at_5.
        """
        # TODO: вызвать evaluate для каждой конфигурации,
        #       собрать результаты и отсортировать по убыванию recall_at_5
        ...
