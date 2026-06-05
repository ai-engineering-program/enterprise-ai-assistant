import math


__all__ = ["cosine_similarity", "get_embedding", "semantic_test"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Вычислить косинусное сходство между двумя векторами.

    Формула: (A · B) / (||A|| * ||B||)

    Args:
        a: первый вектор (список float)
        b: второй вектор (список float, той же размерности)

    Returns:
        Значение в диапазоне [-1.0, 1.0].
        Возвращает 0.0, если любой из векторов нулевой.
    """
    # TODO: реализовать скалярное произведение (dot product)
    # TODO: вычислить L2-нормы (math.sqrt(sum(x*x for x in v)))
    # TODO: вернуть dot / (norm_a * norm_b), обработать деление на ноль
    ...


def get_embedding(text: str) -> list[float]:
    """Получить векторное представление текста.

    Используйте один из вариантов:
    - Локально: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
    - API: OpenAI embeddings (text-embedding-3-small)

    Args:
        text: входной текст (строка)

    Returns:
        Список float — вектор фиксированной размерности.
        Возвращает вектор из нулей размерности 1, если текст пустой.
    """
    # TODO: обработать пустую строку — вернуть [0.0]
    # TODO: реализовать получение векторного представления через выбранную библиотеку или API
    ...


def semantic_test(text1: str, text2: str) -> dict:
    """Вычислить семантическую близость двух текстов.

    Возвращает числовое значение косинусного сходства и текстовую метку:
    - "высокое"  если score > 0.8
    - "среднее"  если 0.5 <= score <= 0.8
    - "низкое"   если score < 0.5

    Args:
        text1: первый текст
        text2: второй текст

    Returns:
        Словарь {"score": float, "label": str}.
        Если любой из текстов пустой, возвращает {"score": 0.0, "label": "низкое"}.
    """
    # TODO: вызвать get_embedding для text1 и text2
    # TODO: вызвать cosine_similarity на полученных векторах
    # TODO: определить метку по порогам (0.8 и 0.5)
    # TODO: вернуть словарь {"score": ..., "label": ...}
    ...
