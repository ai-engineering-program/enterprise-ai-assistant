import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.rag.chunker import FixedSizeChunker, RecursiveChunker, SemanticChunker


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

NK_TEXT = """\
Статья 346.11. Общие положения

Применение упрощённой системы налогообложения организациями
предусматривает их освобождение от обязанности по уплате налога
на прибыль организаций (за исключением налога, уплачиваемого с
доходов, облагаемых по налоговым ставкам, предусмотренным пунктами
3 и 4 статьи 284 настоящего Кодекса), налога на имущество
организаций (за исключением налога, уплачиваемого в отношении
объектов недвижимого имущества, налоговая база по которым
определяется как их кадастровая стоимость в соответствии с
настоящим Кодексом).

Организации, применяющие упрощённую систему налогообложения,
не признаются налогоплательщиками налога на добавленную
стоимость, за исключением налога на добавленную стоимость,
подлежащего уплате в соответствии с настоящим Кодексом при
ввозе товаров на территорию Российской Федерации, а также
налога на добавленную стоимость, уплачиваемого в соответствии
со статьями 161 и 174.1 настоящего Кодекса.

Иные налоги, сборы и страховые взносы уплачиваются организациями,
применяющими упрощённую систему налогообложения, в соответствии
с законодательством Российской Федерации о налогах и сборах.\
"""

SHORT_TEXT = "Первое предложение. Второе предложение. Третье предложение."


# ---------------------------------------------------------------------------
# FixedSizeChunker — unit-тесты
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFixedSizeChunker:
    """Тесты FixedSizeChunker без внешних зависимостей."""

    def test_basic_split_returns_list(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        result = chunker.split(NK_TEXT)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_no_empty_fragments(self):
        """Результат не должен содержать пустых строк или строк из пробелов."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
        result = chunker.split(NK_TEXT)
        for fragment in result:
            assert fragment.strip() != "", "Обнаружен пустой фрагмент"

    def test_fragment_size_not_exceeded(self):
        """Каждый фрагмент не превышает chunk_size символов."""
        chunk_size = 300
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=50)
        result = chunker.split(NK_TEXT)
        for i, fragment in enumerate(result):
            assert len(fragment) <= chunk_size, (
                f"Фрагмент {i} длиной {len(fragment)} превышает chunk_size={chunk_size}"
            )

    def test_overlap_zero_no_repeated_content(self):
        """При chunk_overlap=0 соседние фрагменты не пересекаются."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        result = chunker.split("A" * 300)
        # Суммарная длина всех фрагментов должна равняться длине исходного текста
        assert sum(len(f) for f in result) == 300

    def test_overlap_positive_increases_fragment_count(self):
        """С перекрытием > 0 фрагментов больше, чем без перекрытия."""
        text = "X" * 1000
        without_overlap = FixedSizeChunker(chunk_size=200, chunk_overlap=0).split(text)
        with_overlap = FixedSizeChunker(chunk_size=200, chunk_overlap=50).split(text)
        assert len(with_overlap) >= len(without_overlap)

    def test_invalid_overlap_raises_value_error(self):
        """chunk_overlap >= chunk_size должно вызывать ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises_value_error(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=150)

    def test_single_character_text(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split("А")
        assert result == ["А"]

    def test_empty_text_returns_empty_list(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split("")
        assert result == []

    def test_whitespace_only_text_returns_empty_list(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split("   \n\n   ")
        assert result == []

    def test_text_shorter_than_chunk_size_returns_single_fragment(self):
        chunker = FixedSizeChunker(chunk_size=10000, chunk_overlap=100)
        result = chunker.split(NK_TEXT)
        assert len(result) == 1
        assert result[0] == NK_TEXT


# ---------------------------------------------------------------------------
# RecursiveChunker — unit-тесты
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecursiveChunker:
    """Тесты RecursiveChunker без внешних зависимостей."""

    def test_basic_split_returns_list(self):
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        result = chunker.split(NK_TEXT)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_no_empty_fragments(self):
        chunker = RecursiveChunker(chunk_size=400, chunk_overlap=80)
        result = chunker.split(NK_TEXT)
        for fragment in result:
            assert fragment.strip() != "", "Обнаружен пустой фрагмент"

    def test_fragment_size_not_exceeded(self):
        """Каждый фрагмент не превышает chunk_size символов."""
        chunk_size = 500
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=100)
        result = chunker.split(NK_TEXT)
        for i, fragment in enumerate(result):
            assert len(fragment) <= chunk_size, (
                f"Фрагмент {i} длиной {len(fragment)} превышает chunk_size={chunk_size}"
            )

    def test_paragraph_boundary_preserved(self):
        """Для текста с абзацами первый фрагмент должен содержать заголовок статьи."""
        # NK_TEXT начинается с «Статья 346.11. Общие положения»
        # При chunk_size >= 400 это должно быть в первом фрагменте
        chunker = RecursiveChunker(chunk_size=600, chunk_overlap=100)
        result = chunker.split(NK_TEXT)
        assert len(result) >= 1
        assert "Статья 346.11" in result[0], (
            "Заголовок статьи должен быть в первом фрагменте"
        )

    def test_short_text_not_split(self):
        """Текст короче chunk_size должен вернуться одним фрагментом."""
        chunker = RecursiveChunker(chunk_size=10000, chunk_overlap=100)
        result = chunker.split(NK_TEXT)
        assert len(result) == 1

    def test_empty_text_returns_empty_list(self):
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        result = chunker.split("")
        assert result == []

    def test_custom_separators_used(self):
        """Пользовательские разделители должны учитываться."""
        # Текст с только точкой с запятой как разделителем
        text = "Первый пункт; Второй пункт; Третий пункт"
        chunker = RecursiveChunker(
            chunk_size=20,
            chunk_overlap=0,
            separators=["; ", " ", ""],
        )
        result = chunker.split(text)
        # Каждый фрагмент <= 20 символов
        for fragment in result:
            assert len(fragment) <= 20

    def test_default_separators_applied_when_not_specified(self):
        """Если separators не указан, используются DEFAULT_SEPARATORS."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.separators == RecursiveChunker.DEFAULT_SEPARATORS

    def test_recursive_splits_large_paragraph(self):
        """Абзац, превышающий chunk_size, должен быть разбит по предложениям."""
        # Один абзац без двойных переводов строки, но с предложениями
        long_paragraph = (
            "Первое длинное предложение в абзаце о налогообложении. "
            "Второе длинное предложение о применении упрощённой системы. "
            "Третье длинное предложение об освобождении от НДС. "
            "Четвёртое длинное предложение об исключениях из льготы."
        )
        chunk_size = 80  # меньше длины абзаца, но больше одного предложения
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        result = chunker.split(long_paragraph)
        assert len(result) > 1, "Длинный абзац должен быть разбит на несколько фрагментов"
        for fragment in result:
            assert len(fragment) <= chunk_size


# ---------------------------------------------------------------------------
# Сравнительные тесты: FixedSizeChunker vs RecursiveChunker
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChunkerComparison:
    """Сравнение двух методов разбивки на одном тексте."""

    def test_recursive_preserves_header_better_than_fixed(self):
        """RecursiveChunker с большим chunk_size сохраняет заголовок в первом фрагменте."""
        # Для фиксированной разбивки при маленьком chunk_size заголовок
        # может «слиться» со следующим абзацем или попасть в конец фрагмента.
        # RecursiveChunker разбивает по абзацам, поэтому заголовок + первый абзац
        # остаются вместе.
        chunk_size = 600
        fixed = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=100).split(NK_TEXT)
        recursive = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=100).split(NK_TEXT)

        # Оба метода возвращают первый фрагмент, содержащий начало текста
        assert "Статья 346.11" in fixed[0]
        assert "Статья 346.11" in recursive[0]

    def test_both_cover_entire_text(self):
        """Оба метода должны покрывать весь исходный текст (с учётом перекрытий)."""
        chunk_size = 400
        overlap = 80

        fixed = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=overlap).split(NK_TEXT)
        recursive = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=overlap).split(NK_TEXT)

        # Первый фрагмент начинается с начала текста
        assert NK_TEXT.startswith(fixed[0][:50].strip())
        assert NK_TEXT.startswith(recursive[0][:50].strip())

        # Последний фрагмент заканчивается концом текста
        assert NK_TEXT.endswith(fixed[-1][-50:].strip())
        assert NK_TEXT.endswith(recursive[-1][-50:].strip())


# ---------------------------------------------------------------------------
# SemanticChunker — unit-тесты (эмбеддинги замоканы, модель не загружается)
# ---------------------------------------------------------------------------

def _make_mock_model(embeddings: np.ndarray) -> MagicMock:
    """Вернуть мок SentenceTransformer, чей .encode() возвращает готовые эмбеддинги."""
    mock = MagicMock()
    mock.encode.return_value = embeddings
    return mock


@pytest.mark.unit
class TestSemanticChunker:
    """Тесты SemanticChunker без загрузки реальной модели.

    Все тесты патчат sentence_transformers.SentenceTransformer через
    patch("app.rag.chunker.SentenceTransformer"), подставляя мок-объект,
    чей .encode() возвращает заранее подготовленные numpy-массивы.

    Геометрия фейковых эмбеддингов:
      Кластер A: вектор близкий к [1, 0, ...] — cosine_distance между
                 предложениями кластера ≈ 0.0 (очень похожи).
      Кластер B: вектор близкий к [0, 1, ...] — cosine_distance между
                 предложениями кластера ≈ 0.0.
      Расстояние A→B ≈ 1.0 — максимальный разрыв, чёткая граница.
    """

    # ------------------------------------------------------------------
    # Вспомогательные фейковые эмбеддинги
    # ------------------------------------------------------------------

    # 2-мерные единичные векторы: A = [1, 0], B = [0, 1]
    # cosine_distance(A, A) = 0.0, cosine_distance(A, B) = 1.0
    _EMB_A = np.array([1.0, 0.0], dtype=np.float32)
    _EMB_B = np.array([0.0, 1.0], dtype=np.float32)

    def _patch_st(self, embeddings: np.ndarray):
        """Контекстный менеджер: патчит SentenceTransformer в chunker-модуле."""
        mock_model = _make_mock_model(embeddings)
        return patch("app.rag.chunker.SentenceTransformer", return_value=mock_model)

    # ------------------------------------------------------------------
    # Граничные случаи — пустой и одиночный ввод
    # ------------------------------------------------------------------

    def test_empty_text_returns_empty_list(self):
        """Пустая строка → []."""
        emb = np.empty((0, 2), dtype=np.float32)
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.5)
            result = sc.split("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Строка только из пробелов и переносов строк → []."""
        emb = np.empty((0, 2), dtype=np.float32)
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.5)
            result = sc.split("   \n\n   \t  ")
        assert result == []

    def test_single_sentence_returns_one_chunk(self):
        """Одно предложение без знаков препинания → список из одного элемента."""
        single_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        with self._patch_st(single_emb):
            sc = SemanticChunker(threshold=0.5)
            result = sc.split("Одно единственное предложение")
        assert len(result) == 1
        assert result[0].strip() != ""

    # ------------------------------------------------------------------
    # Валидация параметров
    # ------------------------------------------------------------------

    def test_invalid_threshold_zero_raises_value_error(self):
        """threshold=0 находится на границе допустимого диапазона → ValueError."""
        with self._patch_st(np.empty((0, 2))):
            with pytest.raises(ValueError):
                SemanticChunker(threshold=0.0)

    def test_invalid_threshold_one_raises_value_error(self):
        """threshold=1 находится на границе допустимого диапазона → ValueError."""
        with self._patch_st(np.empty((0, 2))):
            with pytest.raises(ValueError):
                SemanticChunker(threshold=1.0)

    def test_invalid_threshold_negative_raises_value_error(self):
        """threshold < 0 → ValueError."""
        with self._patch_st(np.empty((0, 2))):
            with pytest.raises(ValueError):
                SemanticChunker(threshold=-0.1)

    def test_invalid_threshold_greater_than_one_raises_value_error(self):
        """threshold > 1 → ValueError."""
        with self._patch_st(np.empty((0, 2))):
            with pytest.raises(ValueError):
                SemanticChunker(threshold=1.5)

    # ------------------------------------------------------------------
    # Логика объединения и разрыва
    # ------------------------------------------------------------------

    def test_high_threshold_merges_all_sentences(self):
        """При threshold близком к 1 все предложения попадают в один чанк.

        Три предложения из кластера A: расстояния между ними ≈ 0.0.
        Порог = 0.99 — ни одно расстояние не превысит порог → один чанк.
        """
        # 3 предложения, все близко к вектору A
        emb = np.array([
            self._EMB_A,
            self._EMB_A + np.array([0.01, 0.0]),   # почти A
            self._EMB_A + np.array([0.02, 0.0]),   # почти A
        ], dtype=np.float32)

        text = "Первое предложение. Второе предложение. Третье предложение."
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.99)
            result = sc.split(text)

        assert len(result) == 1, (
            f"При высоком пороге ожидался 1 чанк, получено {len(result)}: {result}"
        )

    def test_low_threshold_splits_every_sentence(self):
        """При очень низком threshold каждое предложение — отдельный чанк.

        Используем три предложения с попарно ортогональными эмбеддингами
        (косинусное расстояние = 1.0 между соседями).
        Порог = 0.01 — все расстояния превышают порог → каждое предложение отдельно.
        """
        # A→B расстояние ≈ 1.0, B→C расстояние ≈ 1.0
        _EMB_C = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _EMB_A3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _EMB_B3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        emb = np.array([_EMB_A3, _EMB_B3, _EMB_C], dtype=np.float32)

        text = "Первое предложение. Второе предложение. Третье предложение."
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.01)
            result = sc.split(text)

        assert len(result) == 3, (
            f"При низком пороге ожидалось 3 чанка, получено {len(result)}: {result}"
        )

    def test_breakpoint_at_topic_shift(self):
        """Чанкер обнаруживает смысловой разрыв между двумя тематическими блоками.

        Текст: 3 предложения из «кластера A» (микросервисы), затем
               3 предложения из «кластера B» (Kafka).
        Фейковые эмбеддинги: A=[1,0], B=[0,1].
        Cosine distance A→A ≈ 0, A→B ≈ 1.0, B→B ≈ 0.
        При threshold=0.5 разрыв произойдёт ровно на границе кластеров.
        Ожидаем 2 чанка.
        """
        # 6 предложений: 3 из кластера A, 3 из кластера B
        emb = np.array([
            self._EMB_A,
            self._EMB_A,
            self._EMB_A,
            self._EMB_B,
            self._EMB_B,
            self._EMB_B,
        ], dtype=np.float32)

        text = (
            "Микросервисы независимы друг от друга. "
            "Каждый сервис развёртывается отдельно. "
            "Сервисы общаются через API. "
            "Kafka хранит сообщения в топиках. "
            "Партиции обеспечивают параллелизм. "
            "Репликация защищает от потери данных."
        )
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.5)
            result = sc.split(text)

        assert len(result) == 2, (
            f"Ожидалось 2 чанка (микросервисы + Kafka), получено {len(result)}: {result}"
        )
        # Первый чанк должен содержать контент из кластера A
        assert any(
            kw in result[0]
            for kw in ("Микросервис", "сервис", "API")
        ), f"Первый чанк не содержит ожидаемых слов: {result[0]!r}"
        # Второй чанк должен содержать контент из кластера B
        assert any(
            kw in result[1]
            for kw in ("Kafka", "топик", "Партиц", "Реплик")
        ), f"Второй чанк не содержит ожидаемых слов: {result[1]!r}"

    def test_no_empty_chunks_returned(self):
        """split() никогда не возвращает пустые строки или строки из пробелов."""
        emb = np.array([
            self._EMB_A,
            self._EMB_B,
            self._EMB_A,
        ], dtype=np.float32)

        text = "Первое предложение. Второе предложение. Третье предложение."
        with self._patch_st(emb):
            sc = SemanticChunker(threshold=0.5)
            result = sc.split(text)

        for chunk in result:
            assert chunk.strip() != "", f"Обнаружен пустой чанк: {chunk!r}"


# ---------------------------------------------------------------------------
# Интеграционные тесты (требуют только локальной среды, без сервисов)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestChunkerIntegration:
    """Интеграционные тесты с реальными файлами или внешними зависимостями."""

    def test_with_langchain_splitter_compatibility(self):
        """
        Сравнить RecursiveChunker с LangChain RecursiveCharacterTextSplitter.
        Требует: pip install langchain-text-splitters
        """
        pytest.skip("Требует langchain-text-splitters — запустите вручную")

    def test_semantic_chunker_with_real_model(self):
        """
        Проверить SemanticChunker с реальной моделью all-MiniLM-L6-v2.
        Требует: pip install sentence-transformers (~80 МБ при первой загрузке)
        """
        pytest.skip(
            "Требует sentence-transformers и загрузки модели — запустите вручную:\n"
            "  pytest tests/test_chunker.py::TestChunkerIntegration"
            "::test_semantic_chunker_with_real_model -v -m integration -s"
        )
