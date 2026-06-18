import pytest

from app.rag.context_builder import ContextChunk, ContextBuilder, reorder_context_for_attention


def make_chunks(n: int) -> list[ContextChunk]:
    """Вспомогательная функция: создаёт n фрагментов с убывающим score."""
    return [
        ContextChunk(
            text=f"Фрагмент {chr(65 + i)}",
            score=round(1.0 - i * 0.1, 1),
            source=f"doc_{i + 1}.pdf",
            metadata={},
        )
        for i in range(n)
    ]


@pytest.mark.unit
class TestReorderContextForAttention:
    """Тесты функции reorder_context_for_attention."""

    def test_empty_list_returns_empty(self):
        result = reorder_context_for_attention([])
        assert result == []

    def test_single_chunk_unchanged(self):
        chunks = make_chunks(1)
        result = reorder_context_for_attention(chunks)
        assert len(result) == 1
        assert result[0].source == "doc_1.pdf"

    def test_best_chunk_at_position_zero(self):
        chunks = make_chunks(5)
        result = reorder_context_for_attention(chunks)
        assert result[0].source == "doc_1.pdf", (
            "Фрагмент с наибольшим score должен быть на позиции 0"
        )

    def test_second_best_chunk_at_last_position(self):
        chunks = make_chunks(5)
        result = reorder_context_for_attention(chunks)
        assert result[-1].source == "doc_2.pdf", (
            "Второй по score фрагмент должен быть на последней позиции"
        )

    def test_worst_chunk_in_middle(self):
        chunks = make_chunks(5)
        result = reorder_context_for_attention(chunks)
        middle_idx = len(result) // 2
        assert result[middle_idx].source == "doc_5.pdf", (
            "Фрагмент с наименьшим score должен быть в середине"
        )

    def test_output_length_equals_input_length(self):
        for n in [2, 3, 4, 6, 7]:
            chunks = make_chunks(n)
            result = reorder_context_for_attention(chunks)
            assert len(result) == n, f"Длина результата должна быть {n}, получили {len(result)}"

    def test_output_contains_all_input_chunks(self):
        chunks = make_chunks(6)
        result = reorder_context_for_attention(chunks)
        result_sources = {c.source for c in result}
        input_sources = {c.source for c in chunks}
        assert result_sources == input_sources, "Все фрагменты должны присутствовать в результате"

    def test_unsorted_input_is_sorted_first(self):
        """Функция должна сортировать по score, не полагаясь на порядок входа."""
        chunks = [
            ContextChunk(text="C", score=0.5, source="low.pdf", metadata={}),
            ContextChunk(text="A", score=0.9, source="high.pdf", metadata={}),
            ContextChunk(text="B", score=0.7, source="mid.pdf", metadata={}),
        ]
        result = reorder_context_for_attention(chunks)
        assert result[0].source == "high.pdf"

    def test_two_chunks_order(self):
        chunks = make_chunks(2)
        result = reorder_context_for_attention(chunks)
        assert result[0].source == "doc_1.pdf"
        assert result[1].source == "doc_2.pdf"


@pytest.mark.unit
class TestContextBuilder:
    """Тесты класса ContextBuilder."""

    def test_empty_chunks_returns_empty_string(self):
        builder = ContextBuilder()
        result = builder.build([])
        assert result == ""

    def test_max_chunks_limits_output(self):
        chunks = make_chunks(10)
        builder = ContextBuilder(max_chunks=3, reorder=False)
        result = builder.build(chunks)
        assert "[Документ 1]" in result
        assert "[Документ 3]" in result
        assert "[Документ 4]" not in result

    def test_document_labels_present(self):
        chunks = make_chunks(3)
        builder = ContextBuilder(max_chunks=3, reorder=False)
        result = builder.build(chunks)
        for i in range(1, 4):
            assert f"[Документ {i}]" in result, f"Метка [Документ {i}] должна присутствовать"

    def test_source_field_present(self):
        chunks = make_chunks(2)
        builder = ContextBuilder(max_chunks=2, reorder=False)
        result = builder.build(chunks)
        assert "Источник:" in result

    def test_no_reorder_preserves_descending_score_order(self):
        chunks = make_chunks(3)
        builder = ContextBuilder(max_chunks=3, reorder=False)
        result = builder.build(chunks)
        idx_a = result.index("Фрагмент A")
        idx_b = result.index("Фрагмент B")
        idx_c = result.index("Фрагмент C")
        assert idx_a < idx_b < idx_c, (
            "Без переупорядочивания фрагменты должны идти по убыванию score"
        )

    def test_reorder_places_best_first(self):
        chunks = make_chunks(5)
        builder = ContextBuilder(max_chunks=5, reorder=True)
        result = builder.build(chunks)
        # После переупорядочивания "Фрагмент A" (лучший) должен быть первым
        assert result.index("Фрагмент A") < result.index("Фрагмент C")

    def test_max_chunks_applied_before_reorder(self):
        """Обрезка происходит до переупорядочивания — берём топ по score."""
        chunks = make_chunks(10)
        builder = ContextBuilder(max_chunks=3, reorder=True)
        result = builder.build(chunks)
        # Фрагменты D–J не должны попасть в контекст
        for letter in "DEFGHIJ":
            assert f"Фрагмент {letter}" not in result

    def test_single_chunk_no_crash(self):
        chunks = make_chunks(1)
        builder = ContextBuilder(max_chunks=5, reorder=True)
        result = builder.build(chunks)
        assert "Фрагмент A" in result
        assert "[Документ 1]" in result
