"""Tests for app/rag/retriever.py — RAGRetriever implementation (M2 L2)."""
from app.rag.retriever import RAGRetriever, RetrievedChunk, cosine_similarity, fake_embed

FAKE_INDEX = [
    {
        "text": "Сотрудникам полагается 28 календарных дней оплачиваемого отпуска в год.",
        "source": "hr_policy_ru.pdf",
        "embedding": fake_embed("отпуск календарные дни"),
    },
    {
        "text": "Employees in Germany are entitled to 20 working days of annual leave.",
        "source": "hr_policy_de.pdf",
        "embedding": fake_embed("annual leave germany working days"),
    },
    {
        "text": "Командировочные расходы возмещаются в течение 10 рабочих дней после подачи отчёта.",
        "source": "expense_policy.pdf",
        "embedding": fake_embed("командировка расходы возмещение"),
    },
    {
        "text": "Удалённая работа разрешена не более трёх дней в неделю по согласованию с руководителем.",
        "source": "remote_work_policy.pdf",
        "embedding": fake_embed("удалённая работа дни неделя"),
    },
]


def test_retrieve_returns_list():
    retriever = RAGRetriever(FAKE_INDEX)
    results = retriever.retrieve("отпуск", top_k=3)
    assert isinstance(results, list), f"retrieve должен возвращать list, получили: {type(results)}"
    assert len(results) <= 3, f"Не более top_k результатов, получили: {len(results)}"


def test_retrieve_sorted_by_score():
    retriever = RAGRetriever(FAKE_INDEX)
    results = retriever.retrieve("отпуск дни", top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Результаты должны быть отсортированы по убыванию score: {scores}"
    )


def test_retrieve_returns_retrieved_chunks():
    retriever = RAGRetriever(FAKE_INDEX)
    results = retriever.retrieve("командировка", top_k=2)
    for r in results:
        assert isinstance(r, RetrievedChunk), (
            f"Каждый элемент должен быть RetrievedChunk, получили: {type(r)}"
        )
        assert isinstance(r.text, str) and len(r.text) > 0
        assert isinstance(r.source, str) and len(r.source) > 0
        assert isinstance(r.score, float)


def test_retrieve_empty_index():
    retriever = RAGRetriever([])
    results = retriever.retrieve("любой запрос", top_k=3)
    assert results == [], f"Пустой индекс должен возвращать [], получили: {results}"


def test_top_k_larger_than_index():
    retriever = RAGRetriever(FAKE_INDEX)
    results = retriever.retrieve("work", top_k=100)
    assert len(results) == len(FAKE_INDEX), (
        f"Если top_k > размера индекса, вернуть все документы. "
        f"Ожидали {len(FAKE_INDEX)}, получили {len(results)}"
    )


def test_build_context_format():
    retriever = RAGRetriever(FAKE_INDEX)
    chunks = [
        RetrievedChunk(text="Текст первого документа.", source="doc1.pdf", score=0.9),
        RetrievedChunk(text="Текст второго документа.", source="doc2.pdf", score=0.8),
    ]
    context = retriever.build_context(chunks)
    assert "[Документ 1]" in context, f"Ожидали '[Документ 1]' в контексте: {context}"
    assert "[Документ 2]" in context, f"Ожидали '[Документ 2]' в контексте: {context}"
    assert "doc1.pdf" in context, f"Ожидали 'doc1.pdf' в контексте: {context}"
    assert "Текст первого документа." in context


def test_cosine_similarity_basic():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    score = cosine_similarity(a, b)
    assert abs(score - 1.0) < 1e-6, f"Идентичные векторы: ожидали 1.0, получили {score}"

    c = [1.0, 0.0, 0.0]
    d = [0.0, 1.0, 0.0]
    score2 = cosine_similarity(c, d)
    assert abs(score2 - 0.0) < 1e-6, f"Ортогональные векторы: ожидали 0.0, получили {score2}"
