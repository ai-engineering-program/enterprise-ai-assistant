"""Tests for app/ingestion/document_deduplicator.py (Lesson 4.3).

Everything here operates on plain Python strings and does not touch any
external service (no Qdrant, no embedding model), so the whole module is
covered by unit tests only.

Run:
    pytest tests/test_document_deduplicator.py -v -m unit
"""
from __future__ import annotations

import pytest

from app.ingestion.document_deduplicator import DocumentDeduplicator, DuplicateMatch


# Синтетический "регламент" длиной около 150 слов - специально длинный,
# чтобы точечная правка одной даты затронула только несколько shingles
# из полутора сотен, а не весь документ (см. текст урока 4.3, история
# про холдинг "Гранит-Инвест" и положение о закупках оборудования).
REGULATION_V1 = (
    "Настоящее положение утверждено приказом генерального директора холдинга и "
    "регулирует порядок проведения регламентных работ по техническому обслуживанию "
    "производственного оборудования на всех предприятиях группы компаний согласно "
    "утверждённому графику плановых осмотров и ремонтов каждой единицы техники. "
    "Ответственный исполнитель обязан составлять график осмотров не позднее пятого "
    "числа каждого месяца и передавать его на согласование главному инженеру "
    "предприятия в течение трёх рабочих дней с момента составления графика. "
    "Положение вступает в силу с 01 марта 2024 года и действует до момента "
    "утверждения новой редакции документа директоратом по производству холдинга. "
    "Все ранее изданные регламенты по данному вопросу признаются утратившими силу "
    "с даты вступления настоящего положения в действие без дополнительного "
    "уведомления структурных подразделений. Контроль за исполнением требований "
    "настоящего положения возлагается на службу главного механика и службу охраны "
    "труда каждого структурного подразделения холдинга совместно с отделом закупок. "
    "Нарушение установленного порядка проведения регламентных работ фиксируется "
    "актом и передаётся в комиссию по производственной дисциплине для рассмотрения "
    "в течение десяти рабочих дней с момента обнаружения нарушения установленного "
    "порядка проведения работ."
)

# Версия из Confluence: единственная правка - дата вступления в силу
# ("01 марта" -> "15 марта"), весь остальной текст побайтово идентичен.
REGULATION_V2_DATE_EDIT = REGULATION_V1.replace(
    "01 марта 2024 года", "15 марта 2024 года"
)

# Совершенно другой документ - инструкция об удалённом доступе, другая
# тема и другая лексика, ни одного совпадающего 5-словного фрагмента с
# REGULATION_V1.
UNRELATED_DOCUMENT = (
    "Настоящая инструкция определяет порядок предоставления удалённого доступа "
    "к корпоративным информационным системам сотрудникам структурных подразделений "
    "холдинга и подрядным организациям, привлечённым для выполнения проектных работ. "
    "Заявка на предоставление доступа оформляется руководителем подразделения и "
    "согласовывается службой информационной безопасности не позднее двух рабочих "
    "дней до планируемой даты начала работ. Доступ предоставляется исключительно "
    "через защищённый канал связи с обязательным использованием многофакторной "
    "аутентификации для каждой учётной записи без исключения. Срок действия "
    "предоставленного доступа не может превышать девяносто календарных дней без "
    "повторного согласования и продления заявки ответственным руководителем. "
    "Служба информационной безопасности ведёт журнал всех предоставленных доступов "
    "и проводит ежеквартальную проверку соответствия действующих учётных записей "
    "утверждённым заявкам структурных подразделений холдинга."
)


@pytest.mark.unit
class TestComputeContentHash:
    def test_identical_text_gives_identical_hash(self):
        dedup = DocumentDeduplicator()
        assert dedup.compute_content_hash(REGULATION_V1) == dedup.compute_content_hash(
            REGULATION_V1
        )

    def test_different_text_gives_different_hash(self):
        dedup = DocumentDeduplicator()
        h1 = dedup.compute_content_hash(REGULATION_V1)
        h2 = dedup.compute_content_hash(REGULATION_V2_DATE_EDIT)
        assert h1 != h2

    def test_hash_is_hex_sha256_length(self):
        dedup = DocumentDeduplicator()
        h = dedup.compute_content_hash("простой текст")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.unit
class TestShingle:
    def test_default_k_produces_overlapping_windows(self):
        dedup = DocumentDeduplicator()
        text = "один два три четыре пять шесть"
        shingles = dedup.shingle(text, k=5)
        assert shingles == {
            "один два три четыре пять",
            "два три четыре пять шесть",
        }

    def test_short_text_returns_whole_text_as_single_shingle(self):
        dedup = DocumentDeduplicator()
        text = "два слова"
        shingles = dedup.shingle(text, k=5)
        assert shingles == {text}

    def test_uses_shingle_size_constant_by_default(self):
        dedup = DocumentDeduplicator()
        text = " ".join(f"слово{i}" for i in range(10))
        shingles = dedup.shingle(text)
        expected_count = 10 - DocumentDeduplicator.SHINGLE_SIZE + 1
        assert len(shingles) == expected_count


@pytest.mark.unit
class TestMinhashSignature:
    def test_signature_length_matches_num_hash_functions(self):
        dedup = DocumentDeduplicator()
        shingles = dedup.shingle(REGULATION_V1)
        signature = dedup.minhash_signature(shingles)
        assert len(signature) == DocumentDeduplicator.NUM_MINHASH_FUNCTIONS

    def test_signature_is_deterministic_for_same_shingles(self):
        dedup = DocumentDeduplicator()
        shingles = dedup.shingle(REGULATION_V1)
        sig_a = dedup.minhash_signature(shingles)
        sig_b = dedup.minhash_signature(shingles)
        assert sig_a == sig_b

    def test_same_seed_gives_same_signature_across_instances(self):
        dedup_a = DocumentDeduplicator(seed=42)
        dedup_b = DocumentDeduplicator(seed=42)
        shingles = dedup_a.shingle(REGULATION_V1)
        assert dedup_a.minhash_signature(shingles) == dedup_b.minhash_signature(
            shingles
        )

    def test_different_seed_gives_different_hash_functions(self):
        dedup_a = DocumentDeduplicator(seed=1)
        dedup_b = DocumentDeduplicator(seed=2)
        assert dedup_a._hash_seeds != dedup_b._hash_seeds


@pytest.mark.unit
class TestEstimateJaccardSimilarity:
    def test_identical_signatures_give_similarity_one(self):
        dedup = DocumentDeduplicator()
        shingles = dedup.shingle(REGULATION_V1)
        signature = dedup.minhash_signature(shingles)
        assert dedup.estimate_jaccard_similarity(signature, signature) == 1.0

    def test_mismatched_length_returns_zero(self):
        dedup = DocumentDeduplicator()
        assert dedup.estimate_jaccard_similarity([1, 2, 3], [1, 2]) == 0.0

    def test_empty_signatures_return_zero(self):
        dedup = DocumentDeduplicator()
        assert dedup.estimate_jaccard_similarity([], []) == 0.0

    def test_near_duplicate_documents_have_high_estimated_similarity(self):
        """Ключевой сценарий урока: единственная правка даты в длинном
        документе должна давать высокую оценку сходства по MinHash."""
        dedup = DocumentDeduplicator()
        sig1 = dedup.minhash_signature(dedup.shingle(REGULATION_V1))
        sig2 = dedup.minhash_signature(dedup.shingle(REGULATION_V2_DATE_EDIT))
        similarity = dedup.estimate_jaccard_similarity(sig1, sig2)
        assert similarity >= 0.75

    def test_unrelated_documents_have_low_estimated_similarity(self):
        dedup = DocumentDeduplicator()
        sig1 = dedup.minhash_signature(dedup.shingle(REGULATION_V1))
        sig2 = dedup.minhash_signature(dedup.shingle(UNRELATED_DOCUMENT))
        similarity = dedup.estimate_jaccard_similarity(sig1, sig2)
        assert similarity < 0.4


@pytest.mark.unit
class TestRegisterDocument:
    def test_first_document_is_unique(self):
        dedup = DocumentDeduplicator()
        match = dedup.register_document("doc_sed_v1", REGULATION_V1)
        assert isinstance(match, DuplicateMatch)
        assert match.match_type == "unique"
        assert match.similarity == 0.0
        assert match.matched_doc_id is None

    def test_exact_duplicate_is_detected_without_storing_new_signature(self):
        dedup = DocumentDeduplicator()
        dedup.register_document("doc_sed_v1", REGULATION_V1)
        match = dedup.register_document("doc_confluence_copy", REGULATION_V1)
        assert match.match_type == "exact"
        assert match.similarity == 1.0
        assert match.matched_doc_id == "doc_sed_v1"
        # Точный дубль не добавляет новую сигнатуру в индекс дублей.
        assert dedup.known_document_count() == 1

    def test_date_edit_is_classified_as_near_duplicate(self):
        """Ровно сценарий истории урока: положение из СЭД с датой
        "01 марта" и версия из Confluence с датой "15 марта"."""
        dedup = DocumentDeduplicator()
        dedup.register_document("doc_sed_v1", REGULATION_V1)
        match = dedup.register_document("doc_confluence_v2", REGULATION_V2_DATE_EDIT)
        assert match.match_type == "near_duplicate"
        assert match.similarity >= DocumentDeduplicator.NEAR_DUPLICATE_THRESHOLD
        assert match.matched_doc_id == "doc_sed_v1"
        # Near-duplicate ВСЁ РАВНО регистрируется, чтобы дальнейшие
        # документы могли совпасть и с этой версией.
        assert dedup.known_document_count() == 2

    def test_unrelated_document_is_unique(self):
        dedup = DocumentDeduplicator()
        dedup.register_document("doc_sed_v1", REGULATION_V1)
        match = dedup.register_document("doc_remote_access_policy", UNRELATED_DOCUMENT)
        assert match.match_type == "unique"
        assert match.matched_doc_id is None

    def test_three_document_scenario_matches_lesson_story(self):
        """Три копии одного положения (СЭД, Confluence, файловый
        сервер без правок вообще) плюс один не связанный документ -
        итоговая классификация должна отражать историю урока целиком."""
        dedup = DocumentDeduplicator()
        m_sed = dedup.register_document("sed_archivarius", REGULATION_V1)
        m_confluence = dedup.register_document(
            "confluence_working_copy", REGULATION_V2_DATE_EDIT
        )
        m_fileserver = dedup.register_document("fileserver_copy", REGULATION_V1)
        m_unrelated = dedup.register_document(
            "remote_access_policy", UNRELATED_DOCUMENT
        )

        assert m_sed.match_type == "unique"
        assert m_confluence.match_type == "near_duplicate"
        assert m_fileserver.match_type == "exact"
        assert m_unrelated.match_type == "unique"
