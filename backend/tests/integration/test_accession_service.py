"""
Integration tests for the library accession register's DB-backed service
(`app.services.accession_service`) against a real (in-memory sqlite)
database -- covers accession-number generation, append-not-duplicate
behavior on reprocessing, and duplicate-upload hash detection (spec
sections 6/8/9).
"""
from datetime import date

from app.models.enums import LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum
from app.schemas.document_json import DocumentBlockJSON, PageJSON, TextRunJSON
from app.schemas.geometry import BBox, NormBBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.services import accession_service, document_service

_BBOX = BBox(x1=0, y1=0, x2=100, y2=20)
_NORM_BBOX = NormBBox(x1=0.0, y1=0.0, x2=1.0, y2=0.1)


def _title_page(text: str) -> PageJSON:
    block = DocumentBlockJSON(
        id="blk-0", type=LayoutBlockType.TITLE, bbox=_BBOX, bbox_norm=_NORM_BBOX,
        confidence=0.9, z_order=0, content=[TextRunJSON(text=text)],
    )
    return PageJSON(document_id="doc-1", page_id="pg-1", page_number=1, page_width=1000, page_height=1400, dpi=150, blocks=[block])


def _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="book.png"):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, filename, sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    return document


def test_generate_next_accession_number_starts_at_configured_start(test_db_session):
    number = accession_service.generate_next_accession_number(test_db_session)
    assert number == "D-1"  # ACCESSION_NUMBER_START default


def test_generate_next_accession_number_increments_past_existing(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Book One")])

    next_number = accession_service.generate_next_accession_number(test_db_session)
    assert next_number == "D-2"


def test_create_accession_record_extracts_and_persists_fields(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    record = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("STATISTICAL ABSTRACT OF PUNJAB 2003")])

    assert record.accession_number == "D-1"
    assert record.book_name == "STATISTICAL ABSTRACT OF PUNJAB 2003"
    assert record.total_pages == doc.page_count
    assert record.record_date == date.today()


def test_reprocessing_the_same_document_updates_its_row_not_a_new_one(test_db_session, tmp_storage, sample_png_bytes):
    """Spec section 6: append, never duplicate -- a document that gets
    reprocessed (e.g. after a fix) must still occupy exactly one row."""
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    first = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Original Title")])
    second = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Corrected Title")])

    assert first.id == second.id
    assert second.book_name == "Corrected Title"
    all_records = accession_service.get_all_accession_records(test_db_session)
    assert len(all_records) == 1


def test_different_documents_each_get_their_own_row_and_accession_number(test_db_session, tmp_storage, sample_png_bytes):
    doc_a = _make_document(test_db_session, tmp_storage, sample_png_bytes, "a.png")
    doc_b = _make_document(test_db_session, tmp_storage, sample_png_bytes, "b.png")
    record_a = accession_service.create_accession_record_for_document(test_db_session, doc_a, [_title_page("Book A")])
    record_b = accession_service.create_accession_record_for_document(test_db_session, doc_b, [_title_page("Book B")])

    assert record_a.accession_number != record_b.accession_number
    assert {r.id for r in accession_service.get_all_accession_records(test_db_session)} == {record_a.id, record_b.id}


def test_duplicate_upload_is_detected_by_content_hash(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    found = document_service.find_document_by_hash(test_db_session, document_service.compute_document_hash(sample_png_bytes))
    assert found is not None
    assert found.id == doc.id


def test_duplicate_check_ignores_soft_deleted_documents(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    document_service.soft_delete_document(test_db_session, tmp_storage, doc)
    found = document_service.find_document_by_hash(test_db_session, document_service.compute_document_hash(sample_png_bytes))
    assert found is None


def test_different_content_never_hash_collides_in_practice(test_db_session, tmp_storage, sample_png_bytes):
    _make_document(test_db_session, tmp_storage, sample_png_bytes)
    other_bytes = sample_png_bytes + b"\x00extra"
    found = document_service.find_document_by_hash(test_db_session, document_service.compute_document_hash(other_bytes))
    assert found is None


def test_list_accession_records_filters_by_year_and_month(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes)
    accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Book One")])

    today = date.today()
    items, total = accession_service.list_accession_records(test_db_session, year=today.year, month=today.month)
    assert total == 1
    assert items[0].book_name == "Book One"

    items, total = accession_service.list_accession_records(test_db_session, year=today.year - 50)
    assert total == 0


def test_accession_summary_counts(test_db_session, tmp_storage, sample_png_bytes):
    doc_a = _make_document(test_db_session, tmp_storage, sample_png_bytes, "a.png")
    doc_b = _make_document(test_db_session, tmp_storage, sample_png_bytes, "b.png")
    # A fully-populated title page (title + author heading + a plausible
    # year) -> extraction should be confident, needs_review=False.
    title_block = DocumentBlockJSON(
        id="blk-title", type=LayoutBlockType.TITLE, bbox=_BBOX, bbox_norm=_NORM_BBOX,
        confidence=0.9, z_order=0, content=[TextRunJSON(text="STATISTICAL ABSTRACT 1998")],
    )
    author_block = DocumentBlockJSON(
        id="blk-author", type=LayoutBlockType.HEADING, bbox=_BBOX, bbox_norm=_NORM_BBOX,
        confidence=0.9, z_order=1, content=[TextRunJSON(text="MINISTRY OF AGRICULTURE")],
    )
    full_page = PageJSON(document_id="doc-a", page_id="pg-1", page_number=1, page_width=1000, page_height=1400, dpi=150, blocks=[title_block, author_block])
    words = {1: [OCRWordResult(text="Statistical", confidence=0.9, bbox=_BBOX, polygon=Polygon.from_xy_list([[0, 0], [10, 0], [10, 10], [0, 10]]), page_number=1, block_id="b", line_id="l", language="en")]}
    accession_service.create_accession_record_for_document(test_db_session, doc_a, [full_page], words)

    # No blocks at all on this one -> falls back to filename, no
    # creator/year/language -> needs_review.
    accession_service.create_accession_record_for_document(test_db_session, doc_b, [PageJSON(document_id="doc-b", page_id="pg-1", page_number=1, page_width=100, page_height=100, dpi=150, blocks=[])])

    summary = accession_service.get_accession_summary(test_db_session)
    assert summary["total_records"] == 2
    assert summary["needs_review_count"] == 1
    assert summary["total_documents_processed"] == 2
    assert summary["latest_record_date"] == date.today()


def test_accession_number_from_filename_is_used_instead_of_auto_generated(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="ECONOMIC_SURVEY_ACC_NO.1087.png")
    record = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Economic Survey")])
    assert record.accession_number == "D-1087"


def test_extracted_accession_number_already_taken_falls_back_to_auto_generated(test_db_session, tmp_storage, sample_png_bytes):
    first = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="BOOK_A_ACC_NO.1087.png")
    accession_service.create_accession_record_for_document(test_db_session, first, [_title_page("Book A")])

    second = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="BOOK_B_ACC_NO.1087.png")
    record = accession_service.create_accession_record_for_document(test_db_session, second, [_title_page("Book B")])

    assert record.accession_number != "D-1087"  # already taken by `first` -- never a silent collision
    # Falls back to the normal sequence -- which itself continues from the
    # highest existing "D-N" already in the DB, including the extracted
    # "D-1087" above, not from D-1.
    assert record.accession_number == "D-1088"
    assert "already assigned" in (record.extraction_notes or "")


def test_reprocessing_never_changes_an_already_assigned_accession_number(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="BOOK_ACC_NO.1087.png")
    first = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Original Title")])
    assert first.accession_number == "D-1087"

    # Reprocessing re-runs extraction against the SAME filename -- would
    # resolve to the same "D-1087" anyway here, but the point is this path
    # never even consults extraction for an EXISTING record's number.
    second = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Corrected Title")])
    assert second.id == first.id
    assert second.accession_number == "D-1087"


def test_backfill_updates_a_placeholder_number_to_the_real_one_from_the_filename(test_db_session, tmp_storage, sample_png_bytes):
    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="a_plain_name.png")
    record = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Book")])
    assert record.accession_number == "D-1"  # no ACC_NO in the filename yet -- auto-generated

    doc.original_filename = "BOOK_ACC_NO.1087.png"
    test_db_session.commit()

    result = accession_service.backfill_extracted_accession_numbers(test_db_session)

    assert result == {"updated": 1, "skipped_taken": 0, "unchanged": 0}
    test_db_session.refresh(record)
    assert record.accession_number == "D-1087"


def test_backfill_never_overwrites_a_number_already_taken_by_another_record(test_db_session, tmp_storage, sample_png_bytes):
    taken = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="OTHER_ACC_NO.1087.png")
    accession_service.create_accession_record_for_document(test_db_session, taken, [_title_page("Other Book")])

    doc = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="a_plain_name.png")
    record = accession_service.create_accession_record_for_document(test_db_session, doc, [_title_page("Book")])
    doc.original_filename = "BOOK_ACC_NO.1087.png"  # collides with `taken`'s real number
    test_db_session.commit()

    result = accession_service.backfill_extracted_accession_numbers(test_db_session)

    # `taken`'s own record is already correct (unchanged); `record` is the
    # one that would collide (skipped).
    assert result == {"updated": 0, "skipped_taken": 1, "unchanged": 1}
    test_db_session.refresh(record)
    assert record.accession_number == "D-1088"  # left alone, not collided with `taken`'s "D-1087"


def test_backfill_leaves_a_correct_or_unmatchable_number_alone(test_db_session, tmp_storage, sample_png_bytes):
    already_correct = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="X_ACC_NO.55.png")
    correct_record = accession_service.create_accession_record_for_document(test_db_session, already_correct, [_title_page("A")])
    assert correct_record.accession_number == "D-55"

    no_match = _make_document(test_db_session, tmp_storage, sample_png_bytes, filename="no_acc_no.png")
    no_match_record = accession_service.create_accession_record_for_document(test_db_session, no_match, [_title_page("B")])

    result = accession_service.backfill_extracted_accession_numbers(test_db_session)

    assert result == {"updated": 0, "skipped_taken": 0, "unchanged": 2}
    assert correct_record.accession_number == "D-55"
    assert no_match_record.accession_number == "D-56"  # auto-generation continues past the highest existing number
