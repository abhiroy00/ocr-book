"""
DB-backed CRUD for the library accession register (spec sections 6/8/9).

The database is the source of truth throughout: the Master Excel export
(`app.exporters.master_register_exporter`) always reads fresh from these
rows, never from a cached/stored workbook, and duplicate-upload detection
is a plain query against `Document.document_hash`, never a side file.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.accession_record import AccessionRecord
from app.models.document import Document
from app.schemas.document_json import PageJSON
from app.schemas.ocr import OCRWordResult
from app.services.accession_extractor import extract_accession_number_from_filename, extract_book_metadata

logger = get_logger(__name__)

_ACCESSION_NUMBER_RE = re.compile(r"^(?P<prefix>.+)-(?P<n>\d+)$")


def generate_next_accession_number(db: Session) -> str:
    """`{prefix}-{n}`, continuing from the highest existing `n` for this
    prefix (or `ACCESSION_NUMBER_START` if there are none yet) --
    configurable rather than always starting at 1, since this system may
    be continuing an existing physical/manual register (spec section 6)."""
    settings = get_settings()
    prefix = settings.accession_number_prefix
    existing = db.scalars(
        select(AccessionRecord.accession_number).where(AccessionRecord.accession_number.like(f"{prefix}-%"))
    ).all()
    max_n = settings.accession_number_start - 1
    for value in existing:
        match = _ACCESSION_NUMBER_RE.match(value)
        if match and match.group("prefix") == prefix:
            max_n = max(max_n, int(match.group("n")))
    return f"{prefix}-{max_n + 1}"


def create_accession_record_for_document(
    db: Session,
    document: Document,
    pages: list[PageJSON],
    words_by_page: dict[int, list[OCRWordResult]] | None = None,
    accession_number: str | None = None,
) -> AccessionRecord:
    """Called once a document finishes processing (see
    `app.workers.pipeline_tasks._run_pipeline`) -- extracts bibliographic
    metadata and appends one row to the register. Idempotent per document:
    reprocessing the SAME document updates its existing row in place
    (re-extracting from the fresh pipeline output) rather than appending a
    second row for the same source file, so the master dataset never
    double-counts a document (spec section 6: append, never duplicate).

    Accession number: the caller's `accession_number` argument (if given)
    wins; otherwise a real one is used if `extract_book_metadata` found it
    printed on the book itself (filename or a stamped "ACC.NO." on the
    cover -- see `accession_extractor._extract_accession_number`); only
    when NEITHER is available does this fall back to the next sequential
    `{prefix}-N`. A REPROCESS never changes an already-assigned number,
    even if the fresh extraction disagrees -- that would silently
    renumber a book already catalogued under the old one."""
    metadata = extract_book_metadata(document.original_filename, pages, words_by_page)

    existing = db.scalar(select(AccessionRecord).where(AccessionRecord.document_id == document.id))

    resolved_number = accession_number or metadata.accession_number
    if resolved_number and not existing:
        taken_by = db.scalar(
            select(AccessionRecord).where(
                AccessionRecord.accession_number == resolved_number,
                AccessionRecord.document_id != document.id,
            )
        )
        if taken_by is not None:
            # A misread digit or two documents genuinely sharing a stamped
            # number -- never silently collide with (or overwrite) another
            # document's row; fall back to auto-generation instead.
            logger.warning(
                "extracted_accession_number_already_taken", document_id=document.id,
                accession_number=resolved_number, taken_by_document_id=taken_by.document_id,
            )
            metadata.notes.append(
                f"accession_number: extracted '{resolved_number}' is already assigned to another document; generated a new one instead"
            )
            resolved_number = None

    if existing:
        record = existing
        record.book_name = metadata.book_name
        record.creator = metadata.creator
        record.author = metadata.author
        record.publisher = metadata.publisher
        record.language = metadata.language
        record.year_of_publication = metadata.year_of_publication
        record.total_pages = document.page_count
        record.needs_review = metadata.needs_review
        record.extraction_notes = metadata.notes_text
        # accession_number is deliberately left untouched here.
    else:
        record = AccessionRecord(
            document_id=document.id,
            accession_number=resolved_number or generate_next_accession_number(db),
            book_name=metadata.book_name,
            creator=metadata.creator,
            author=metadata.author,
            publisher=metadata.publisher,
            language=metadata.language,
            year_of_publication=metadata.year_of_publication,
            total_pages=document.page_count,
            record_date=datetime.now(timezone.utc).date(),
            needs_review=metadata.needs_review,
            extraction_notes=metadata.notes_text,
        )
        db.add(record)

    db.commit()
    db.refresh(record)
    logger.info(
        "accession_record_upserted", document_id=document.id, accession_number=record.accession_number,
        needs_review=record.needs_review,
    )
    return record


def backfill_extracted_accession_numbers(db: Session) -> dict:
    """Explicit, opt-in correction for records still holding an
    auto-generated placeholder from before this project could read a real
    accession number at all (filename-only -- see
    `accession_extractor.extract_accession_number_from_filename`; the
    cover-OCR fallback needs full page data and isn't worth reloading for
    every record here, so a book without an ACC_NO-bearing filename just
    keeps its current number).

    Deliberately NOT part of `create_accession_record_for_document`'s
    normal reprocess path, which never touches an existing record's
    number on purpose (renumbering an already-catalogued book on routine
    reprocess would be wrong) -- this is a one-off, explicitly invoked
    migration for records that were never given the chance to have a real
    number in the first place. Never overwrites a number that's already
    correct, and never collides with a DIFFERENT record's number --
    including two records in this SAME run resolving to the same target
    (e.g. the same book uploaded twice, both filenames carrying the
    identical ACC_NO): confirmed as a real crash otherwise -- this
    function only commits once at the end (one atomic pass, not N), so an
    in-DB `taken_by` check alone can't see an EARLIER record's not-yet-
    flushed reassignment from later in the same loop, and two records
    landing on the same number blows the column's UNIQUE constraint at
    commit time, failing the whole batch instead of just that one row.
    `_reserved_this_run` closes that gap without needing a flush per row."""
    updated = skipped_taken = unchanged = 0
    reserved_this_run: set[str] = set()
    for record in db.scalars(select(AccessionRecord)).all():
        document = db.get(Document, record.document_id)
        if document is None:
            continue
        extracted = extract_accession_number_from_filename(document.original_filename)
        if extracted is None or extracted == record.accession_number:
            unchanged += 1
            continue
        taken_by = db.scalar(
            select(AccessionRecord).where(
                AccessionRecord.accession_number == extracted, AccessionRecord.id != record.id
            )
        )
        if taken_by is not None or extracted in reserved_this_run:
            logger.warning(
                "accession_number_backfill_collision", document_id=document.id,
                current=record.accession_number, extracted=extracted,
                taken_by_document_id=getattr(taken_by, "document_id", None),
            )
            skipped_taken += 1
            continue
        logger.info(
            "accession_number_backfilled", document_id=document.id,
            old=record.accession_number, new=extracted,
        )
        record.accession_number = extracted
        reserved_this_run.add(extracted)
        updated += 1

    db.commit()
    return {"updated": updated, "skipped_taken": skipped_taken, "unchanged": unchanged}


def list_accession_records(
    db: Session, year: int | None = None, month: int | None = None, page: int = 1, page_size: int = 50
) -> tuple[list[AccessionRecord], int]:
    stmt = select(AccessionRecord)
    if year is not None:
        stmt = stmt.where(func.extract("year", AccessionRecord.record_date) == year)
    if month is not None:
        stmt = stmt.where(func.extract("month", AccessionRecord.record_date) == month)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    stmt = stmt.order_by(AccessionRecord.record_date, AccessionRecord.accession_number).offset((page - 1) * page_size).limit(page_size)
    items = list(db.scalars(stmt).all())
    return items, total


def get_all_accession_records(db: Session) -> list[AccessionRecord]:
    """Unpaginated -- for building the Master Excel, which must always
    contain every record (spec section 6/7)."""
    return list(db.scalars(select(AccessionRecord).order_by(AccessionRecord.record_date, AccessionRecord.accession_number)).all())


def get_accession_summary(db: Session) -> dict:
    total_records = db.scalar(select(func.count()).select_from(AccessionRecord.__table__)) or 0
    needs_review = db.scalar(select(func.count()).where(AccessionRecord.needs_review.is_(True))) or 0
    latest: date | None = db.scalar(select(func.max(AccessionRecord.record_date)))
    total_documents = db.scalar(
        select(func.count()).select_from(Document.__table__).where(Document.is_deleted.is_(False))
    ) or 0
    return {
        "total_records": total_records,
        "needs_review_count": needs_review,
        "latest_record_date": latest,
        "total_documents_processed": total_documents,
    }


def refresh_metadata_from_stored_ocr(db: Session, document: Document) -> AccessionRecord | None:
    """Re-reads a document's bibliographic metadata from its ALREADY-STORED
    layout + OCR data (no re-OCR, no re-processing) and updates its
    accession row in place -- the accession number and date never change.
    For books processed before the extractor improved: their rows hold the
    old, weaker values, and re-uploading a 150-page scan just to re-read
    its title page would be a needless 40 minutes of OCR.

    Returns the updated record, or None if the document has no stored
    pages to read."""
    # Local import: document_service is a much larger module that this
    # one otherwise has no reason to depend on.
    from app.services import document_service

    pages = document_service.get_document_json(db, document).pages
    if not pages:
        return None

    words_by_id = document_service.get_ocr_words_by_page_for_document(db, document.id)
    words_by_page = {p.page_number: words_by_id.get(p.page_id, []) for p in pages}
    return create_accession_record_for_document(db, document, pages, words_by_page)
