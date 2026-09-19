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
from app.services.accession_extractor import extract_book_metadata

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
    double-counts a document (spec section 6: append, never duplicate)."""
    metadata = extract_book_metadata(document.original_filename, pages, words_by_page)

    existing = db.scalar(select(AccessionRecord).where(AccessionRecord.document_id == document.id))
    if existing:
        record = existing
        record.book_name = metadata.book_name
        record.creator = metadata.creator
        record.language = metadata.language
        record.year_of_publication = metadata.year_of_publication
        record.total_pages = document.page_count
        record.needs_review = metadata.needs_review
        record.extraction_notes = metadata.notes_text
    else:
        record = AccessionRecord(
            document_id=document.id,
            accession_number=accession_number or generate_next_accession_number(db),
            book_name=metadata.book_name,
            creator=metadata.creator,
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
