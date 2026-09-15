from __future__ import annotations

from datetime import date as date_type
from typing import Optional

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid


class AccessionRecord(TimestampMixin, Base):
    """One row in the cumulative library accession register (spec: PDF
    extraction -> DB -> Master Excel).

    Created automatically once a document finishes processing (see
    `app.workers.pipeline_tasks._run_pipeline`'s call into
    `app.services.accession_service.create_accession_record_for_document`)
    and never overwritten by a later, unrelated upload -- one row per
    document, accumulating forever. This table (not any Excel file) is
    the source of truth: `app.exporters.master_register_exporter`
    generates the Master Excel fresh from these rows on every download,
    it never reads/writes a stored workbook.
    """

    __tablename__ = "accession_records"
    __table_args__ = (
        # One record per document -- reprocessing a document updates its
        # existing row (see `accession_service`) rather than appending a
        # second one for the same source file.
        UniqueConstraint("document_id", name="uq_accession_records_document_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Library-assigned catalog number (e.g. "D-305") -- not something any
    # book's own content could contain, so this is always either supplied
    # by the caller or auto-generated sequentially (see
    # `accession_service.generate_next_accession_number`), never OCR'd.
    accession_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # Everything below IS extracted from the document's own OCR'd content
    # (title page) -- best-effort, never invented: a field that can't be
    # confidently read is left blank/null with `needs_review` set, rather
    # than guessed (spec section 14).
    book_name: Mapped[str] = mapped_column(Text, nullable=False)
    creator: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    year_of_publication: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Denormalized copy of document.page_count -- always exact (no OCR
    # guesswork needed for this one field) and convenient for the Excel
    # export to read without a join.
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    # Free-text "Type" column (per the reference register) -- not
    # auto-classified; left for manual entry/future use.
    record_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # The accession/processing date -- the day THIS record was created,
    # matching the reference register's "Date" column (which tracks when
    # each book was catalogued, not the book's own publication year --
    # that is `year_of_publication`, a separate column).
    record_date: Mapped[date_type] = mapped_column(Date, nullable=False)

    # True if any extracted field is uncertain (heuristic extraction, not
    # a hallucination-prone guess) -- surfaced so a human can review/edit
    # rather than silently trusting a low-confidence value.
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    extraction_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    document: Mapped["Document"] = relationship()
