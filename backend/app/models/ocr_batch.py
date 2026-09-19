"""
Multiple-file OCR ("batch") bookkeeping.

A batch groups N ordinary Documents that share one set of OCR settings and
are scheduled together. Each file is still a normal `Document` +
`ProcessingJob` that runs through the exact same pipeline as a single-file
upload -- these tables only record *scheduling* state (which files belong
together, their priority, whether they are waiting/running/done), so the
existing documents/processing_jobs tables and the single-file flow are
untouched. Per-file progress (stage, percent, processed pages) is read from
the file's own ProcessingJob, never duplicated here.

`state` columns are plain strings validated in application code (see the
enums below), not Postgres ENUM types -- these two tables are new, so there
is no reason to take on `CREATE TYPE` migration/rollback complexity.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid


class BatchItemState(str, enum.Enum):
    PENDING = "pending"  # uploaded, batch not started yet
    QUEUED = "queued"  # started, waiting for a free slot
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"  # identical content already exists; not reprocessed


TERMINAL_ITEM_STATES = {
    BatchItemState.COMPLETED,
    BatchItemState.FAILED,
    BatchItemState.CANCELLED,
    BatchItemState.DUPLICATE,
}


class OCRBatch(TimestampMixin, Base):
    __tablename__ = "ocr_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)

    # Settings applied uniformly to every file in the batch (spec: the
    # user's engine/DPI/profile choice covers the whole batch).
    ocr_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    preprocess_profile: Mapped[str] = mapped_column(String(32), nullable=False)

    uploaded_by: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # NULL while files are still being uploaded; set by `start`. Scheduling
    # (priority across the whole set) can only be decided once every file is
    # known, which is why upload and start are separate steps.
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    items: Mapped[list["OCRBatchItem"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="OCRBatchItem.position"
    )


class OCRBatchItem(TimestampMixin, Base):
    """One file in a batch. `id` is the stable per-file job id exposed to the
    frontend; `processing_job_id` is the current pipeline run (a retry
    creates a fresh ProcessingJob under the same item)."""

    __tablename__ = "ocr_batch_items"
    __table_args__ = (Index("ix_ocr_batch_items_batch_id_state", "batch_id", "state"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    batch_id: Mapped[str] = mapped_column(String(36), ForeignKey("ocr_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    processing_job_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("processing_jobs.id", ondelete="SET NULL"), nullable=True
    )

    position: Mapped[int] = mapped_column(Integer, nullable=False)  # upload order (stable tiebreak)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default=BatchItemState.PENDING.value)

    # Lightweight workload estimate, fixed at start (see
    # batch_scheduler.estimate_workload) -- lower runs earlier.
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    batch: Mapped[OCRBatch] = relationship(back_populates="items")
