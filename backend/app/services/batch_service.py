"""
Multiple-file OCR: batch CRUD, per-file status aggregation, retry.

Files are added one request at a time (`add_file`) and only scheduled once
the client calls `start_batch`, so (a) each upload stays a normal
single-file-sized request the existing nginx limit already allows, (b) only
one file's bytes are in the API process at once, and (c) priority is decided
over the complete set. Each file becomes an ordinary Document +
ProcessingJob via the existing `document_service.create_document_from_upload`
-- this module adds no OCR, storage or validation logic of its own.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum
from app.models.ocr_batch import TERMINAL_ITEM_STATES, BatchItemState, OCRBatch, OCRBatchItem
from app.models.processing_job import ProcessingJob
from app.schemas.batch import BatchItemRead, BatchRead, ConcurrencyInfo
from app.services import batch_scheduler, document_service
from app.services.storage import LocalStorageBackend, StorageBackend
from app.utils.file_safety import UploadValidationError, validate_upload

logger = get_logger(__name__)

_TERMINAL_VALUES = {s.value for s in TERMINAL_ITEM_STATES}


class BatchError(ValueError):
    """A rejected batch operation; the message is safe to show to the user.
    `status_code` lets the API layer map it without string-matching."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


# ----------------------------------------------------------------------------
# Create / add / start
# ----------------------------------------------------------------------------

def create_batch(
    db: Session,
    ocr_provider: OCRProviderEnum,
    dpi: int,
    preprocess_profile: PreprocessProfileEnum,
    uploaded_by: Optional[str] = None,
) -> OCRBatch:
    batch = OCRBatch(
        ocr_provider=ocr_provider.value,
        dpi=dpi,
        preprocess_profile=preprocess_profile.value,
        uploaded_by=uploaded_by,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    logger.info("batch_created", batch_id=batch.id, provider=batch.ocr_provider, dpi=dpi, profile=batch.preprocess_profile)
    return batch


def _ensure_disk_headroom(storage: StorageBackend, pages: int) -> None:
    """Refuses a file that would push local disk below a safe reserve --
    an uncontrolled batch must not be able to fill the volume (which would
    also take Postgres/Redis down with it). Only meaningful for local
    storage; S3 has no such constraint."""
    if not isinstance(storage, LocalStorageBackend):
        return
    settings = get_settings()
    need_mb = settings.batch_disk_reserve_mb + pages * settings.batch_disk_mb_per_page
    free_mb = shutil.disk_usage(storage.root).free // (1024 * 1024)
    if free_mb < need_mb:
        raise BatchError(
            f"Not enough free disk space to process this file ({free_mb}MB free, ~{need_mb}MB needed).", status_code=507
        )


def add_file(
    db: Session,
    storage: StorageBackend,
    batch: OCRBatch,
    filename: str,
    content: bytes,
    force: bool = False,
    uploaded_by: Optional[str] = None,
) -> OCRBatchItem:
    settings = get_settings()
    if batch.started_at is not None:
        raise BatchError("This batch has already started; create a new batch to add more files.", status_code=409)

    existing_items = db.scalars(select(OCRBatchItem).where(OCRBatchItem.batch_id == batch.id)).all()
    if len(existing_items) >= settings.batch_max_files:
        raise BatchError(f"A batch can contain at most {settings.batch_max_files} files.")

    total_bytes = db.scalar(
        select(func.coalesce(func.sum(Document.file_size_bytes), 0))
        .join(OCRBatchItem, OCRBatchItem.document_id == Document.id)
        .where(OCRBatchItem.batch_id == batch.id)
    ) or 0
    if total_bytes + len(content) > settings.batch_max_total_mb * 1024 * 1024:
        raise BatchError(f"A batch is limited to {settings.batch_max_total_mb}MB in total.", status_code=413)

    position = len(existing_items)

    # Same duplicate rule as the single-file upload (spec section 9): identical
    # content that already exists is reported, not silently reprocessed.
    if not force:
        duplicate = document_service.find_document_by_hash(db, document_service.compute_document_hash(content))
        if duplicate is not None:
            latest_job = db.scalar(
                select(ProcessingJob).where(ProcessingJob.document_id == duplicate.id).order_by(ProcessingJob.created_at.desc())
            )
            item = OCRBatchItem(
                batch_id=batch.id,
                document_id=duplicate.id,
                processing_job_id=latest_job.id if latest_job else None,
                position=position,
                state=BatchItemState.DUPLICATE.value,
                priority_score=0.0,
            )
            db.add(item)
            db.commit()
            db.refresh(item)
            logger.info("batch_file_duplicate", batch_id=batch.id, item_id=item.id, document_id=duplicate.id)
            return item

    # Validates extension/size/content and reads the page count (cheap: PDF
    # page tree only, nothing is rendered) -- all inside the existing function.
    try:
        validated = validate_upload(filename, content)
    except UploadValidationError as exc:
        raise BatchError(str(exc)) from exc
    _ensure_disk_headroom(storage, validated.page_count)

    try:
        document, job = document_service.create_document_from_upload(
            db,
            storage,
            filename or "upload",
            content,
            OCRProviderEnum(batch.ocr_provider),
            batch.dpi,
            PreprocessProfileEnum(batch.preprocess_profile),
            uploaded_by=uploaded_by,
        )
    except UploadValidationError as exc:
        raise BatchError(str(exc)) from exc

    item = OCRBatchItem(
        batch_id=batch.id,
        document_id=document.id,
        processing_job_id=job.id,
        position=position,
        state=BatchItemState.PENDING.value,
        priority_score=batch_scheduler.estimate_workload(document.page_count, document.file_size_bytes),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info(
        "batch_file_added", batch_id=batch.id, item_id=item.id, document_id=document.id,
        pages=document.page_count, size_bytes=document.file_size_bytes, priority=round(item.priority_score, 3),
    )
    return item


def start_batch(db: Session, batch: OCRBatch) -> int:
    """Marks every pending file queued and returns how many slot tasks the
    caller must enqueue (one per queued file). Idempotent: a second call on
    an already-started batch queues nothing."""
    if batch.started_at is not None:
        return 0
    items = db.scalars(select(OCRBatchItem).where(OCRBatchItem.batch_id == batch.id)).all()
    if not items:
        raise BatchError("Add at least one file before starting the batch.")

    queued = 0
    for item in items:
        if item.state == BatchItemState.PENDING.value:
            item.state = BatchItemState.QUEUED.value
            queued += 1
    batch.started_at = datetime.now(timezone.utc)
    db.commit()

    for item in sorted((i for i in items if i.state == BatchItemState.QUEUED.value), key=lambda i: (i.priority_score, i.position)):
        logger.info(
            "batch_queue_add", batch_id=batch.id, item_id=item.id, document_id=item.document_id,
            priority=round(item.priority_score, 3),
        )
    return queued


def revert_start(db: Session, batch: OCRBatch) -> None:
    """Undo `start_batch` when the tasks could not be queued (broker down),
    so the client can simply call start again instead of the batch sitting
    'processing' with nothing scheduled."""
    for item in db.scalars(select(OCRBatchItem).where(OCRBatchItem.batch_id == batch.id)).all():
        if item.state == BatchItemState.QUEUED.value:
            item.state = BatchItemState.PENDING.value
    batch.started_at = None
    db.commit()


# ----------------------------------------------------------------------------
# State sync (the pipeline owns job state; the item mirrors it)
# ----------------------------------------------------------------------------

_JOB_TO_ITEM_STATE = {
    DocumentStatus.COMPLETED: BatchItemState.COMPLETED,
    DocumentStatus.FAILED: BatchItemState.FAILED,
    DocumentStatus.CANCELLED: BatchItemState.CANCELLED,
}


def sync_item_with_job(db: Session, item: OCRBatchItem, job: Optional[ProcessingJob] = None) -> OCRBatchItem:
    """Copies a finished pipeline run's outcome onto its batch item. Only
    ever moves a RUNNING/QUEUED item to a terminal state; never the
    reverse, so it is safe to call repeatedly from the status endpoint."""
    if item.state in _TERMINAL_VALUES:
        return item
    job = job or (db.get(ProcessingJob, item.processing_job_id) if item.processing_job_id else None)
    if job is None:
        return item
    mapped = _JOB_TO_ITEM_STATE.get(job.status)
    if mapped is None:
        return item
    item.state = mapped.value
    if mapped is BatchItemState.FAILED:
        item.error_message = job.error_message or "Processing failed"
    db.commit()
    return item


def refresh_running_items(db: Session, batch: OCRBatch) -> None:
    """Called on every status read: lets a job that died without ever
    reaching a terminal state (killed worker) be noticed via the existing
    lazy stale-job detection, then mirrors any terminal job outcome."""
    items = db.scalars(
        select(OCRBatchItem).where(
            OCRBatchItem.batch_id == batch.id,
            OCRBatchItem.state.in_([BatchItemState.RUNNING.value, BatchItemState.QUEUED.value]),
        )
    ).all()
    for item in items:
        if not item.processing_job_id:
            continue
        job = db.get(ProcessingJob, item.processing_job_id)
        if job is None:
            continue
        if item.state == BatchItemState.RUNNING.value:
            document_service.detect_and_fail_stale_job(db, job)
        sync_item_with_job(db, item, job)


# ----------------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------------

@dataclass
class ItemView:
    item: OCRBatchItem
    document: Document
    job: Optional[ProcessingJob]
    processed_pages: int


def _processed_pages_by_item(db: Session, batch_id: str) -> dict[str, int]:
    """Pages this item's CURRENT run has already written. Scoped to pages
    touched at/after the current job started, so a retry is not credited
    with pages a previous failed attempt left behind."""
    rows = db.execute(
        select(OCRBatchItem.id, func.count(DocumentPage.id))
        .join(ProcessingJob, ProcessingJob.id == OCRBatchItem.processing_job_id)
        .join(
            DocumentPage,
            and_(DocumentPage.document_id == OCRBatchItem.document_id, DocumentPage.updated_at >= ProcessingJob.started_at),
        )
        .where(OCRBatchItem.batch_id == batch_id, ProcessingJob.started_at.is_not(None))
        .group_by(OCRBatchItem.id)
    ).all()
    return {item_id: count for item_id, count in rows}


def load_item_views(db: Session, batch_id: str) -> list[ItemView]:
    rows = db.execute(
        select(OCRBatchItem, Document, ProcessingJob)
        .join(Document, Document.id == OCRBatchItem.document_id)
        .outerjoin(ProcessingJob, ProcessingJob.id == OCRBatchItem.processing_job_id)
        .where(OCRBatchItem.batch_id == batch_id)
        .order_by(OCRBatchItem.position)
    ).all()
    processed = _processed_pages_by_item(db, batch_id)
    views: list[ItemView] = []
    for item, document, job in rows:
        total = document.page_count or 0
        if item.state in (BatchItemState.COMPLETED.value,) or (
            item.state == BatchItemState.DUPLICATE.value and document.status == DocumentStatus.COMPLETED
        ):
            done = total
        else:
            done = min(total, processed.get(item.id, 0))
        views.append(ItemView(item=item, document=document, job=job, processed_pages=done))
    return views


def _item_read(view: ItemView) -> BatchItemRead:
    item, doc, job = view.item, view.document, view.job
    duration = None
    if job is not None and job.started_at is not None and job.finished_at is not None:
        duration = max(0.0, (job.finished_at - job.started_at).total_seconds())
    error = item.error_message
    if not error and job is not None and job.status == DocumentStatus.FAILED:
        error = job.error_message
    return BatchItemRead(
        job_id=item.id,
        document_id=item.document_id,
        processing_job_id=item.processing_job_id,
        filename=doc.original_filename,
        file_size_bytes=doc.file_size_bytes,
        position=item.position,
        state=item.state,
        status=job.status if job is not None else doc.status,
        stage=job.stage if job is not None else None,
        progress_percent=100 if item.state == BatchItemState.COMPLETED.value else (job.progress_percent if job is not None else 0),
        processed_pages=view.processed_pages,
        total_pages=doc.page_count or 0,
        priority_score=round(item.priority_score, 4),
        attempts=item.attempts or 0,
        error_message=error,
        started_at=job.started_at if job is not None else None,
        finished_at=job.finished_at if job is not None else None,
        duration_seconds=duration,
    )


def build_batch_status(db: Session, batch: OCRBatch, with_concurrency: bool = True) -> BatchRead:
    """Everything the queue page needs, in a fixed number of queries
    regardless of how many files the batch holds."""
    refresh_running_items(db, batch)
    views = load_item_views(db, batch.id)
    items = [_item_read(v) for v in views]

    def count(state: BatchItemState) -> int:
        return sum(1 for v in views if v.item.state == state.value)

    duplicates_done = sum(
        1 for v in views if v.item.state == BatchItemState.DUPLICATE.value and v.document.status == DocumentStatus.COMPLETED
    )
    work = [v for v in views if v.item.state != BatchItemState.DUPLICATE.value]
    total_pages = sum(v.document.page_count or 0 for v in work)
    processed_pages = sum(v.processed_pages for v in work)
    finished = sum(1 for v in views if v.item.state in _TERMINAL_VALUES)
    label = batch_status_label(batch, views)

    finished_at = None
    elapsed = None
    if batch.started_at is not None:
        started = batch.started_at if batch.started_at.tzinfo else batch.started_at.replace(tzinfo=timezone.utc)
        if label in ("completed", "completed_with_errors", "cancelled"):
            ends = [v.job.finished_at for v in work if v.job is not None and v.job.finished_at is not None]
            if ends:
                finished_at = max(ends)
        end = finished_at or datetime.now(timezone.utc)
        end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        elapsed = max(0.0, (end - started).total_seconds())

    concurrency = None
    if with_concurrency:
        decision = batch_scheduler.resolve_concurrency(batch.ocr_provider)
        concurrency = ConcurrencyInfo(**decision.as_dict())

    return BatchRead(
        batch_id=batch.id,
        status=label,
        ocr_provider=OCRProviderEnum(batch.ocr_provider),
        dpi=batch.dpi,
        preprocess_profile=PreprocessProfileEnum(batch.preprocess_profile),
        created_at=batch.created_at,
        started_at=batch.started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        total_files=len(views),
        completed_files=count(BatchItemState.COMPLETED) + duplicates_done,
        failed_files=count(BatchItemState.FAILED),
        cancelled_files=count(BatchItemState.CANCELLED),
        duplicate_files=count(BatchItemState.DUPLICATE),
        queued_files=count(BatchItemState.QUEUED) + count(BatchItemState.PENDING),
        processing_files=count(BatchItemState.RUNNING),
        finished_files=finished,
        total_pages=total_pages,
        processed_pages=processed_pages,
        # Falls back to a file-count ratio when there are no non-duplicate
        # pages to count (e.g. every file in the batch was a duplicate) --
        # without this a fully "completed" batch would show 0%.
        overall_percent=(
            int(100 * processed_pages / total_pages)
            if total_pages
            else (int(100 * finished / len(views)) if views else 0)
        ),
        concurrency=concurrency,
        items=items,
    )


def batch_status_label(batch: OCRBatch, views: list[ItemView]) -> str:
    if batch.started_at is None:
        return "uploading"
    if any(v.item.state not in _TERMINAL_VALUES for v in views):
        return "processing"
    states = {v.item.state for v in views}
    if states <= {BatchItemState.COMPLETED.value, BatchItemState.DUPLICATE.value}:
        return "completed"
    if states == {BatchItemState.CANCELLED.value}:
        return "cancelled"
    return "completed_with_errors"


# ----------------------------------------------------------------------------
# Retry
# ----------------------------------------------------------------------------

def retry_item(db: Session, batch: OCRBatch, item: OCRBatchItem) -> OCRBatchItem:
    """Re-queues ONE failed/cancelled file under the same item id with a
    fresh ProcessingJob (exactly what the existing reprocess endpoint
    does); the rest of the batch is untouched."""
    if item.batch_id != batch.id:
        raise BatchError("File not found in this batch.", status_code=404)
    if item.state not in (BatchItemState.FAILED.value, BatchItemState.CANCELLED.value):
        raise BatchError("Only a failed or cancelled file can be retried.", status_code=409)

    document = db.get(Document, item.document_id)
    if document is None or document.is_deleted:
        raise BatchError("The original document no longer exists.", status_code=404)

    job = document_service.create_processing_job(
        db, document, OCRProviderEnum(batch.ocr_provider), batch.dpi, PreprocessProfileEnum(batch.preprocess_profile)
    )
    item.processing_job_id = job.id
    item.state = BatchItemState.QUEUED.value
    item.error_message = None
    item.claimed_at = None
    item.celery_task_id = None
    db.commit()
    db.refresh(item)
    logger.info("batch_file_retry", batch_id=batch.id, item_id=item.id, attempt=(item.attempts or 0) + 1)
    return item
