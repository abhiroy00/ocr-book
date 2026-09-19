"""
Multiple-file OCR API. An add-on beside `documents.py`: it never changes a
single-file endpoint, and every file it accepts becomes an ordinary
Document that the existing document endpoints (view, edit, export, cancel)
keep working on.

Flow (one request per file keeps each upload single-file-sized, so the
existing nginx `client_max_body_size` needs no change and only one file's
bytes are in memory at a time):
    POST /batches                          -> create, apply engine/DPI/profile to the whole batch
    POST /batches/{id}/files   (x N)       -> upload one file each
    POST /batches/{id}/start               -> prioritise + enqueue
    GET  /batches/{id}                     -> live status of the batch and every file
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_storage_backend
from app.api.v1.documents import cancel_processing
from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum
from app.models.ocr_batch import BatchItemState, OCRBatch, OCRBatchItem
from app.models.processing_job import ProcessingJob
from app.schemas.batch import (
    BatchCapacityRead,
    BatchCreateRequest,
    BatchItemRead,
    BatchRead,
    BatchStartResponse,
    ConcurrencyInfo,
)
from app.services import batch_scheduler, batch_service, document_service
from app.services.batch_service import BatchError
from app.services.storage import StorageBackend

logger = get_logger(__name__)

router = APIRouter(prefix="/batches", tags=["batches"])

_SLOT_TASK = "pipeline.run_batch_slot"


def _get_batch_or_404(db: Session, batch_id: str) -> OCRBatch:
    batch = db.get(OCRBatch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


def _raise_http(exc: BatchError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _enqueue_slots(batch_id: str, count: int) -> None:
    for _ in range(count):
        celery_app.send_task(_SLOT_TASK, args=[batch_id])


@router.post("", response_model=BatchRead, status_code=status.HTTP_201_CREATED)
def create_batch(
    body: BatchCreateRequest, db: Session = Depends(get_db), user: Optional[str] = Depends(get_current_user)
):
    settings = get_settings()
    dpi = body.dpi or settings.default_dpi
    if dpi not in settings.allowed_dpi_list:
        raise HTTPException(status_code=422, detail=f"dpi must be one of {settings.allowed_dpi_list}")
    batch = batch_service.create_batch(
        db,
        body.ocr_provider or OCRProviderEnum(settings.ocr_provider.value),
        dpi,
        body.preprocess_profile or PreprocessProfileEnum(settings.preprocess_profile.value),
        uploaded_by=user,
    )
    return batch_service.build_batch_status(db, batch)


# Registered before "/{batch_id}" so "capacity" is not parsed as a batch id.
@router.get("/capacity", response_model=BatchCapacityRead)
def batch_capacity(ocr_provider: Optional[OCRProviderEnum] = None):
    settings = get_settings()
    provider = (ocr_provider or OCRProviderEnum(settings.ocr_provider.value)).value
    decision = batch_scheduler.resolve_concurrency(provider)
    return BatchCapacityRead(
        concurrency=ConcurrencyInfo(**decision.as_dict()),
        max_files=settings.batch_max_files,
        max_total_mb=settings.batch_max_total_mb,
        max_file_mb=settings.max_upload_size_mb,
        allowed_extensions=settings.allowed_extensions_list,
    )


@router.post("/{batch_id}/files", response_model=BatchItemRead, status_code=status.HTTP_201_CREATED)
async def add_batch_file(
    batch_id: str,
    file: UploadFile = File(...),
    force: bool = Form(default=False),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    user: Optional[str] = Depends(get_current_user),
):
    batch = _get_batch_or_404(db, batch_id)
    content = await file.read()
    try:
        item = batch_service.add_file(db, storage, batch, file.filename or "upload", content, force=force, uploaded_by=user)
    except BatchError as exc:
        _raise_http(exc)
    finally:
        del content  # one file's bytes at a time; don't hold them past this request
    status_view = batch_service.build_batch_status(db, batch, with_concurrency=False)
    return next(i for i in status_view.items if i.job_id == item.id)


@router.post("/{batch_id}/start", response_model=BatchStartResponse)
def start_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = _get_batch_or_404(db, batch_id)
    try:
        queued = batch_service.start_batch(db, batch)
    except BatchError as exc:
        _raise_http(exc)
    try:
        _enqueue_slots(batch.id, queued)
    except Exception as exc:  # noqa: BLE001 - broker down: undo the start so the client can simply retry it
        logger.error("batch_enqueue_failed", batch_id=batch.id, error=str(exc))
        batch_service.revert_start(db, batch)
        raise HTTPException(status_code=503, detail="Could not queue the batch (task broker unreachable). Please retry.") from exc
    logger.info("batch_started", batch_id=batch.id, queued=queued)
    return BatchStartResponse(batch_id=batch.id, queued_files=queued, status="processing" if queued else "already_started")


@router.get("/{batch_id}", response_model=BatchRead)
def get_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = _get_batch_or_404(db, batch_id)
    return batch_service.build_batch_status(db, batch)


@router.get("/{batch_id}/items/{job_id}", response_model=BatchItemRead)
def get_batch_item(batch_id: str, job_id: str, db: Session = Depends(get_db)):
    batch = _get_batch_or_404(db, batch_id)
    view = batch_service.build_batch_status(db, batch, with_concurrency=False)
    for item in view.items:
        if item.job_id == job_id:
            return item
    raise HTTPException(status_code=404, detail="File not found in this batch")


@router.post("/{batch_id}/items/{job_id}/retry", response_model=BatchItemRead)
def retry_batch_item(batch_id: str, job_id: str, db: Session = Depends(get_db)):
    batch = _get_batch_or_404(db, batch_id)
    item = db.get(OCRBatchItem, job_id)
    if item is None or item.batch_id != batch.id:
        raise HTTPException(status_code=404, detail="File not found in this batch")
    try:
        batch_service.retry_item(db, batch, item)
    except BatchError as exc:
        _raise_http(exc)
    _enqueue_slots(batch.id, 1)
    view = batch_service.build_batch_status(db, batch, with_concurrency=False)
    return next(i for i in view.items if i.job_id == item.id)


@router.post("/{batch_id}/cancel")
def cancel_batch(batch_id: str, db: Session = Depends(get_db)):
    """Cancels everything not yet finished. Files still waiting are dropped
    immediately; a file already in OCR is stopped by the existing
    cooperative per-document cancel (the page in flight finishes first),
    so no file is ever left half-written."""
    batch = _get_batch_or_404(db, batch_id)
    items = db.scalars(select(OCRBatchItem).where(OCRBatchItem.batch_id == batch.id)).all()
    cancelled_waiting = 0
    cancel_requested = 0
    for item in items:
        if item.state in (BatchItemState.PENDING.value, BatchItemState.QUEUED.value):
            item.state = BatchItemState.CANCELLED.value
            job = db.get(ProcessingJob, item.processing_job_id) if item.processing_job_id else None
            db.commit()
            if job is not None:
                document_service.update_job_progress(
                    db, job, job.stage, job.progress_percent, DocumentStatus.CANCELLED, message="Cancelled by user"
                )
            cancelled_waiting += 1
        elif item.state == BatchItemState.RUNNING.value:
            try:
                cancel_processing(item.document_id, db)
                cancel_requested += 1
            except HTTPException:
                pass  # already finishing on its own -- nothing left to cancel
    logger.info("batch_cancelled", batch_id=batch.id, waiting=cancelled_waiting, running=cancel_requested)
    return {"status": "cancel_requested", "cancelled_waiting": cancelled_waiting, "cancelling_running": cancel_requested}
