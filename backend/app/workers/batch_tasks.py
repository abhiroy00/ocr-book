"""
Celery task for multiple-file OCR.

One `pipeline.run_batch_slot` task is enqueued per file in a batch, but the
tasks are interchangeable: each one, when a worker picks it up, *claims* the
best next file for its batch (see `batch_scheduler.claim_next_item` -- small
files first, one long lane, host-wide concurrency limit) and runs it through
the SAME `run_document_pipeline` a single-file upload uses. There is no
second OCR implementation here; this module only sequences and isolates.

Design points:
  * Document-level parallelism = one Celery prefork process per document.
    Each document then forks its own page-worker pool exactly as before, so
    the fork-after-threads hazard documented in celery_app.py cannot occur.
  * A claim that finds the host already at its concurrency limit does not
    hold the worker: it re-queues itself and tries again shortly.
  * One file's failure never stops the batch: exceptions are recorded on
    that file's item and the task ends normally so the next file proceeds.
"""
from __future__ import annotations

from celery.signals import task_failure

from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.enums import DocumentStatus, ProcessingStage
from app.models.ocr_batch import TERMINAL_ITEM_STATES, BatchItemState, OCRBatch, OCRBatchItem
from app.models.processing_job import ProcessingJob
from app.ocr.factory import get_ocr_provider
from app.services import batch_scheduler, batch_service, document_service
from app.services.batch_scheduler import Claim, ClaimOutcome
from app.workers.pipeline_tasks import run_document_pipeline

logger = get_logger(__name__)

CLAIM_RETRY_SECONDS = 20
_TERMINAL_VALUES = {s.value for s in TERMINAL_ITEM_STATES}


def _effective_provider(requested: str) -> str:
    """The engine that will REALLY run (requested -> paddleocr -> tesseract,
    same fallback chain as the pipeline), so a batch that asks for an
    unavailable NVIDIA/Ollama is admitted as the heavy Paddle job it will
    actually become, not as a cheap remote-API job. PaddleOCR itself is not
    probed: that would import the whole paddle runtime just to decide a
    limit, and assuming Paddle is also the conservative choice."""
    if requested == "paddleocr":
        return requested
    try:
        return get_ocr_provider(requested).name
    except Exception:  # noqa: BLE001 - no engine at all: the pipeline will fail the file clearly; be conservative here
        return "paddleocr"


def _claim(batch_id: str, task_id: str | None) -> tuple[ClaimOutcome, Claim | None]:
    db = SessionLocal()
    try:
        batch = db.get(OCRBatch, batch_id)
        if batch is None:
            return ClaimOutcome.EMPTY, None
        provider = _effective_provider(batch.ocr_provider)
        return batch_scheduler.claim_next_item(db, batch_id, task_id, provider)
    finally:
        db.close()


def _finalize_item(item_id: str, error: str | None) -> None:
    """Mirror the pipeline's outcome onto the batch item; if the pipeline
    never reached a terminal state (an exception before it could record
    one) make the failure visible rather than leaving the file 'running'."""
    db = SessionLocal()
    try:
        item = db.get(OCRBatchItem, item_id)
        if item is None:
            return
        batch_service.sync_item_with_job(db, item)
        if item.state not in _TERMINAL_VALUES:
            item.state = BatchItemState.FAILED.value
            item.error_message = error or "Processing ended without a result"
            db.commit()
    finally:
        db.close()


@celery_app.task(bind=True, name="pipeline.run_batch_slot", max_retries=None)
def run_batch_slot(self, batch_id: str) -> str:
    outcome, claim = _claim(batch_id, self.request.id)

    if outcome is ClaimOutcome.EMPTY:
        return "empty"
    if outcome is ClaimOutcome.NO_CAPACITY:
        logger.info("batch_slot_waiting", batch_id=batch_id, retry_in=CLAIM_RETRY_SECONDS)
        raise self.retry(countdown=CLAIM_RETRY_SECONDS)

    assert claim is not None
    logger.info(
        "batch_slot_started", batch_id=batch_id, item_id=claim.item_id, document_id=claim.document_id,
        provider=claim.provider, concurrency_limit=claim.limit, pool_cap=claim.pool_cap,
    )
    error: str | None = None
    try:
        run_document_pipeline(self.request.id, claim.document_id, claim.job_id, claim.pool_cap)
    except Exception as exc:  # noqa: BLE001 - error isolation: the pipeline already marked THIS file's job FAILED; the batch must carry on
        error = str(exc)
        logger.error("batch_file_failed", batch_id=batch_id, item_id=claim.item_id, document_id=claim.document_id, error=error)
    finally:
        _finalize_item(claim.item_id, error)

    logger.info("batch_slot_finished", batch_id=batch_id, item_id=claim.item_id, ok=error is None)
    return "failed" if error else "done"


@task_failure.connect
def _on_batch_task_failure(sender=None, task_id=None, exception=None, **extra) -> None:
    """Safety net for a batch slot whose worker process was killed outright
    (e.g. a native crash surfaces as `WorkerLostError`): the in-task
    handling above never got to run, so without this the file would stay
    'running' forever. Mirrors what pipeline_tasks does for single-file
    tasks, keyed by the Celery task id recorded at claim time."""
    if getattr(sender, "name", None) != "pipeline.run_batch_slot" or not task_id:
        return
    db = SessionLocal()
    try:
        item = db.query(OCRBatchItem).filter(OCRBatchItem.celery_task_id == task_id).first()
        if item is None or item.state in _TERMINAL_VALUES:
            return
        message = f"Processing worker crashed: {exception}"
        job = db.get(ProcessingJob, item.processing_job_id) if item.processing_job_id else None
        if job is not None and job.status not in (DocumentStatus.COMPLETED, DocumentStatus.FAILED, DocumentStatus.CANCELLED):
            document_service.update_job_progress(
                db, job, job.stage or ProcessingStage.DONE, job.progress_percent, DocumentStatus.FAILED, error=message
            )
        item.state = BatchItemState.FAILED.value
        item.error_message = message
        db.commit()
        logger.error("batch_worker_lost", item_id=item.id, task_id=task_id, error=str(exception))
    finally:
        db.close()
