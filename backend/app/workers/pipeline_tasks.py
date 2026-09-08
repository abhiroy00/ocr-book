"""
The Celery pipeline (spec sections 3 & 39, phases 3-11): ingest -> render ->
preprocess -> OCR -> layout -> table -> Document JSON -> reconstruct ->
export -> quality validation.

Pages are processed one at a time (`iter_render_pages` streams from
PyMuPDF), and only lightweight per-page results (Document JSON, not raw
pixel arrays) are kept across the whole document, so large (150-500 page)
documents never require the full page-image set in memory at once (spec
section 28).

Every stage that can plausibly fail on a single page (OCR engine hiccup,
table detector edge case) is wrapped so one bad page degrades gracefully
rather than failing the entire document (spec section 29).
"""
from __future__ import annotations

import os
import uuid

import cv2
import redis
from celery.signals import task_failure

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.logging import get_logger, stage_timer
from app.db.session import SessionLocal
from app.layout.detector import LayoutDetector
from app.models.document import Document
from app.models.enums import DocumentStatus, ExportType, LayoutBlockType, ProcessingStage
from app.models.processing_job import ProcessingJob
from app.ocr.factory import get_ocr_provider
from app.ocr.subprocess_runner import IsolatedOCRWorker
from app.reconstruction import clean_pdf, pdf_renderer, searchable_pdf
from app.reconstruction.fonts import resolve_body_font_path
from app.services import document_service
from app.services.quality import compare_images
from app.services.storage import get_storage
from app.services.pdf_ingest import iter_render_pages
from app.tables.engine import detect_tables
from app.exporters.docx_exporter import render_document_docx
from app.vision.preprocessor import PreprocessProfile, preprocess_page

logger = get_logger(__name__)

_GRAPHIC_TYPES = {
    LayoutBlockType.IMAGE,
    LayoutBlockType.CHART,
    LayoutBlockType.SIGNATURE,
    LayoutBlockType.STAMP,
    LayoutBlockType.HANDWRITTEN,
}

_DOCUMENT_LOCK_PREFIX = "pipeline:lock:document:"
# A lease, not a one-shot lock: short enough that a hard-killed holder (e.g.
# PaddlePaddle's native runtime aborting the whole process on SIGTERM
# instead of unwinding Python and running `finally` -- an observed, real
# failure mode, not hypothetical) self-heals in minutes rather than hours;
# renewed after every page so a genuinely still-running job never loses it
# mid-document. 10 minutes covers the slowest single page seen in practice
# (first-page model loading) with real margin.
_DOCUMENT_LOCK_TTL_SECONDS = 600
_RENEW_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) else return 0 end"
_RELEASE_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"


def _acquire_document_lock(document_id: str) -> str | None:
    """Redis `SET NX EX` lock, keyed by document_id. Returns a token to
    release/renew with, or None if another instance already holds it (a
    redelivered/duplicate task should skip itself, not run alongside the
    original). Self-heals via TTL if a holder dies without releasing it."""
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    token = uuid.uuid4().hex
    try:
        acquired = client.set(_DOCUMENT_LOCK_PREFIX + document_id, token, nx=True, ex=_DOCUMENT_LOCK_TTL_SECONDS)
        return token if acquired else None
    finally:
        client.close()


def _renew_document_lock(document_id: str, token: str) -> None:
    """Extends the lease. Called after every page so a long-running,
    genuinely-alive job's lock never expires out from under it, while a
    job that dies without releasing still self-heals within one TTL
    window of its last renewal instead of sitting stuck for hours."""
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.eval(_RENEW_LUA, 1, _DOCUMENT_LOCK_PREFIX + document_id, token, str(_DOCUMENT_LOCK_TTL_SECONDS))
    except Exception as exc:  # noqa: BLE001 - a missed renewal just means the lock expires a bit early; not fatal
        logger.warning("document_lock_renew_failed", document_id=document_id, error=str(exc))
    finally:
        client.close()


def _release_document_lock(document_id: str, token: str) -> None:
    """Only releases the lock if it still holds OUR token (avoids releasing
    a lock some other, later task legitimately acquired after ours expired)."""
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.eval(_RELEASE_LUA, 1, _DOCUMENT_LOCK_PREFIX + document_id, token)
    except Exception as exc:  # noqa: BLE001 - releasing is best-effort; the TTL is the real safety net
        logger.warning("document_lock_release_failed", document_id=document_id, error=str(exc))
    finally:
        client.close()


@celery_app.task(bind=True, name="pipeline.process_document")
def process_document(self, document_id: str, job_id: str) -> str:
    db = SessionLocal()
    storage = get_storage()
    lock_token = None
    try:
        document = db.get(Document, document_id)
        job = db.get(ProcessingJob, job_id)
        if document is None or job is None:
            logger.error("process_document_missing_record", document_id=document_id, job_id=job_id)
            return "missing"

        # `task_acks_late=True` (so a killed worker's task gets retried, not
        # silently lost) means restarting/recreating the worker mid-task
        # causes Celery to redeliver that same task -- while the original
        # process may *also* still be finishing up, or a second redelivery
        # stacks on a first. Without a guard, two copies of the same
        # document's pipeline end up racing, both writing pages/OCR/table
        # rows and both burning a worker slot. A Redis lock keyed by
        # document_id makes any redelivered/duplicate invocation a no-op
        # instead of a second full run.
        lock_token = _acquire_document_lock(document_id)
        if lock_token is None:
            # Returning here without touching `job` would leave this
            # specific ProcessingJob permanently stuck at QUEUED (Celery
            # sees a clean "completed" return, so it never retries this
            # task, and the DB row was never updated to say otherwise) --
            # the job needs an explicit, visible terminal state instead.
            logger.warning("duplicate_task_skipped", document_id=document_id, job_id=job_id)
            document_service.update_job_progress(
                db,
                job,
                job.stage,
                job.progress_percent,
                DocumentStatus.FAILED,
                error="Skipped: another processing run for this document was already active. Re-run to try again.",
            )
            return "duplicate_skipped"

        job.celery_task_id = self.request.id
        db.commit()

        _run_pipeline(db, storage, document, job, lock_token)
        return "completed"
    except Exception as exc:  # noqa: BLE001 - top-level guard: never leave a job stuck mid-status
        logger.error("pipeline_failed", document_id=document_id, error=str(exc))
        job = db.get(ProcessingJob, job_id)
        if job:
            document_service.update_job_progress(
                db, job, ProcessingStage.DONE, job.progress_percent, DocumentStatus.FAILED, error=str(exc)
            )
        raise
    finally:
        if lock_token is not None:
            _release_document_lock(document_id, lock_token)
        db.close()


@task_failure.connect
def _on_any_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, **extra) -> None:
    """Safety net for failures that never reach `process_document`'s own
    try/except: specifically a `WorkerLostError` from the Celery/billiard
    pool when the *worker process itself* was killed by a native crash
    (e.g. a SIGILL from a paddlepaddle build incompatible with the host
    CPU/virtualization — see IsolatedOCRWorker and the `enable_pp_structure`
    flag). Without this, such a job is left stuck in its last in-progress
    status forever with no error ever surfaced to the user."""
    task_name = getattr(sender, "name", None) if sender else None
    if task_name != "pipeline.process_document" or not args or len(args) < 2:
        return

    document_id, job_id = args[0], args[1]
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if job is None or job.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED):
            return  # already terminal, or the in-process handler already recorded it
        document_service.update_job_progress(
            db, job, job.stage, job.progress_percent, DocumentStatus.FAILED,
            error=f"Processing worker crashed: {exception}",
        )
        logger.error("pipeline_worker_lost", document_id=document_id, job_id=job_id, error=str(exception))
    finally:
        db.close()


def _run_pipeline(db, storage, document: Document, job: ProcessingJob, lock_token: str) -> None:
    settings = get_settings()
    # Resolve which provider is actually available (may already fall back
    # e.g. paddleocr -> tesseract here if paddleocr isn't installed at all).
    resolved_provider_name = get_ocr_provider(job.ocr_provider.value).name
    profile = PreprocessProfile(job.preprocess_profile.value)

    document_service.update_job_progress(db, job, ProcessingStage.RENDER, 5, DocumentStatus.PROCESSING, message="Rendering pages")

    original_bytes = storage.read(document.storage_original_path)
    is_pdf = document.file_extension == ".pdf"
    total_pages = document.page_count or 1

    clean_doc = _open_fitz_doc()
    searchable_doc = _open_fitz_doc()
    font_path = resolve_body_font_path()
    all_page_jsons = []

    layout_detector = LayoutDetector()
    # OCR inference runs in a separate, long-lived subprocess so a native
    # engine crash (see IsolatedOCRWorker docstring) can never take the
    # Celery worker process itself down mid-document — it just falls back
    # to Tesseract in-process for the rest of this job.
    ocr_worker = IsolatedOCRWorker(resolved_provider_name)

    try:
        for rendered in iter_render_pages(original_bytes, is_pdf, job.dpi):
            page_progress_base = 10
            page_progress_span = 60  # 10..70% covers render+preprocess+ocr+layout+table across pages
            percent = page_progress_base + int(page_progress_span * (rendered.page_number - 1) / max(1, total_pages))

            with stage_timer(logger, document.id, "render", page=rendered.page_number):
                original_path = f"pages/{document.id}/page_{rendered.page_number:04d}_original.png"
                storage.write(original_path, _encode_png(rendered.image))

            db_page = document_service.upsert_page(
                db, document, rendered.page_number, rendered.width, rendered.height, rendered.dpi, rendered.rotation, original_path
            )

            document_service.update_job_progress(
                db, job, ProcessingStage.PREPROCESS, percent, DocumentStatus.PROCESSING, page=rendered.page_number, message="Cleaning image"
            )
            with stage_timer(logger, document.id, "preprocess", page=rendered.page_number):
                result = preprocess_page(rendered.image, profile)
            processed_image = result.image
            processed_path = f"processed/{document.id}/page_{rendered.page_number:04d}_processed.png"
            storage.write(processed_path, _encode_png(processed_image))

            document_service.update_job_progress(
                db, job, ProcessingStage.OCR, percent, DocumentStatus.OCR_PROCESSING, page=rendered.page_number, message="Running OCR"
            )
            with stage_timer(logger, document.id, "ocr", page=rendered.page_number):
                words = _run_ocr_page(ocr_worker, processed_image, rendered.page_number, rendered.dpi)

            document_service.update_job_progress(
                db, job, ProcessingStage.TABLE, percent, DocumentStatus.TABLE_PROCESSING, page=rendered.page_number, message="Detecting tables"
            )
            tables = _detect_tables_safely(processed_image, words, rendered.width, rendered.height)

            document_service.update_job_progress(
                db, job, ProcessingStage.LAYOUT, percent, DocumentStatus.LAYOUT_PROCESSING, page=rendered.page_number, message="Detecting layout"
            )
            layout_results = layout_detector.detect(processed_image, rendered.image, words, tables, rendered.width, rendered.height)

            _crop_and_attach_graphic_images(storage, document.id, rendered.page_number, processed_image, layout_results)

            page_json = document_service.persist_page_pipeline_result(
                db, document, db_page, processed_path, words, layout_results, tables
            )
            all_page_jsons.append(page_json)

            clean_pdf.add_clean_page(clean_doc, processed_image, rendered.dpi)
            searchable_pdf.add_searchable_page(searchable_doc, processed_image, rendered.dpi, words, font_path)

            _renew_document_lock(document.id, lock_token)
            del processed_image, rendered  # release page-sized arrays before the next iteration
    finally:
        ocr_worker.shutdown()

    document_service.update_job_progress(db, job, ProcessingStage.RECONSTRUCT, 75, DocumentStatus.RECONSTRUCTING, message="Rendering reconstructed PDF")
    reconstructed_bytes = pdf_renderer.render_document_pdf(all_page_jsons)

    document_service.update_job_progress(db, job, ProcessingStage.EXPORT, 85, DocumentStatus.EXPORTING, message="Exporting PDF/DOCX")
    clean_bytes = clean_doc.tobytes(deflate=True, garbage=4)
    clean_doc.close()
    searchable_bytes = searchable_doc.tobytes(deflate=True, garbage=4)
    searchable_doc.close()
    docx_bytes = render_document_docx(all_page_jsons)

    _store_export(db, storage, document, ExportType.CLEAN_PDF, f"output/{document.id}/clean.pdf", clean_bytes)
    _store_export(db, storage, document, ExportType.SEARCHABLE_PDF, f"output/{document.id}/searchable.pdf", searchable_bytes)
    _store_export(db, storage, document, ExportType.RECONSTRUCTED_PDF, f"output/{document.id}/reconstructed.pdf", reconstructed_bytes)
    _store_export(db, storage, document, ExportType.DOCX, f"output/{document.id}/document.docx", docx_bytes)

    document_service.update_job_progress(db, job, ProcessingStage.QUALITY, 95, DocumentStatus.EXPORTING, message="Scoring visual similarity")
    _run_quality_pass(db, storage, document, reconstructed_bytes)

    document_service.update_job_progress(db, job, ProcessingStage.DONE, 100, DocumentStatus.COMPLETED, message="Done")


def _run_ocr_page(ocr_worker: IsolatedOCRWorker, image, page_number: int, dpi: int):
    """OCR for one page: prefer the isolated primary-engine subprocess;
    fall back to in-process Tesseract once that subprocess has crashed or
    for any page it reports a (non-crash) failure on."""
    if ocr_worker.alive:
        page_result = ocr_worker.run_page(image, page_number, dpi)
        if page_result is not None:
            return page_result.words
        if not ocr_worker.poll_crashed():
            # The primary engine is still alive; this was just a normal
            # per-page OCR error (already logged by the subprocess) — an
            # empty page still gets a paragraph-less/table-less layout
            # pass rather than blocking the whole document.
            return []
        logger.warning("ocr_engine_crashed_falling_back_to_tesseract", page=page_number)

    try:
        fallback = get_ocr_provider("tesseract")
        return fallback.recognize_page(image, page_number, dpi).words
    except Exception as exc:  # noqa: BLE001 - never let one page's OCR failure kill the job
        logger.error("ocr_fully_failed", page=page_number, error=str(exc))
        return []


def _detect_tables_safely(image, words, width, height):
    try:
        return detect_tables(image, words, width, height)
    except Exception as exc:  # noqa: BLE001 - spec 29: table failure falls back to OCR positioning, never crashes
        logger.error("table_detection_failed", error=str(exc))
        return []


def _crop_and_attach_graphic_images(storage, document_id: str, page_number: int, image, layout_results) -> None:
    for i, result in enumerate(layout_results):
        if result.block_type not in _GRAPHIC_TYPES:
            continue
        x1, y1 = max(0, int(result.bbox.x1)), max(0, int(result.bbox.y1))
        x2, y2 = min(image.shape[1], int(result.bbox.x2)), min(image.shape[0], int(result.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        rel_path = f"processed/{document_id}/page_{page_number:04d}_block_{i:04d}.png"
        storage.write(rel_path, _encode_png(crop))
        result.image_ref = rel_path


def _run_quality_pass(db, storage, document: Document, reconstructed_pdf_bytes: bytes) -> None:
    import fitz

    from app.models.document_page import DocumentPage
    from sqlalchemy import select

    pages = db.scalars(
        select(DocumentPage).where(DocumentPage.document_id == document.id).order_by(DocumentPage.page_number)
    ).all()

    recon_doc = fitz.open(stream=reconstructed_pdf_bytes, filetype="pdf")
    try:
        for page in pages:
            idx = page.page_number - 1
            if idx >= recon_doc.page_count or not page.processed_image_path:
                continue
            try:
                processed_bytes = storage.read(page.processed_image_path)
                import numpy as np

                processed_image = cv2.imdecode(np.frombuffer(processed_bytes, dtype="uint8"), cv2.IMREAD_COLOR)

                zoom = page.dpi / 72.0
                pix = recon_doc.load_page(idx).get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                recon_arr = np.frombuffer(pix.samples, dtype="uint8").reshape(pix.height, pix.width, pix.n)
                recon_bgr = cv2.cvtColor(recon_arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(recon_arr, cv2.COLOR_RGBA2BGR)

                similarity = compare_images(processed_image, recon_bgr)
                page.visual_similarity_score = similarity.ssim_score
            except Exception as exc:  # noqa: BLE001 - quality scoring must never fail the job
                logger.warning("quality_scoring_failed", page=page.page_number, error=str(exc))
        db.commit()
    finally:
        recon_doc.close()


def _store_export(db, storage, document: Document, export_type: ExportType, relative_path: str, data: bytes) -> None:
    storage.write(relative_path, data)
    document_service.record_export_file(db, document, export_type, relative_path, len(data))


def _encode_png(image) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode page image as PNG")
    return encoded.tobytes()


def _open_fitz_doc():
    import fitz

    return fitz.open()
