"""Documents API (spec section 31)."""
from __future__ import annotations

from typing import Optional

import redis
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_storage_backend
from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.locks import DOCUMENT_CANCEL_PREFIX
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.enums import DocumentStatus, ExportType, OCRProviderEnum, PreprocessProfileEnum, TextAlign, VerticalAlign
from app.models.export_file import ExportFile
from app.models.layout_block import LayoutBlock
from app.models.ocr_block import OCRBlock
from app.models.processing_job import ProcessingJob
from app.models.table import Table
from app.models.table_cell import TableCell
from app.schemas.document import (
    DocumentListResponse,
    DocumentRead,
    DocumentStats,
    DocumentUploadResponse,
    PageRead,
    ProcessingJobRead,
    ProcessRequest,
    QualityPageReport,
    QualityReport,
)
from app.schemas.layout import LayoutBlockRead, LayoutBlockUpdate
from app.schemas.table import TableCellRead, TableCellUpdate, TableRead, TableUpdate
from app.services import document_service
from app.services.quality import needs_review
from app.services.storage import StorageBackend
from app.utils.file_safety import UploadValidationError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    ocr_provider: Optional[OCRProviderEnum] = Form(default=None),
    dpi: Optional[int] = Form(default=None),
    preprocess_profile: Optional[PreprocessProfileEnum] = Form(default=None),
    force: bool = Form(default=False),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    user: Optional[str] = Depends(get_current_user),
):
    settings = get_settings()
    content = await file.read()

    resolved_dpi = dpi or settings.default_dpi
    if resolved_dpi not in settings.allowed_dpi_list:
        raise HTTPException(status_code=422, detail=f"dpi must be one of {settings.allowed_dpi_list}")

    # Duplicate-upload detection (spec section 9): a content-hash match
    # against an already-processed document is reported back rather than
    # silently reprocessed -- `force=true` explicitly opts into
    # reprocessing anyway (the existing reprocess workflow already
    # supports this need; this just guards the common "oops, uploaded the
    # same PDF twice" case from creating a second, wasteful full OCR run).
    if not force:
        existing = document_service.find_document_by_hash(db, document_service.compute_document_hash(content))
        if existing is not None:
            latest_job = db.scalar(
                select(ProcessingJob).where(ProcessingJob.document_id == existing.id).order_by(ProcessingJob.created_at.desc())
            )
            return DocumentUploadResponse(
                document_id=existing.id, job_id=latest_job.id if latest_job else "", status=existing.status,
                original_filename=existing.original_filename, page_count=existing.page_count, is_duplicate=True,
            )

    try:
        document, job = document_service.create_document_from_upload(
            db,
            storage,
            file.filename or "upload",
            content,
            ocr_provider or settings.ocr_provider,
            resolved_dpi,
            preprocess_profile or settings.preprocess_profile,
            uploaded_by=user,
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async_result = celery_app.send_task("pipeline.process_document", args=[document.id, job.id])
    # Captured here too (not only inside the task itself at `self.request.id`)
    # so `/cancel` can revoke a job that is still sitting QUEUED and hasn't
    # started its first page yet, not only one already mid-OCR.
    job.celery_task_id = async_result.id
    db.commit()

    return DocumentUploadResponse(
        document_id=document.id, job_id=job.id, status=document.status, original_filename=document.original_filename, page_count=document.page_count
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    status_filter: Optional[DocumentStatus] = None, page: int = 1, page_size: int = 20, db: Session = Depends(get_db)
):
    items, total = document_service.list_documents(db, status_filter, page, page_size)
    return DocumentListResponse(items=[DocumentRead.model_validate(d) for d in items], total=total, page=page, page_size=page_size)


@router.get("/stats/summary", response_model=DocumentStats)
def document_stats(db: Session = Depends(get_db)):
    return DocumentStats(**document_service.get_stats(db))


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = _get_document_or_404(db, document_id)
    return DocumentRead.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    document = _get_document_or_404(db, document_id)
    document_service.soft_delete_document(db, storage, document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{document_id}/process", response_model=ProcessingJobRead)
def reprocess_document(document_id: str, body: ProcessRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    document = _get_document_or_404(db, document_id)
    job = document_service.create_processing_job(
        db,
        document,
        body.ocr_provider or document.ocr_provider,
        body.dpi or document.dpi,
        body.preprocess_profile or document.preprocess_profile,
    )
    async_result = celery_app.send_task("pipeline.process_document", args=[document.id, job.id])
    job.celery_task_id = async_result.id
    db.commit()
    return ProcessingJobRead.model_validate(job)


@router.get("/{document_id}/progress", response_model=ProcessingJobRead)
def get_progress(document_id: str, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    job = db.scalar(
        select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc())
    )
    if job is None:
        raise HTTPException(status_code=404, detail="No processing job found for this document")
    job = document_service.detect_and_fail_stale_job(db, job)
    return ProcessingJobRead.model_validate(job)


_CANCELLABLE_STATUSES = {
    DocumentStatus.QUEUED,
    DocumentStatus.PROCESSING,
    DocumentStatus.OCR_PROCESSING,
    DocumentStatus.LAYOUT_PROCESSING,
    DocumentStatus.TABLE_PROCESSING,
    DocumentStatus.RECONSTRUCTING,
    DocumentStatus.EXPORTING,
}


@router.post("/{document_id}/cancel")
def cancel_processing(document_id: str, db: Session = Depends(get_db)):
    """Stops an in-flight pipeline run. Cooperative, not a hard kill: sets
    a Redis flag the running task checks between pages (see
    `app.workers.pipeline_tasks._is_cancel_requested`), so the page
    currently being processed always finishes cleanly and the document is
    left with whatever pages already completed rather than a half-written
    one. Also revokes the Celery task outright, which matters if it is
    still queued and hasn't started page 1 yet (the cooperative flag alone
    would never get checked in that case)."""
    _get_document_or_404(db, document_id)
    job = db.scalar(
        select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc())
    )
    if job is None or job.status not in _CANCELLABLE_STATUSES:
        raise HTTPException(status_code=400, detail="Document is not currently processing")

    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.set(DOCUMENT_CANCEL_PREFIX + document_id, "1", ex=3600)
    finally:
        client.close()

    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id)

    # Covers the case where the task hasn't started yet (still QUEUED) and
    # so will never reach the cooperative flag check at all now that it's
    # revoked -- without this the job would sit at QUEUED forever. If the
    # task IS already mid-page, its own next per-page progress update can
    # briefly overwrite this back to an active status before the flag
    # check catches up on the following page; that self-corrects within
    # one page's processing time and is not worth adding coordination for.
    document_service.update_job_progress(
        db, job, job.stage, job.progress_percent, DocumentStatus.CANCELLED, message="Cancelled by user"
    )

    return {"status": "cancel_requested"}


@router.get("/{document_id}/pages", response_model=list[PageRead])
def list_pages(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    _get_document_or_404(db, document_id)
    pages = db.scalars(select(DocumentPage).where(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number)).all()
    return [_page_to_read(p, storage) for p in pages]


@router.get("/{document_id}/pages/{page_number}", response_model=PageRead)
def get_page(document_id: str, page_number: int, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    page = _get_page_or_404(db, document_id, page_number)
    return _page_to_read(page, storage)


@router.get("/{document_id}/pages/{page_number}/json")
def get_page_json(document_id: str, page_number: int, db: Session = Depends(get_db)):
    page = _get_page_or_404(db, document_id, page_number)
    if not page.document_json:
        raise HTTPException(status_code=404, detail="Page has not been processed yet")
    return page.document_json


@router.get("/{document_id}/ocr", response_model=list[dict])
def get_ocr_blocks(document_id: str, page: Optional[int] = None, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    stmt = select(OCRBlock).where(OCRBlock.document_id == document_id)
    if page is not None:
        page_row = _get_page_or_404(db, document_id, page)
        stmt = stmt.where(OCRBlock.page_id == page_row.id)
    blocks = db.scalars(stmt).all()
    return [
        {
            "id": b.id,
            "page_id": b.page_id,
            "block_id": b.block_id,
            "line_id": b.line_id,
            "text": b.corrected_text or b.text,
            "original_text": b.text,
            "confidence": b.confidence,
            "bbox": b.bbox,
            "polygon": b.polygon,
            "language": b.language,
            "is_reviewed": b.is_reviewed,
        }
        for b in blocks
    ]


@router.get("/{document_id}/layout", response_model=list[LayoutBlockRead])
def get_layout_blocks(document_id: str, page: Optional[int] = None, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    stmt = select(LayoutBlock).where(LayoutBlock.document_id == document_id)
    if page is not None:
        page_row = _get_page_or_404(db, document_id, page)
        stmt = stmt.where(LayoutBlock.page_id == page_row.id)
    blocks = db.scalars(stmt.order_by(LayoutBlock.z_order)).all()
    return [LayoutBlockRead.model_validate(b) for b in blocks]


@router.get("/{document_id}/tables", response_model=list[TableRead])
def get_tables(document_id: str, page: Optional[int] = None, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    stmt = select(Table).where(Table.document_id == document_id)
    if page is not None:
        page_row = _get_page_or_404(db, document_id, page)
        stmt = stmt.where(Table.page_id == page_row.id)
    tables = db.scalars(stmt).all()
    return [TableRead.model_validate(t) for t in tables]


@router.put("/{document_id}/tables/{table_id}", response_model=TableRead)
def update_table(document_id: str, table_id: str, body: TableUpdate, db: Session = Depends(get_db)):
    """Structural table edits (spec section 22): add/delete row/column,
    merge/split cells."""
    _get_document_or_404(db, document_id)
    table = document_service.update_table_structure(db, table_id, body.op, body.row, body.column, body.cell_ids)
    if table is None:
        raise HTTPException(status_code=404, detail="Table not found")
    return TableRead.model_validate(table)


@router.put("/{document_id}/blocks/{block_id}", response_model=LayoutBlockRead)
def update_block(document_id: str, block_id: str, body: LayoutBlockUpdate, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    block = document_service.update_layout_block(db, block_id, body.text, body.style)
    if block is None:
        raise HTTPException(status_code=404, detail="Layout block not found")
    return LayoutBlockRead.model_validate(block)


@router.put("/{document_id}/cells/{cell_id}", response_model=TableCellRead)
def update_cell(document_id: str, cell_id: str, body: TableCellUpdate, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    align_h = TextAlign(body.align_h) if body.align_h else None
    align_v = VerticalAlign(body.align_v) if body.align_v else None
    cell = document_service.update_table_cell(db, cell_id, body.text, align_h, align_v, body.rowspan, body.colspan)
    if cell is None:
        raise HTTPException(status_code=404, detail="Table cell not found")
    return TableCellRead.model_validate(cell)


def _dispatch_export(document_id: str, db: Session, target: str) -> dict:
    """Queues a regenerate task and returns immediately -- never blocks
    the HTTP request on the actual PDF/DOCX/Excel work. This replaced a
    synchronous implementation after a confirmed real problem: a 145-page
    document's regenerate request outlived a 300s client timeout while
    still completing correctly server-side, which for a large document
    library (any single document over roughly 100-150 pages on a
    resource-constrained host) makes every regenerate action from the UI
    look broken even though nothing actually failed."""
    from app.workers.export_tasks import regenerate_document_export

    _get_document_or_404(db, document_id)  # 404 before queuing a task for nothing
    async_result = regenerate_document_export.delay(document_id, target)
    return {"status": "queued", "task_id": async_result.id, "target": target}


@router.post("/{document_id}/reconstruct")
def reconstruct(document_id: str, db: Session = Depends(get_db)):
    """Queues a re-render of the reconstructed PDF from the current
    (possibly user-edited) Document JSON, without re-running OCR/layout/
    table detection. Poll `GET .../export/status/{task_id}` for
    completion."""
    return _dispatch_export(document_id, db, "reconstructed_pdf")


@router.post("/{document_id}/export/pdf")
def export_pdf(document_id: str, db: Session = Depends(get_db)):
    return reconstruct(document_id, db)


@router.post("/{document_id}/export/docx")
def export_docx(document_id: str, db: Session = Depends(get_db)):
    return _dispatch_export(document_id, db, "docx")


@router.post("/{document_id}/export/pdf/searchable")
def export_searchable_pdf(document_id: str, db: Session = Depends(get_db)):
    """Queues a regeneration of clean.pdf + searchable.pdf (the image-based
    exports, including the primary "cleaned searchable PDF" download) from
    the currently-stored processed page images + OCR words, without a full
    reprocess -- see `app.workers.export_tasks.regenerate_document_export`
    and `document_service.rebuild_image_based_exports`. Poll
    `GET .../export/status/{task_id}` for completion."""
    return _dispatch_export(document_id, db, "searchable_pdf")


@router.post("/{document_id}/export/excel")
def export_excel(document_id: str, db: Session = Depends(get_db)):
    """Structured-data export (Pipeline B) -- one sheet per detected table,
    read from the same Document JSON PDF/DOCX read, never re-derived from
    either of them (spec section 52: Document JSON is the one source of
    truth all three exports branch from). Poll
    `GET .../export/status/{task_id}` for completion."""
    return _dispatch_export(document_id, db, "excel")


@router.get("/{document_id}/export/status/{task_id}")
def export_status(document_id: str, task_id: str, db: Session = Depends(get_db)):
    """Polls an async regenerate task started by one of the export/
    reconstruct endpoints above. `state` is one of Celery's own task
    states (PENDING while queued or running -- Celery does not distinguish
    the two without extra instrumentation this lightweight task doesn't
    need -- SUCCESS, or FAILURE); `result` carries the target's output
    dict (e.g. `size_bytes`) once state is SUCCESS."""
    _get_document_or_404(db, document_id)
    async_result = celery_app.AsyncResult(task_id)
    payload = {"task_id": task_id, "state": async_result.state}
    if async_result.state == "SUCCESS":
        payload["result"] = async_result.result
    elif async_result.state == "FAILURE":
        payload["error"] = str(async_result.result)
    return payload


@router.get("/{document_id}/download/pdf")
def download_pdf(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    """The primary "cleaned searchable PDF" download (spec sections 1/14/
    16/29/38): the cleaned page image with an invisible OCR text layer,
    NOT the from-scratch font/textbox reconstruction -- visual fidelity to
    the original scanned book is the explicit priority for this download,
    with search/copy support layered invisibly on top rather than
    replacing the page's appearance. The fully reconstructed, editable
    (real vector text/tables) PDF is still generated every run and stays
    available at /download/pdf/reconstructed for the editing workflow."""
    document = _get_document_or_404(db, document_id)
    filename = f"{_export_basename(document.original_filename)}_cleaned_searchable.pdf"
    return _download_export(db, storage, document_id, ExportType.SEARCHABLE_PDF, "application/pdf", filename)


@router.get("/{document_id}/download/pdf/reconstructed")
def download_reconstructed_pdf(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    document = _get_document_or_404(db, document_id)
    filename = f"{_export_basename(document.original_filename)}_reconstructed.pdf"
    return _download_export(db, storage, document_id, ExportType.RECONSTRUCTED_PDF, "application/pdf", filename)


@router.get("/{document_id}/download/docx")
def download_docx(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    document = _get_document_or_404(db, document_id)
    filename = f"{_export_basename(document.original_filename)}.docx"
    return _download_export(
        db,
        storage,
        document_id,
        ExportType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename,
    )


@router.get("/{document_id}/download/excel")
def download_excel(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    document = _get_document_or_404(db, document_id)
    filename = f"{_export_basename(document.original_filename)}_extracted_data.xlsx"
    return _download_export(
        db, storage, document_id, ExportType.XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename,
    )


@router.get("/{document_id}/quality", response_model=QualityReport)
def get_quality(document_id: str, db: Session = Depends(get_db)):
    _get_document_or_404(db, document_id)
    pages = db.scalars(select(DocumentPage).where(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number)).all()
    reports = [
        QualityPageReport(
            page_number=p.page_number,
            ocr_confidence_avg=p.ocr_confidence_avg,
            layout_confidence_avg=p.layout_confidence_avg,
            table_confidence_avg=p.table_confidence_avg,
            visual_similarity_score=p.visual_similarity_score,
            needs_review=needs_review(p.ocr_confidence_avg, p.visual_similarity_score),
        )
        for p in pages
    ]
    scores = [r.visual_similarity_score for r in reports if r.visual_similarity_score is not None]
    overall = sum(scores) / len(scores) if scores else None
    return QualityReport(document_id=document_id, pages=reports, overall_visual_similarity=overall)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _get_document_or_404(db: Session, document_id: str) -> Document:
    document = document_service.get_document(db, document_id)
    if document is None or document.is_deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _get_page_or_404(db: Session, document_id: str, page_number: int) -> DocumentPage:
    _get_document_or_404(db, document_id)
    page = db.scalar(
        select(DocumentPage).where(DocumentPage.document_id == document_id, DocumentPage.page_number == page_number)
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found")
    return page


def _page_to_read(page: DocumentPage, storage: StorageBackend) -> PageRead:
    return PageRead(
        id=page.id,
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        dpi=page.dpi,
        rotation=page.rotation,
        original_image_url=storage.url_for(page.original_image_path),
        processed_image_url=storage.url_for(page.processed_image_path) if page.processed_image_path else None,
        ocr_status=page.ocr_status,
        layout_status=page.layout_status,
        table_status=page.table_status,
        ocr_confidence_avg=page.ocr_confidence_avg,
        layout_confidence_avg=page.layout_confidence_avg,
        table_confidence_avg=page.table_confidence_avg,
        visual_similarity_score=page.visual_similarity_score,
    )


def _export_basename(original_filename: str) -> str:
    """`book.pdf` -> `book` (spec section 30: exports are named
    `{original}_cleaned_searchable.pdf` / `{original}_extracted_data.xlsx`,
    never the generic `reconstructed.pdf`/`document.docx` internal storage
    names, which stay as-is since they're never user-facing)."""
    from pathlib import Path

    stem = Path(original_filename or "document").stem
    return stem or "document"


def _download_export(db: Session, storage: StorageBackend, document_id: str, export_type: ExportType, media_type: str, filename: str):
    document = _get_document_or_404(db, document_id)
    export = db.scalar(
        select(ExportFile).where(ExportFile.document_id == document.id, ExportFile.export_type == export_type)
    )
    if export is None or not storage.exists(export.storage_path):
        raise HTTPException(status_code=404, detail=f"{export_type.value} has not been generated yet")
    data = storage.read(export.storage_path)
    return Response(
        content=data, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
