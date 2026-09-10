"""Documents API (spec section 31)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_storage_backend
from app.core.celery_app import celery_app
from app.core.config import get_settings
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
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    user: Optional[str] = Depends(get_current_user),
):
    settings = get_settings()
    content = await file.read()

    resolved_dpi = dpi or settings.default_dpi
    if resolved_dpi not in settings.allowed_dpi_list:
        raise HTTPException(status_code=422, detail=f"dpi must be one of {settings.allowed_dpi_list}")

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

    celery_app.send_task("pipeline.process_document", args=[document.id, job.id])

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
    celery_app.send_task("pipeline.process_document", args=[document.id, job.id])
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


@router.post("/{document_id}/reconstruct")
def reconstruct(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    """Re-renders the reconstructed PDF from the current (possibly
    user-edited) Document JSON, without re-running OCR/layout/table
    detection."""
    from app.reconstruction.pdf_renderer import render_document_pdf

    document = _get_document_or_404(db, document_id)
    doc_json = document_service.get_document_json(db, document)
    if not doc_json.pages:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")

    pdf_bytes = render_document_pdf(doc_json.pages)
    relative_path = f"output/{document.id}/reconstructed.pdf"
    storage.write(relative_path, pdf_bytes)
    document_service.record_export_file(db, document, ExportType.RECONSTRUCTED_PDF, relative_path, len(pdf_bytes))
    return {"status": "ok", "size_bytes": len(pdf_bytes)}


@router.post("/{document_id}/export/pdf")
def export_pdf(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    return reconstruct(document_id, db, storage)


@router.post("/{document_id}/export/docx")
def export_docx(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    from app.exporters.docx_exporter import render_document_docx

    document = _get_document_or_404(db, document_id)
    doc_json = document_service.get_document_json(db, document)
    if not doc_json.pages:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")

    docx_bytes = render_document_docx(doc_json.pages)
    relative_path = f"output/{document.id}/document.docx"
    storage.write(relative_path, docx_bytes)
    document_service.record_export_file(db, document, ExportType.DOCX, relative_path, len(docx_bytes))
    return {"status": "ok", "size_bytes": len(docx_bytes)}


@router.get("/{document_id}/download/pdf")
def download_pdf(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    return _download_export(db, storage, document_id, ExportType.RECONSTRUCTED_PDF, "application/pdf", "reconstructed.pdf")


@router.get("/{document_id}/download/docx")
def download_docx(document_id: str, db: Session = Depends(get_db), storage: StorageBackend = Depends(get_storage_backend)):
    return _download_export(
        db,
        storage,
        document_id,
        ExportType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "document.docx",
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
