"""
Document/page/job CRUD + pipeline-result persistence. This is the one place
that translates between the in-memory pipeline outputs (OCRWordResult,
LayoutBlockResult, DetectedTable, PageJSON) and the database — API routers
and Celery tasks both go through here rather than touching models directly.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.db.base import new_uuid
from app.layout.detector import LayoutBlockResult
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.enums import DocumentStatus, ExportType, LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage
from app.models.export_file import ExportFile
from app.models.layout_block import LayoutBlock
from app.models.ocr_block import OCRBlock
from app.models.processing_job import ProcessingJob
from app.models.table import Table
from app.models.table_cell import TableCell
from app.reconstruction.document_json_builder import build_page_json
from app.schemas.document import ProgressEvent
from app.schemas.document_json import DocumentJSON, PageJSON
from app.schemas.ocr import OCRWordResult
from app.services.progress import publish_progress_sync
from app.tables.models import DetectedTable
from app.utils.file_safety import sanitize_filename, validate_upload

PAGE_SIZE_DEFAULT = 20


def compute_document_hash(content: bytes) -> str:
    """sha256 of the raw uploaded bytes -- duplicate-upload detection
    (spec section 9). A pure function of the file content only (not
    filename/timestamp), so the same PDF re-uploaded under a different
    name is still recognized as a duplicate."""
    return hashlib.sha256(content).hexdigest()


def find_document_by_hash(db: Session, document_hash: str) -> Document | None:
    """Never matches a soft-deleted document -- re-uploading something the
    user explicitly removed is not a duplicate."""
    return db.scalar(select(Document).where(Document.document_hash == document_hash, Document.is_deleted.is_(False)))


# ----------------------------------------------------------------------------
# Document CRUD
# ----------------------------------------------------------------------------

def create_document_from_upload(
    db: Session,
    storage,
    filename: str,
    content: bytes,
    ocr_provider: OCRProviderEnum,
    dpi: int,
    preprocess_profile: PreprocessProfileEnum,
    uploaded_by: str | None = None,
) -> tuple[Document, ProcessingJob]:
    validated = validate_upload(filename, content)
    safe_name = sanitize_filename(filename)

    document = Document(
        original_filename=safe_name,
        file_extension=validated.extension,
        mime_type=validated.mime_type,
        file_size_bytes=validated.size_bytes,
        page_count=validated.page_count,
        status=DocumentStatus.UPLOADED,
        ocr_provider=ocr_provider,
        dpi=dpi,
        preprocess_profile=preprocess_profile,
        storage_original_path="",
        uploaded_by=uploaded_by,
        document_hash=compute_document_hash(content),
    )
    db.add(document)
    db.flush()  # assigns document.id

    relative_path = f"original/{document.id}/{safe_name}"
    storage.write(relative_path, content)
    document.storage_original_path = relative_path

    job = ProcessingJob(
        document_id=document.id,
        status=DocumentStatus.QUEUED,
        stage=ProcessingStage.UPLOAD,
        progress_percent=0,
        ocr_provider=ocr_provider,
        dpi=dpi,
        preprocess_profile=preprocess_profile,
    )
    db.add(job)
    document.status = DocumentStatus.QUEUED
    db.commit()
    db.refresh(document)
    db.refresh(job)
    return document, job


def get_document(db: Session, document_id: str) -> Document | None:
    return db.get(Document, document_id)


def list_documents(
    db: Session, status: DocumentStatus | None = None, page: int = 1, page_size: int = PAGE_SIZE_DEFAULT
) -> tuple[list[Document], int]:
    stmt = select(Document).where(Document.is_deleted.is_(False))
    if status:
        stmt = stmt.where(Document.status == status)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    stmt = stmt.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    items = list(db.scalars(stmt).all())
    return items, total


def get_stats(db: Session) -> dict:
    base = select(Document).where(Document.is_deleted.is_(False))
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    processing = db.scalar(
        select(func.count()).select_from(
            base.where(
                Document.status.in_(
                    [
                        DocumentStatus.QUEUED,
                        DocumentStatus.PROCESSING,
                        DocumentStatus.OCR_PROCESSING,
                        DocumentStatus.LAYOUT_PROCESSING,
                        DocumentStatus.TABLE_PROCESSING,
                        DocumentStatus.RECONSTRUCTING,
                        DocumentStatus.EXPORTING,
                    ]
                )
            ).subquery()
        )
    ) or 0
    completed = db.scalar(select(func.count()).select_from(base.where(Document.status == DocumentStatus.COMPLETED).subquery())) or 0
    failed = db.scalar(select(func.count()).select_from(base.where(Document.status == DocumentStatus.FAILED).subquery())) or 0
    return {"total": total, "processing": processing, "completed": completed, "failed": failed}


def soft_delete_document(db: Session, storage, document: Document) -> None:
    document.is_deleted = True
    db.commit()
    storage.delete_prefix(f"original/{document.id}")
    storage.delete_prefix(f"pages/{document.id}")
    storage.delete_prefix(f"processed/{document.id}")
    storage.delete_prefix(f"output/{document.id}")


# ----------------------------------------------------------------------------
# Processing jobs / progress
# ----------------------------------------------------------------------------

def create_processing_job(
    db: Session, document: Document, ocr_provider: OCRProviderEnum, dpi: int, preprocess_profile: PreprocessProfileEnum
) -> ProcessingJob:
    job = ProcessingJob(
        document_id=document.id,
        status=DocumentStatus.QUEUED,
        stage=ProcessingStage.UPLOAD,
        progress_percent=0,
        ocr_provider=ocr_provider,
        dpi=dpi,
        preprocess_profile=preprocess_profile,
    )
    db.add(job)
    document.status = DocumentStatus.QUEUED
    document.ocr_provider = ocr_provider
    document.dpi = dpi
    document.preprocess_profile = preprocess_profile
    # A previous run's error is not this run's error -- without clearing
    # it here, a document that failed once and was later successfully
    # reprocessed keeps showing its old failure message forever (nothing
    # else ever clears `error_message`; `update_job_progress` only ever
    # sets it, on the FAILED/CANCELLED path). Confirmed as a real bug
    # against live data: a document sitting at status=COMPLETED still
    # displayed "Processing stalled: no update for over 15 minutes..."
    # from a run that failed hours earlier.
    document.error_message = None
    db.commit()
    db.refresh(job)
    return job


def update_job_progress(
    db: Session,
    job: ProcessingJob,
    stage: ProcessingStage,
    percent: int,
    status: DocumentStatus,
    page: int | None = None,
    message: str = "",
    error: str | None = None,
) -> None:
    job.stage = stage
    job.progress_percent = percent
    job.status = status
    if error:
        job.error_message = error
    elif status == DocumentStatus.COMPLETED:
        # Defensive clear: the main guard against a stale error message is
        # `create_processing_job` resetting it at the start of a run, but
        # clearing it again here too means a document that reaches
        # COMPLETED is never showing an error, regardless of the path
        # that got it there.
        job.error_message = None
    if status == DocumentStatus.PROCESSING and job.started_at is None:
        job.started_at = datetime.now(timezone.utc)
    if status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED, DocumentStatus.CANCELLED):
        job.finished_at = datetime.now(timezone.utc)

    document = db.get(Document, job.document_id)
    if document:
        document.status = status
        if error:
            document.error_message = error
        elif status == DocumentStatus.COMPLETED:
            document.error_message = None
    db.commit()

    publish_progress_sync(
        ProgressEvent(
            document_id=job.document_id, job_id=job.id, stage=stage, percent=percent, page=page, message=message, status=status
        )
    )


# A job is only ever "actively running" once it has left QUEUED -- a QUEUED
# job with no lock yet is completely normal (Celery just hasn't picked it up
# or the task hasn't reached its lock-acquisition line yet).
_STARTED_ACTIVE_STATUSES = {
    DocumentStatus.PROCESSING,
    DocumentStatus.OCR_PROCESSING,
    DocumentStatus.LAYOUT_PROCESSING,
    DocumentStatus.TABLE_PROCESSING,
    DocumentStatus.RECONSTRUCTING,
    DocumentStatus.EXPORTING,
}
# The pipeline's document lock (TTL 600s, renewed after every page -- see
# `app.workers.pipeline_tasks`) is the ground truth for "is a worker still
# actually alive on this document". Its own self-heal window is 600s; this
# is deliberately looser so a job is never flagged mid-legitimate-renewal-
# gap, only once it's unambiguously past even a slow page plus the lock's
# own TTL.
_STALE_JOB_GRACE_SECONDS = 900


def detect_and_fail_stale_job(db: Session, job: ProcessingJob) -> ProcessingJob:
    """Called from the progress-polling read path (not a background sweep --
    this project has no Celery beat scheduler running, so a lazy check on
    read is the simplest reliable way to surface this). If a job has been in
    an active, already-started status well past the document lock's TTL and
    that lock no longer exists, the worker that held it is gone and nothing
    will ever advance this job again -- Celery's own broker-level redelivery
    for a `task_acks_late` task is not guaranteed to happen promptly (default
    Redis transport visibility_timeout is 1 hour), so without this a stuck
    job leaves the UI showing "processing" indefinitely with no way for the
    user to know a retry is needed. Confirmed against a real incident: a
    7-page document's worker was killed mid-task by a container restart and
    sat reporting OCR_PROCESSING for 57+ minutes with no active Celery task
    anywhere and no Redis lock -- silently unrecoverable without this check.
    """
    if job.status not in _STARTED_ACTIVE_STATUSES:
        return job
    if job.updated_at is None:
        return job
    now = datetime.now(timezone.utc)
    updated_at = job.updated_at if job.updated_at.tzinfo else job.updated_at.replace(tzinfo=timezone.utc)
    age_seconds = (now - updated_at).total_seconds()
    if age_seconds < _STALE_JOB_GRACE_SECONDS:
        return job

    import redis as redis_sync

    from app.core.config import get_settings
    from app.core.locks import DOCUMENT_LOCK_PREFIX

    settings = get_settings()
    try:
        client = redis_sync.Redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5)
        try:
            lock_exists = client.exists(DOCUMENT_LOCK_PREFIX + job.document_id)
        finally:
            client.close()
    except redis_sync.exceptions.RedisError:
        # Can't tell either way right now -- don't fail a possibly-healthy
        # job just because this particular check couldn't reach Redis.
        return job

    if lock_exists:
        return job

    update_job_progress(
        db,
        job,
        job.stage,
        job.progress_percent,
        DocumentStatus.FAILED,
        error=(
            f"Processing stalled: no update for over {int(age_seconds // 60)} minutes and the worker's "
            "processing lock is gone (the worker likely died or was restarted mid-task). Re-run to try again."
        ),
    )
    return job


# ----------------------------------------------------------------------------
# Page + pipeline-result persistence
# ----------------------------------------------------------------------------

def upsert_page(
    db: Session,
    document: Document,
    page_number: int,
    width: int,
    height: int,
    dpi: int,
    rotation: float,
    original_image_path: str,
) -> DocumentPage:
    page = db.scalar(
        select(DocumentPage).where(DocumentPage.document_id == document.id, DocumentPage.page_number == page_number)
    )
    if page is None:
        page = DocumentPage(document_id=document.id, page_number=page_number)
        db.add(page)
    page.width = width
    page.height = height
    page.dpi = dpi
    page.rotation = rotation
    page.original_image_path = original_image_path
    db.commit()
    db.refresh(page)
    return page


def persist_page_pipeline_result(
    db: Session,
    document: Document,
    page: DocumentPage,
    processed_image_path: str,
    words: list[OCRWordResult],
    layout_results: list[LayoutBlockResult],
    tables: list[DetectedTable],
) -> PageJSON:
    """Writes OCRBlock/LayoutBlock/Table/TableCell rows for one page and
    builds+stores its Document JSON. Replaces any prior rows for the page
    (idempotent re-processing)."""
    db.query(OCRBlock).filter(OCRBlock.page_id == page.id).delete()
    db.query(TableCell).filter(TableCell.table_id.in_(select(Table.id).where(Table.page_id == page.id))).delete(
        synchronize_session=False
    )
    db.query(Table).filter(Table.page_id == page.id).delete()
    db.query(LayoutBlock).filter(LayoutBlock.page_id == page.id).delete()
    db.flush()

    page.processed_image_path = processed_image_path

    for w in words:
        db.add(
            OCRBlock(
                document_id=document.id,
                page_id=page.id,
                block_id=w.block_id,
                line_id=w.line_id,
                word_id=w.word_id,
                text=w.text,
                confidence=w.confidence,
                bbox=w.bbox.model_dump(),
                polygon=w.polygon.to_xy_list(),
                language=w.language,
                font_size_estimate=w.font_size_estimate,
            )
        )
    page.ocr_confidence_avg = float(sum(w.confidence for w in words) / len(words)) if words else None
    page.ocr_status = "completed"

    table_rows: list[Table] = []
    for table in tables:
        table_row = Table(
            document_id=document.id,
            page_id=page.id,
            bbox=table.bbox.model_dump(),
            n_rows=table.n_rows,
            n_cols=table.n_cols,
            confidence=table.confidence,
            detection_method=table.detection_method,
            column_widths=table.column_widths,
            row_heights=table.row_heights,
            border_style=table.border_style,
        )
        db.add(table_row)
        db.flush()
        for cell in table.cells:
            db.add(
                TableCell(
                    table_id=table_row.id,
                    row=cell.row,
                    column=cell.column,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    text=cell.text,
                    confidence=cell.confidence,
                    bbox=cell.bbox.model_dump(),
                    align_h=cell.align_h,
                    align_v=cell.align_v,
                    is_header=cell.is_header,
                )
            )
        table_rows.append(table_row)
    page.table_status = "completed"
    table_confidences = [t.confidence for t in tables]
    page.table_confidence_avg = float(sum(table_confidences) / len(table_confidences)) if table_confidences else None

    layout_confidences = []
    for result in layout_results:
        layout_confidences.append(result.confidence)
        # Assign the id up front (rather than relying on the column default,
        # which only materializes on flush) so it's known immediately below
        # and can be threaded into the Document JSON block + the Table's
        # back-reference — both need to agree with the DB row's id for
        # structural table edits to resync the right JSON block later.
        layout_block = LayoutBlock(
            id=new_uuid(),
            document_id=document.id,
            page_id=page.id,
            block_type=result.block_type,
            bbox=result.bbox.model_dump(),
            confidence=result.confidence,
            z_order=result.z_order,
            content={"word_ids": result.word_ids, "text": result.text},
            style=result.style,
        )
        db.add(layout_block)
        result.db_id = layout_block.id
        if result.block_type == LayoutBlockType.TABLE and result.table_ref is not None:
            table_rows[result.table_ref].layout_block_id = layout_block.id
    page.layout_status = "completed"
    page.layout_confidence_avg = float(sum(layout_confidences) / len(layout_confidences)) if layout_confidences else None

    page_json = build_page_json(
        document_id=document.id,
        page_id=page.id,
        page_number=page.page_number,
        page_width=page.width,
        page_height=page.height,
        dpi=page.dpi,
        rotation=page.rotation,
        layout_results=layout_results,
        tables=tables,
    )
    page.document_json = page_json.model_dump(mode="json")
    db.commit()
    db.refresh(page)
    return page_json


def get_document_json(db: Session, document: Document) -> DocumentJSON:
    pages = db.scalars(
        select(DocumentPage).where(DocumentPage.document_id == document.id).order_by(DocumentPage.page_number)
    ).all()
    page_jsons = [PageJSON(**p.document_json) for p in pages if p.document_json]
    return DocumentJSON(document_id=document.id, page_count=len(page_jsons), pages=page_jsons)


def update_layout_block(db: Session, block_id: str, text: str | None, style: dict | None) -> LayoutBlock | None:
    block = db.get(LayoutBlock, block_id)
    if block is None:
        return None
    if text is not None:
        block.is_edited = True
        block.edited_text = text
        content = dict(block.content or {})
        content["text"] = text
        block.content = content
    if style is not None:
        block.style = {**(block.style or {}), **style}
    db.commit()
    db.refresh(block)
    _touch_page_json_block_text(db, block)
    return block


def update_table_cell(
    db: Session, cell_id: str, text: str | None, align_h=None, align_v=None, rowspan: int | None = None, colspan: int | None = None
) -> TableCell | None:
    cell = db.get(TableCell, cell_id)
    if cell is None:
        return None
    if text is not None:
        cell.is_edited = True
        cell.edited_text = text
        cell.text = text
    if align_h is not None:
        cell.align_h = align_h
    if align_v is not None:
        cell.align_v = align_v
    if rowspan is not None:
        cell.rowspan = rowspan
    if colspan is not None:
        cell.colspan = colspan
    db.commit()
    db.refresh(cell)
    _touch_page_json_table_cell(db, cell)
    return cell


def update_table_structure(
    db: Session, table_id: str, op: str, row: int | None, column: int | None, cell_ids: list[str] | None
) -> Table | None:
    """Structural table edits (spec section 22): add/delete row or column,
    merge/split cells. Operates on the TableCell rows directly, then
    resyncs that table's block inside DocumentPage.document_json so
    reconstruction (PDF/DOCX) reflects the edit without a full pipeline
    re-run."""
    table = db.get(Table, table_id)
    if table is None:
        return None

    if op == "add_row":
        _add_row(db, table, row if row is not None else table.n_rows)
    elif op == "delete_row":
        if row is not None:
            _delete_row(db, table, row)
    elif op == "add_column":
        _add_column(db, table, column if column is not None else table.n_cols)
    elif op == "delete_column":
        if column is not None:
            _delete_column(db, table, column)
    elif op == "merge_cells":
        if cell_ids:
            _merge_cells(db, table, cell_ids)
    elif op == "split_cell":
        if cell_ids:
            _split_cell(db, table, cell_ids[0])

    db.commit()
    db.refresh(table)
    _sync_page_json_table(db, table)
    return table


def _add_row(db: Session, table: Table, at: int) -> None:
    for cell in table.cells:
        if cell.row >= at:
            cell.row += 1
    for c in range(table.n_cols):
        db.add(TableCell(table_id=table.id, row=at, column=c, bbox={"x1": 0, "y1": 0, "x2": 0, "y2": 0}))
    table.n_rows += 1
    table.row_heights = [*table.row_heights[:at], 30.0, *table.row_heights[at:]]


def _delete_row(db: Session, table: Table, at: int) -> None:
    if table.n_rows <= 1:
        return
    for cell in list(table.cells):
        if cell.row == at:
            db.delete(cell)
        elif cell.row > at:
            cell.row -= 1
    table.n_rows -= 1
    if 0 <= at < len(table.row_heights):
        table.row_heights = [*table.row_heights[:at], *table.row_heights[at + 1 :]]


def _add_column(db: Session, table: Table, at: int) -> None:
    for cell in table.cells:
        if cell.column >= at:
            cell.column += 1
    for r in range(table.n_rows):
        db.add(TableCell(table_id=table.id, row=r, column=at, bbox={"x1": 0, "y1": 0, "x2": 0, "y2": 0}))
    table.n_cols += 1
    table.column_widths = [*table.column_widths[:at], 80.0, *table.column_widths[at:]]


def _delete_column(db: Session, table: Table, at: int) -> None:
    if table.n_cols <= 1:
        return
    for cell in list(table.cells):
        if cell.column == at:
            db.delete(cell)
        elif cell.column > at:
            cell.column -= 1
    table.n_cols -= 1
    if 0 <= at < len(table.column_widths):
        table.column_widths = [*table.column_widths[:at], *table.column_widths[at + 1 :]]


def _merge_cells(db: Session, table: Table, cell_ids: list[str]) -> None:
    cells = [c for c in table.cells if c.id in cell_ids]
    if len(cells) < 2:
        return
    min_row = min(c.row for c in cells)
    max_row = max(c.row + c.rowspan - 1 for c in cells)
    min_col = min(c.column for c in cells)
    max_col = max(c.column + c.colspan - 1 for c in cells)

    anchor = min(cells, key=lambda c: (c.row, c.column))
    anchor.row, anchor.column = min_row, min_col
    anchor.rowspan = max_row - min_row + 1
    anchor.colspan = max_col - min_col + 1
    anchor.text = " ".join(c.text for c in sorted(cells, key=lambda c: (c.row, c.column)) if c.text).strip()
    anchor.is_edited = True

    for c in cells:
        if c.id != anchor.id:
            db.delete(c)


def _split_cell(db: Session, table: Table, cell_id: str) -> None:
    """Resets a merged cell back to a single 1x1 cell at its origin. This is
    a simplified "split" — it does not regenerate the individual sibling
    cells the merge previously consumed (their original text is not
    recoverable), consistent with never fabricating source content."""
    cell = db.get(TableCell, cell_id)
    if cell is None or (cell.rowspan == 1 and cell.colspan == 1):
        return
    cell.rowspan = 1
    cell.colspan = 1
    cell.is_edited = True


def _sync_page_json_table(db: Session, table: Table) -> None:
    page = db.get(DocumentPage, table.page_id)
    if not page or not page.document_json:
        return
    rows: dict[int, list[dict]] = {}
    for cell in sorted(table.cells, key=lambda c: (c.row, c.column)):
        rows.setdefault(cell.row, []).append(
            {
                "row": cell.row,
                "column": cell.column,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "text": cell.text,
                "bbox": cell.bbox,
                "align_h": cell.align_h.value,
                "align_v": cell.align_v.value,
                "confidence": cell.confidence,
                "is_header": cell.is_header,
                "is_edited": cell.is_edited,
            }
        )
    table_json = {
        "rows": [{"cells": rows[r]} for r in sorted(rows.keys())],
        "column_widths": table.column_widths,
        "row_heights": table.row_heights,
        "border_style": table.border_style,
        "detection_method": table.detection_method.value,
    }

    data = page.document_json
    for b in data.get("blocks", []):
        if b.get("type") == "table" and b.get("id") and table.layout_block_id and b["id"] == table.layout_block_id:
            b["table"] = table_json
            break
    else:
        # Fall back to matching by bbox proximity if no direct layout_block_id link.
        for b in data.get("blocks", []):
            if b.get("type") == "table" and b.get("bbox") == table.bbox:
                b["table"] = table_json
                break
    page.document_json = data
    flag_modified(page, "document_json")  # in-place dict mutation needs an explicit flag (plain JSON column)
    db.commit()


def _touch_page_json_block_text(db: Session, block: LayoutBlock) -> None:
    """Keeps DocumentPage.document_json (the IR reconstruction reads from)
    in sync with a direct block edit, without a full pipeline re-run."""
    page = db.get(DocumentPage, block.page_id)
    if not page or not page.document_json:
        return
    data = page.document_json
    for b in data.get("blocks", []):
        if b.get("id") == block.id or (b.get("type") == block.block_type.value and b.get("bbox") == block.bbox):
            if b.get("content"):
                b["content"] = [{**b["content"][0], "text": block.edited_text}] if b["content"] else b["content"]
            b["is_edited"] = True
            break
    page.document_json = data
    flag_modified(page, "document_json")  # in-place dict mutation needs an explicit flag (plain JSON column)
    db.commit()


def _touch_page_json_table_cell(db: Session, cell: TableCell) -> None:
    table = db.get(Table, cell.table_id)
    if not table:
        return
    page = db.get(DocumentPage, table.page_id)
    if not page or not page.document_json:
        return
    data = page.document_json
    for b in data.get("blocks", []):
        if b.get("type") != "table" or not b.get("table"):
            continue
        for row in b["table"].get("rows", []):
            for c in row.get("cells", []):
                if c.get("row") == cell.row and c.get("column") == cell.column:
                    c["text"] = cell.text
                    c["is_edited"] = True
    page.document_json = data
    flag_modified(page, "document_json")  # in-place dict mutation needs an explicit flag (plain JSON column)
    db.commit()


def _ocr_block_to_word(b: OCRBlock) -> OCRWordResult:
    from app.schemas.geometry import BBox, Polygon

    bbox = b.bbox or {}
    polygon_pts = b.polygon or [[bbox.get("x1", 0), bbox.get("y1", 0)], [bbox.get("x2", 0), bbox.get("y1", 0)],
                                 [bbox.get("x2", 0), bbox.get("y2", 0)], [bbox.get("x1", 0), bbox.get("y2", 0)]]
    return OCRWordResult(
        text=b.corrected_text or b.text,
        confidence=b.confidence,
        bbox=BBox(**bbox),
        polygon=Polygon.from_xy_list(polygon_pts),
        page_number=0,  # not needed by callers here; per-page context is already known
        block_id=b.block_id,
        line_id=b.line_id,
        language=b.language or "und",
    )


def get_page_ocr_words(db: Session, page_id: str) -> list[OCRWordResult]:
    """Word-level OCR results for one page, read back from `OCRBlock` rows.
    Used to rebuild image-based exports (clean/searchable PDF) on demand --
    those need per-word boxes, unlike the Document JSON's `PageJSON.blocks`
    content, which is already grouped into layout-block-level text runs.

    For a whole document (many pages), prefer
    `get_ocr_words_by_page_for_document` -- one query for every page
    instead of one query per page."""
    blocks = db.scalars(
        select(OCRBlock).where(OCRBlock.page_id == page_id).order_by(OCRBlock.reading_order_index)
    ).all()
    return [_ocr_block_to_word(b) for b in blocks]


def get_ocr_words_by_page_for_document(db: Session, document_id: str) -> dict[str, list[OCRWordResult]]:
    """Same word-level data as `get_page_ocr_words`, but for every page of
    a document in a single query -- avoids N round-trips (one per page)
    when rebuilding a whole document's image-based exports, which for a
    100+ page document is otherwise a real, measurable contributor to how
    long a "regenerate" request takes."""
    rows = db.scalars(
        select(OCRBlock)
        .where(OCRBlock.document_id == document_id)
        .order_by(OCRBlock.page_id, OCRBlock.reading_order_index)
    ).all()
    by_page: dict[str, list[OCRWordResult]] = {}
    for b in rows:
        by_page.setdefault(b.page_id, []).append(_ocr_block_to_word(b))
    return by_page


def rebuild_image_based_exports(db: Session, storage, document: Document) -> dict:
    """Regenerates `clean.pdf` and `searchable.pdf` from the currently-
    stored processed page images + OCR words, without re-running OCR/
    layout/table detection. This is the on-demand equivalent of what the
    full pipeline does incrementally per page -- needed because those two
    exports (unlike reconstructed.pdf/docx/xlsx, which all rebuild from the
    lightweight Document JSON) depend on the actual page raster images, so
    there was previously no way to regenerate them short of a full
    reprocess -- meaning a bug fix or a corrected/re-OCR'd page had no way
    to reach an already-completed document's searchable PDF at all."""
    import fitz

    from app.reconstruction import clean_pdf, searchable_pdf
    from app.reconstruction.fonts import resolve_body_font_path

    pages = db.scalars(
        select(DocumentPage).where(DocumentPage.document_id == document.id).order_by(DocumentPage.page_number)
    ).all()
    if not pages:
        return {"clean_pdf_bytes": 0, "searchable_pdf_bytes": 0, "pages": 0}

    font_path = resolve_body_font_path()
    clean_doc = fitz.open()
    searchable_doc = fitz.open()
    import cv2
    import numpy as np

    # One query for every page's OCR words instead of one query per page --
    # a real, measurable difference for 100+ page documents.
    words_by_page = get_ocr_words_by_page_for_document(db, document.id)

    for page in pages:
        if not page.processed_image_path or not storage.exists(page.processed_image_path):
            continue
        image_bytes = storage.read(page.processed_image_path)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        words = words_by_page.get(page.id, [])
        clean_pdf.add_clean_page(clean_doc, image, page.dpi)
        searchable_pdf.add_searchable_page(searchable_doc, image, page.dpi, words, font_path)

    clean_bytes = clean_doc.tobytes(deflate=True, garbage=4)
    clean_doc.close()
    searchable_bytes = searchable_doc.tobytes(deflate=True, garbage=4)
    searchable_doc.close()

    storage.write(f"output/{document.id}/clean.pdf", clean_bytes)
    storage.write(f"output/{document.id}/searchable.pdf", searchable_bytes)
    record_export_file(db, document, ExportType.CLEAN_PDF, f"output/{document.id}/clean.pdf", len(clean_bytes))
    record_export_file(db, document, ExportType.SEARCHABLE_PDF, f"output/{document.id}/searchable.pdf", len(searchable_bytes))

    return {"clean_pdf_bytes": len(clean_bytes), "searchable_pdf_bytes": len(searchable_bytes), "pages": len(pages)}


def record_export_file(db: Session, document: Document, export_type: ExportType, storage_path: str, size_bytes: int) -> ExportFile:
    existing = db.scalar(
        select(ExportFile).where(ExportFile.document_id == document.id, ExportFile.export_type == export_type)
    )
    if existing:
        existing.storage_path = storage_path
        existing.file_size_bytes = size_bytes
        db.commit()
        db.refresh(existing)
        return existing
    export = ExportFile(document_id=document.id, export_type=export_type, storage_path=storage_path, file_size_bytes=size_bytes)
    db.add(export)
    db.commit()
    db.refresh(export)
    return export
