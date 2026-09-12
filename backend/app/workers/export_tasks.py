"""
Async export/regenerate tasks.

Every export/regenerate action (searchable+clean PDF, DOCX, Excel,
reconstructed PDF) reads already-processed data (Document JSON, stored
page images, OCR blocks) and does deterministic, no-AI, no-OCR work --
but for a large document (100+ pages) that work can still take several
minutes of wall-clock time on a resource-constrained host, and doing it
synchronously inside an HTTP request risks a client-side timeout well
before the server-side work actually finishes (confirmed: a 145-page
regenerate request outlived a 300s client timeout while completing
correctly server-side).

This mirrors the same fix already applied to the main OCR pipeline
(Celery + polled status) rather than inventing a second pattern: dispatch
one lightweight task per regenerate request, return its task_id
immediately, and let the caller poll `GET .../export/status/{task_id}`
(via Celery's own AsyncResult -- no new DB model needed, since unlike the
main pipeline this has no meaningful sub-stages to report and no need to
survive a broker restart).
"""
from __future__ import annotations

from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.document import Document
from app.services import document_service
from app.services.storage import get_storage

logger = get_logger(__name__)

# What `target` values `regenerate_document_export` accepts, and the human
# name used in log lines. Kept as a plain set (not an enum) since this is
# an internal task-dispatch detail, not part of the Document JSON/DB
# schema that other code needs to agree on.
VALID_TARGETS = {"searchable_pdf", "reconstructed_pdf", "docx", "excel"}


@celery_app.task(bind=True, name="export.regenerate_document_export")
def regenerate_document_export(self, document_id: str, target: str) -> dict:
    """Regenerates exactly one export type for an already-processed
    document, without re-running OCR/layout/table detection. Safe to call
    repeatedly (each run fully replaces that export's stored file)."""
    if target not in VALID_TARGETS:
        return {"status": "error", "error": f"unknown export target: {target}"}

    db = SessionLocal()
    storage = get_storage()
    try:
        document = db.get(Document, document_id)
        if document is None or document.is_deleted:
            return {"status": "error", "error": "document not found"}

        logger.info("export_regenerate_start", document_id=document_id, target=target, task_id=self.request.id)

        if target in ("searchable_pdf", "reconstructed_pdf"):
            if target == "searchable_pdf":
                result = document_service.rebuild_image_based_exports(db, storage, document)
            else:
                result = _regenerate_reconstructed_pdf(db, storage, document)
        elif target == "docx":
            result = _regenerate_docx(db, storage, document)
        else:
            result = _regenerate_excel(db, storage, document)

        logger.info("export_regenerate_done", document_id=document_id, target=target, result=result)
        return {"status": "ok", "target": target, **result}
    except Exception as exc:  # noqa: BLE001 - report the failure to the poller rather than a bare Celery traceback
        logger.error("export_regenerate_failed", document_id=document_id, target=target, error=str(exc))
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()


def _regenerate_reconstructed_pdf(db, storage, document: Document) -> dict:
    from app.models.enums import ExportType
    from app.reconstruction.pdf_renderer import render_document_pdf

    doc_json = document_service.get_document_json(db, document)
    if not doc_json.pages:
        return {"error": "document has not been processed yet"}
    pdf_bytes = render_document_pdf(doc_json.pages)
    relative_path = f"output/{document.id}/reconstructed.pdf"
    storage.write(relative_path, pdf_bytes)
    document_service.record_export_file(db, document, ExportType.RECONSTRUCTED_PDF, relative_path, len(pdf_bytes))
    return {"size_bytes": len(pdf_bytes)}


def _regenerate_docx(db, storage, document: Document) -> dict:
    from app.exporters.docx_exporter import render_document_docx
    from app.models.enums import ExportType

    doc_json = document_service.get_document_json(db, document)
    if not doc_json.pages:
        return {"error": "document has not been processed yet"}
    docx_bytes = render_document_docx(doc_json.pages)
    relative_path = f"output/{document.id}/document.docx"
    storage.write(relative_path, docx_bytes)
    document_service.record_export_file(db, document, ExportType.DOCX, relative_path, len(docx_bytes))
    return {"size_bytes": len(docx_bytes)}


def _regenerate_excel(db, storage, document: Document) -> dict:
    from app.exporters.excel_exporter import render_document_excel
    from app.models.enums import ExportType

    doc_json = document_service.get_document_json(db, document)
    if not doc_json.pages:
        return {"error": "document has not been processed yet"}
    excel_bytes = render_document_excel(doc_json.pages, document.original_filename)
    relative_path = f"output/{document.id}/data.xlsx"
    storage.write(relative_path, excel_bytes)
    document_service.record_export_file(db, document, ExportType.XLSX, relative_path, len(excel_bytes))
    return {"size_bytes": len(excel_bytes)}
