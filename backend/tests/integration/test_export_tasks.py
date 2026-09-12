"""
Integration tests for `app.workers.export_tasks` -- the async regenerate
tasks that replaced blocking HTTP-request-time PDF/DOCX/Excel generation
(a confirmed real problem: a 145-page document's regenerate request
outlived a 300s client timeout while completing correctly server-side).

These call the task's underlying helper functions directly against the
same in-memory-sqlite fixtures the rest of the document_service
integration tests use, rather than going through `.delay()` (which needs
a live Celery broker) -- what matters here is that the regenerate logic
itself is correct and reports errors properly, not Celery's own dispatch
machinery.
"""
from unittest.mock import MagicMock, patch

from app.layout.detector import LayoutBlockResult
from app.models.enums import LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum, TableDetectionMethod, TextAlign
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.services import document_service
from app.tables.models import DetectedCell, DetectedTable
from app.workers.export_tasks import VALID_TARGETS, _regenerate_docx, _regenerate_excel, _regenerate_reconstructed_pdf


def _make_processed_document(test_db_session, tmp_storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    page = document_service.upsert_page(test_db_session, document, 1, 1000, 1400, 300, 0.0, document.storage_original_path)
    word = OCRWordResult(
        text="Hello", confidence=0.92, bbox=BBox(x1=10, y1=10, x2=100, y2=40),
        polygon=Polygon.from_xy_list([[10, 10], [100, 10], [100, 40], [10, 40]]),
        page_number=1, block_id="b0", line_id="l0",
    )
    table = DetectedTable(
        bbox=BBox(x1=0, y1=100, x2=200, y2=200), n_rows=1, n_cols=1, confidence=0.9,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=[DetectedCell(row=0, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=100, x2=200, y2=200), text="42", align_h=TextAlign.LEFT)],
    )
    layout_results = [
        LayoutBlockResult(block_type=LayoutBlockType.PARAGRAPH, bbox=word.bbox, confidence=0.9, z_order=0, text="Hello", word_ids=["b0:l0"]),
        LayoutBlockResult(block_type=LayoutBlockType.TABLE, bbox=table.bbox, confidence=0.9, z_order=1, table_ref=0),
    ]
    document_service.persist_page_pipeline_result(test_db_session, document, page, "x.png", [word], layout_results, [table])
    return document


def test_regenerate_docx_writes_export_file(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_processed_document(test_db_session, tmp_storage, sample_png_bytes)
    result = _regenerate_docx(test_db_session, tmp_storage, document)

    assert "size_bytes" in result and result["size_bytes"] > 0
    assert tmp_storage.exists(f"output/{document.id}/document.docx")


def test_regenerate_excel_writes_export_file(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_processed_document(test_db_session, tmp_storage, sample_png_bytes)
    result = _regenerate_excel(test_db_session, tmp_storage, document)

    assert "size_bytes" in result and result["size_bytes"] > 0
    assert tmp_storage.exists(f"output/{document.id}/data.xlsx")


def test_regenerate_reconstructed_pdf_writes_export_file(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_processed_document(test_db_session, tmp_storage, sample_png_bytes)
    result = _regenerate_reconstructed_pdf(test_db_session, tmp_storage, document)

    assert "size_bytes" in result and result["size_bytes"] > 0
    assert tmp_storage.exists(f"output/{document.id}/reconstructed.pdf")


def test_regenerate_helpers_report_error_for_unprocessed_document(test_db_session, tmp_storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    for fn in (_regenerate_docx, _regenerate_excel, _regenerate_reconstructed_pdf):
        result = fn(test_db_session, tmp_storage, document)
        assert "error" in result


def test_valid_targets_matches_what_the_api_dispatches():
    assert VALID_TARGETS == {"searchable_pdf", "reconstructed_pdf", "docx", "excel"}


def test_regenerate_document_export_rejects_unknown_target(test_db_session, tmp_storage, sample_png_bytes):
    """The top-level task must not silently no-op or crash on a bad
    target -- it must report a clear error the poller can surface."""
    document = _make_processed_document(test_db_session, tmp_storage, sample_png_bytes)

    with patch("app.workers.export_tasks.SessionLocal", return_value=test_db_session), \
         patch("app.workers.export_tasks.get_storage", return_value=tmp_storage), \
         patch.object(test_db_session, "close", lambda: None):  # keep the shared test session open for later assertions
        from app.workers.export_tasks import regenerate_document_export

        result = regenerate_document_export.run(document.id, "not_a_real_target")

    assert result["status"] == "error"


def test_regenerate_document_export_reports_error_for_missing_document():
    fake_db = MagicMock()
    fake_db.get.return_value = None
    with patch("app.workers.export_tasks.SessionLocal", return_value=fake_db), \
         patch("app.workers.export_tasks.get_storage", return_value=MagicMock()):
        from app.workers.export_tasks import regenerate_document_export

        result = regenerate_document_export.run("does-not-exist", "excel")

    assert result["status"] == "error"
    assert "not found" in result["error"]
