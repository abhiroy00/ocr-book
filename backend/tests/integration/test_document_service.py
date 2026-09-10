"""
Integration tests for the document persistence service against a real
(in-memory sqlite) database — covers upload -> DB record -> page/OCR/table
persistence -> Document JSON assembly, without requiring Postgres/Redis/
Celery/PaddleOCR to be running (see docker-compose for the full stack).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from sqlalchemy import update

from app.layout.detector import LayoutBlockResult
from app.models.enums import DocumentStatus, LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum, TableDetectionMethod, TextAlign
from app.models.processing_job import ProcessingJob
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.services import document_service
from app.tables.models import DetectedCell, DetectedTable


def test_create_document_from_upload_persists_record(test_db_session, tmp_storage, sample_png_bytes):
    document, job = document_service.create_document_from_upload(
        test_db_session,
        tmp_storage,
        "scan.png",
        sample_png_bytes,
        OCRProviderEnum.PADDLEOCR,
        300,
        PreprocessProfileEnum.BALANCED,
    )

    assert document.id
    assert document.status == DocumentStatus.QUEUED
    assert document.page_count == 1
    assert tmp_storage.exists(document.storage_original_path)
    assert job.document_id == document.id


def test_list_documents_excludes_deleted(test_db_session, tmp_storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    document_service.soft_delete_document(test_db_session, tmp_storage, document)

    items, total = document_service.list_documents(test_db_session)
    assert total == 0
    assert items == []


def test_persist_page_pipeline_result_builds_document_json(test_db_session, tmp_storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    page = document_service.upsert_page(test_db_session, document, 1, 1000, 1400, 300, 0.0, document.storage_original_path)

    word = OCRWordResult(
        text="Hello",
        confidence=0.92,
        bbox=BBox(x1=10, y1=10, x2=100, y2=40),
        polygon=Polygon.from_xy_list([[10, 10], [100, 10], [100, 40], [10, 40]]),
        page_number=1,
        block_id="b0",
        line_id="l0",
    )
    table = DetectedTable(
        bbox=BBox(x1=0, y1=100, x2=200, y2=200),
        n_rows=1,
        n_cols=1,
        confidence=0.8,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=[DetectedCell(row=0, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=100, x2=200, y2=200), text="Cell", align_h=TextAlign.LEFT)],
    )
    layout_results = [
        LayoutBlockResult(block_type=LayoutBlockType.PARAGRAPH, bbox=word.bbox, confidence=0.9, z_order=0, text="Hello", word_ids=["b0:l0"]),
        LayoutBlockResult(block_type=LayoutBlockType.TABLE, bbox=table.bbox, confidence=0.8, z_order=1, table_ref=0),
    ]

    page_json = document_service.persist_page_pipeline_result(
        test_db_session, document, page, "processed/fake.png", [word], layout_results, [table]
    )

    assert len(page_json.blocks) == 2
    assert page.ocr_confidence_avg == 0.92
    assert page.table_confidence_avg == 0.8

    doc_json = document_service.get_document_json(test_db_session, document)
    assert doc_json.page_count == 1
    assert doc_json.pages[0].blocks[1].table.rows[0].cells[0].text == "Cell"


def test_update_table_cell_marks_edited(test_db_session, tmp_storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    page = document_service.upsert_page(test_db_session, document, 1, 1000, 1400, 300, 0.0, document.storage_original_path)
    table = DetectedTable(
        bbox=BBox(x1=0, y1=0, x2=100, y2=100), n_rows=1, n_cols=1, confidence=0.9,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=[DetectedCell(row=0, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=0, x2=100, y2=100), text="16.5", align_h=TextAlign.LEFT)],
    )
    layout_results = [LayoutBlockResult(block_type=LayoutBlockType.TABLE, bbox=table.bbox, confidence=0.9, z_order=0, table_ref=0)]
    document_service.persist_page_pipeline_result(test_db_session, document, page, "x.png", [], layout_results, [table])

    from app.models.table import Table

    table_row = test_db_session.query(Table).filter(Table.page_id == page.id).first()
    cell = table_row.cells[0]
    assert cell.text == "16.5"  # source of truth preserved until explicit user edit

    updated = document_service.update_table_cell(test_db_session, cell.id, text="16.5*")
    assert updated.is_edited is True
    assert updated.text == "16.5*"


def _backdate_job(db, job, seconds_ago: int) -> None:
    """Bypass the ORM's onupdate=utcnow on `updated_at` (a raw Core UPDATE,
    unlike an ORM-flushed change, doesn't trigger it) so tests can simulate
    a job that's genuinely been sitting untouched for a while."""
    old = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    db.execute(update(ProcessingJob).where(ProcessingJob.id == job.id).values(updated_at=old))
    db.commit()
    db.refresh(job)


def _make_active_job(test_db_session, tmp_storage, sample_png_bytes):
    document, job = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    job.status = DocumentStatus.OCR_PROCESSING
    test_db_session.commit()
    return document, job


def test_detect_and_fail_stale_job_leaves_recently_updated_job_alone(test_db_session, tmp_storage, sample_png_bytes):
    _, job = _make_active_job(test_db_session, tmp_storage, sample_png_bytes)
    result = document_service.detect_and_fail_stale_job(test_db_session, job)
    assert result.status == DocumentStatus.OCR_PROCESSING


def test_detect_and_fail_stale_job_leaves_queued_job_alone_even_if_old(test_db_session, tmp_storage, sample_png_bytes):
    """A QUEUED job has never acquired the lock yet -- that's normal, not stale."""
    document, job = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    _backdate_job(test_db_session, job, seconds_ago=3600)
    result = document_service.detect_and_fail_stale_job(test_db_session, job)
    assert result.status == DocumentStatus.QUEUED


def test_detect_and_fail_stale_job_leaves_active_job_alone_when_lock_still_held(test_db_session, tmp_storage, sample_png_bytes):
    _, job = _make_active_job(test_db_session, tmp_storage, sample_png_bytes)
    _backdate_job(test_db_session, job, seconds_ago=1000)

    fake_client = MagicMock()
    fake_client.exists.return_value = 1
    with patch("redis.Redis.from_url", return_value=fake_client):
        result = document_service.detect_and_fail_stale_job(test_db_session, job)

    assert result.status == DocumentStatus.OCR_PROCESSING


def test_detect_and_fail_stale_job_fails_job_when_lock_gone(test_db_session, tmp_storage, sample_png_bytes):
    """Regression test for a real incident: a worker was killed mid-task by
    a container restart, and the document sat reporting OCR_PROCESSING for
    57+ minutes with no active Celery task and no Redis lock -- silently
    unrecoverable without this check."""
    document, job = _make_active_job(test_db_session, tmp_storage, sample_png_bytes)
    _backdate_job(test_db_session, job, seconds_ago=1000)

    fake_client = MagicMock()
    fake_client.exists.return_value = 0
    with patch("redis.Redis.from_url", return_value=fake_client):
        result = document_service.detect_and_fail_stale_job(test_db_session, job)

    assert result.status == DocumentStatus.FAILED
    assert "stalled" in result.error_message.lower()
    test_db_session.refresh(document)
    assert document.status == DocumentStatus.FAILED
