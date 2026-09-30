"""
Integration tests for the pipeline's resume support
(`app.workers.pipeline_tasks._already_completed_pages`).

Regression for a confirmed real incident: `task_acks_late=True` means a
worker container being recreated mid-task (a redeploy, an OOM kill, a
host reboot) gets that document's whole task redelivered from the top --
without this, the redelivered run reprocessed EVERY page from scratch,
discarding however many hours of already-completed OCR/layout/table work
were already sitting in the DB. Observed firsthand: a real 420-page
document restarted at stage=render/5% despite all 420 of its pages
already showing `ocr_status=completed`.

These call the helper directly against the same in-memory-sqlite
fixtures `test_export_tasks.py` uses to build a fully-processed page,
rather than exercising the whole Celery task (which needs a live OCR
provider/worker pool) -- what matters here is that "which pages are
already done" is computed correctly, not the rest of the pipeline.
"""
from __future__ import annotations

from app.layout.detector import LayoutBlockResult
from app.models.enums import LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.services import document_service
from app.workers.pipeline_tasks import _already_completed_pages


def _make_document(test_db_session, tmp_storage, sample_png_bytes, page_count):
    document, _ = document_service.create_document_from_upload(
        test_db_session, tmp_storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    document.page_count = page_count
    test_db_session.commit()
    return document


def _finish_page(test_db_session, document, page_number, word_text="Hello"):
    """Persists a page exactly as the real pipeline does once its OCR/
    layout/table stage completes -- this is what sets `document_json`,
    the resume signal `_already_completed_pages` looks for."""
    page = document_service.upsert_page(
        test_db_session, document, page_number, 1000, 1400, 300, 0.0, document.storage_original_path
    )
    word = OCRWordResult(
        text=word_text, confidence=0.9, bbox=BBox(x1=10, y1=10, x2=100, y2=40),
        polygon=Polygon.from_xy_list([[10, 10], [100, 10], [100, 40], [10, 40]]),
        page_number=page_number, block_id="b0", line_id="l0",
    )
    layout_results = [
        LayoutBlockResult(block_type=LayoutBlockType.PARAGRAPH, bbox=word.bbox, confidence=0.9, z_order=0, text=word_text, word_ids=["b0:l0"]),
    ]
    processed_path = f"processed/page_{page_number:04d}.png"
    document_service.persist_page_pipeline_result(test_db_session, document, page, processed_path, [word], layout_results, [])
    return processed_path


def test_no_prior_work_returns_empty(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_document(test_db_session, tmp_storage, sample_png_bytes, page_count=3)

    done, words, paths = _already_completed_pages(test_db_session, document)

    assert done == set()
    assert words == {}
    assert paths == {}


def test_finished_pages_are_reported_done_with_their_words_and_paths(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_document(test_db_session, tmp_storage, sample_png_bytes, page_count=3)
    path1 = _finish_page(test_db_session, document, 1, word_text="Page One")
    path3 = _finish_page(test_db_session, document, 3, word_text="Page Three")
    # Page 2 deliberately left untouched -- as if the run was interrupted
    # partway through, after pages 1 and 3 (arrival order, not page order)
    # but before page 2.

    done, words, paths = _already_completed_pages(test_db_session, document)

    assert done == {1, 3}
    assert paths == {1: path1, 3: path3}
    assert [w.text for w in words[1]] == ["Page One"]
    assert [w.text for w in words[3]] == ["Page Three"]
    assert 2 not in done


def test_a_page_missing_its_processed_image_is_never_treated_as_done(test_db_session, tmp_storage, sample_png_bytes):
    """Defensive: `document_json` alone isn't trusted if the processed
    image path is somehow missing -- reconstruction needs that image, so
    treating such a page as "done" would just fail later instead of
    simply being retried like any other incomplete page."""
    document = _make_document(test_db_session, tmp_storage, sample_png_bytes, page_count=1)
    _finish_page(test_db_session, document, 1)

    from app.models.document_page import DocumentPage
    from sqlalchemy import select

    page = test_db_session.scalar(select(DocumentPage).where(DocumentPage.document_id == document.id))
    page.processed_image_path = None
    test_db_session.commit()

    done, words, paths = _already_completed_pages(test_db_session, document)

    assert done == set()


def test_all_pages_already_done_means_nothing_left_to_resubmit(test_db_session, tmp_storage, sample_png_bytes):
    document = _make_document(test_db_session, tmp_storage, sample_png_bytes, page_count=2)
    _finish_page(test_db_session, document, 1)
    _finish_page(test_db_session, document, 2)

    done, _, _ = _already_completed_pages(test_db_session, document)
    remaining = [n for n in range(1, document.page_count + 1) if n not in done]

    assert done == {1, 2}
    assert remaining == []
