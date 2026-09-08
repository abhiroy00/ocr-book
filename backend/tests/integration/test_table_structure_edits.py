"""
Covers the structural table editor operations (spec section 22): add/
delete row/column, merge cells — and that the edit is resynced into
DocumentPage.document_json (what reconstruction/export actually reads).
"""
from app.layout.detector import LayoutBlockResult
from app.models.enums import LayoutBlockType, OCRProviderEnum, PreprocessProfileEnum, TableDetectionMethod, TextAlign
from app.models.table import Table
from app.models.table_cell import TableCell
from app.schemas.geometry import BBox
from app.services import document_service
from app.tables.models import DetectedCell, DetectedTable


def _seed_2x2_table(db, storage, sample_png_bytes):
    document, _ = document_service.create_document_from_upload(
        db, storage, "a.png", sample_png_bytes, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )
    page = document_service.upsert_page(db, document, 1, 1000, 1000, 300, 0.0, document.storage_original_path)
    table = DetectedTable(
        bbox=BBox(x1=0, y1=0, x2=200, y2=200),
        n_rows=2,
        n_cols=2,
        confidence=0.9,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=[
            DetectedCell(row=0, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=0, x2=100, y2=100), text="A1", align_h=TextAlign.LEFT),
            DetectedCell(row=0, column=1, rowspan=1, colspan=1, bbox=BBox(x1=100, y1=0, x2=200, y2=100), text="B1", align_h=TextAlign.LEFT),
            DetectedCell(row=1, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=100, x2=100, y2=200), text="A2", align_h=TextAlign.LEFT),
            DetectedCell(row=1, column=1, rowspan=1, colspan=1, bbox=BBox(x1=100, y1=100, x2=200, y2=200), text="B2", align_h=TextAlign.LEFT),
        ],
        column_widths=[100, 100],
        row_heights=[100, 100],
    )
    layout_results = [LayoutBlockResult(block_type=LayoutBlockType.TABLE, bbox=table.bbox, confidence=0.9, z_order=0, table_ref=0)]
    document_service.persist_page_pipeline_result(db, document, page, "x.png", [], layout_results, [table])

    table_row = db.query(Table).filter(Table.page_id == page.id).first()
    return document, page, table_row


def test_add_row_shifts_existing_rows_down(test_db_session, tmp_storage, sample_png_bytes):
    _, page, table_row = _seed_2x2_table(test_db_session, tmp_storage, sample_png_bytes)

    updated = document_service.update_table_structure(test_db_session, table_row.id, "add_row", 1, None, None)

    assert updated.n_rows == 3
    cells_by_pos = {(c.row, c.column): c.text for c in updated.cells}
    assert cells_by_pos[(0, 0)] == "A1"
    assert cells_by_pos[(2, 0)] == "A2"  # shifted from row 1 -> row 2
    assert cells_by_pos[(1, 0)] == ""  # new empty row


def test_delete_row_removes_and_reindexes(test_db_session, tmp_storage, sample_png_bytes):
    _, page, table_row = _seed_2x2_table(test_db_session, tmp_storage, sample_png_bytes)

    updated = document_service.update_table_structure(test_db_session, table_row.id, "delete_row", 0, None, None)

    assert updated.n_rows == 1
    remaining = {(c.row, c.column): c.text for c in updated.cells}
    assert remaining == {(0, 0): "A2", (0, 1): "B2"}


def test_add_column_and_delete_column(test_db_session, tmp_storage, sample_png_bytes):
    _, page, table_row = _seed_2x2_table(test_db_session, tmp_storage, sample_png_bytes)

    updated = document_service.update_table_structure(test_db_session, table_row.id, "add_column", None, 1, None)
    assert updated.n_cols == 3

    updated2 = document_service.update_table_structure(test_db_session, table_row.id, "delete_column", None, 1, None)
    assert updated2.n_cols == 2
    remaining_texts = sorted(c.text for c in updated2.cells)
    assert remaining_texts == ["A1", "A2", "B1", "B2"]


def test_merge_cells_combines_into_one_span(test_db_session, tmp_storage, sample_png_bytes):
    _, page, table_row = _seed_2x2_table(test_db_session, tmp_storage, sample_png_bytes)
    top_row_cells = [c.id for c in table_row.cells if c.row == 0]

    updated = document_service.update_table_structure(test_db_session, table_row.id, "merge_cells", None, None, top_row_cells)

    merged = [c for c in updated.cells if c.row == 0]
    assert len(merged) == 1
    assert merged[0].colspan == 2
    assert "A1" in merged[0].text and "B1" in merged[0].text


def test_table_edit_resyncs_document_json(test_db_session, tmp_storage, sample_png_bytes):
    _, page, table_row = _seed_2x2_table(test_db_session, tmp_storage, sample_png_bytes)

    document_service.update_table_structure(test_db_session, table_row.id, "delete_row", 0, None, None)

    test_db_session.refresh(page)
    table_block = next(b for b in page.document_json["blocks"] if b["type"] == "table")
    assert len(table_block["table"]["rows"]) == 1
    assert table_block["table"]["rows"][0]["cells"][0]["text"] == "A2"
