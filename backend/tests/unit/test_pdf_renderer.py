"""
Unit tests for `app.reconstruction.pdf_renderer` — in particular the table
cell text-fit fix (see `_render_table`'s docstring for the full incident).
"""
import fitz

from app.models.enums import LayoutBlockType, TextAlign
from app.reconstruction.pdf_renderer import render_document_pdf
from app.schemas.document_json import DocumentBlockJSON, PageJSON, TableBlockJSON, TableCellJSON, TableRowJSON
from app.schemas.geometry import BBox, NormBBox


def _table_cell(row: int, col: int, x1: float, y1: float, x2: float, y2: float, text: str) -> TableCellJSON:
    return TableCellJSON(
        row=row, column=col, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), text=text, align_h=TextAlign.LEFT, confidence=0.9
    )


def _table_page(rows: list[TableRowJSON], dpi: int = 150, page_width: int = 800, page_height: int = 1200) -> PageJSON:
    block = DocumentBlockJSON(
        id="b0",
        type=LayoutBlockType.TABLE,
        bbox=BBox(x1=40, y1=40, x2=700, y2=1000),
        bbox_norm=NormBBox(x1=0.05, y1=0.03, x2=0.87, y2=0.83),
        confidence=0.9,
        table=TableBlockJSON(rows=rows),
    )
    return PageJSON(document_id="d0", page_id="p0", page_number=1, page_width=page_width, page_height=page_height, dpi=dpi, blocks=[block])


def test_render_table_preserves_text_in_normal_height_rows():
    row = TableRowJSON(cells=[_table_cell(0, 0, 40, 40, 200, 70, "Irrigated"), _table_cell(0, 1, 200, 40, 360, 70, "18297")])
    page_json = _table_page([row])

    pdf_bytes = render_document_pdf([page_json])
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()
    doc.close()

    assert "Irrigated" in text
    assert "18297" in text


def test_render_table_does_not_drop_text_in_very_short_rows():
    """Regression test for a confirmed real-production defect: a dense
    statistical table's row heights (2.7-10.9pt after the old fixed 2pt
    inset, at real document DPI) were shorter than the old fixed 9pt font
    could ever fit, so `insert_textbox` silently inserted nothing for
    every such row while its border rectangle was still drawn -- an
    "empty grid of boxes" in the exported PDF with the real data missing
    entirely. Reproduces the exact real page-63 geometry pattern: a
    ~16px-tall header row and several ~14-30px data rows at 150 DPI."""
    rows = [
        TableRowJSON(cells=[_table_cell(0, 0, 40, 195, 178, 211, "1"), _table_cell(0, 1, 178, 195, 320, 211, "2")]),
        TableRowJSON(cells=[_table_cell(1, 0, 40, 211, 178, 240, "Irrigated"), _table_cell(1, 1, 178, 211, 320, 240, "18297")]),
        TableRowJSON(cells=[_table_cell(2, 0, 40, 240, 178, 254, "179190201404"), _table_cell(2, 1, 178, 240, 320, 254, "624 337")]),
    ]
    page_json = _table_page(rows)

    pdf_bytes = render_document_pdf([page_json])
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()
    doc.close()

    for expected in ("1", "2", "Irrigated", "18297", "179190201404", "624"):
        assert expected in text, f"expected {expected!r} in rendered table text, got: {text!r}"


def test_render_table_skips_genuinely_empty_cells_without_error():
    row = TableRowJSON(cells=[_table_cell(0, 0, 40, 40, 200, 70, ""), _table_cell(0, 1, 200, 40, 360, 70, "value")])
    page_json = _table_page([row])

    pdf_bytes = render_document_pdf([page_json])  # must not raise
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()
    doc.close()
    assert "value" in text
