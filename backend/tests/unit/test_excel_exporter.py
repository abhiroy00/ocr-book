"""
Unit tests for `app.exporters.excel_exporter` (Pipeline B — structured
data extraction). Numeric-integrity preservation is the highest-priority
requirement here (spec section 47): every assertion checks the cell text
round-trips byte-for-byte, never reformatted/coerced to a numeric type.
"""
from openpyxl import load_workbook

from app.exporters.excel_exporter import render_document_excel
from app.models.enums import LayoutBlockType, TextAlign
from app.schemas.document_json import DocumentBlockJSON, PageJSON, TableBlockJSON, TableCellJSON, TableRowJSON, TextRunJSON
from app.schemas.geometry import BBox, NormBBox


def _cell(row, col, text, is_header=False):
    return TableCellJSON(row=row, column=col, bbox=BBox(x1=0, y1=0, x2=10, y2=10), text=text, is_header=is_header, confidence=0.9)


def _table_page(page_number, rows, heading_text=None):
    blocks = []
    z = 0
    if heading_text:
        blocks.append(
            DocumentBlockJSON(
                id="heading", type=LayoutBlockType.HEADING, bbox=BBox(x1=0, y1=0, x2=100, y2=20),
                bbox_norm=NormBBox(x1=0, y1=0, x2=1, y2=0.02), confidence=0.9, z_order=z,
                content=[TextRunJSON(text=heading_text)],
            )
        )
        z += 1
    blocks.append(
        DocumentBlockJSON(
            id="table", type=LayoutBlockType.TABLE, bbox=BBox(x1=0, y1=30, x2=200, y2=200),
            bbox_norm=NormBBox(x1=0, y1=0.03, x2=1, y2=0.5), confidence=0.87, z_order=z,
            table=TableBlockJSON(rows=rows),
        )
    )
    return PageJSON(document_id="d0", page_id=f"p{page_number}", page_number=page_number, page_width=800, page_height=1200, dpi=150, blocks=blocks)


def test_render_document_excel_creates_info_index_and_table_sheets():
    rows = [
        TableRowJSON(cells=[_cell(0, 0, "District", is_header=True), _cell(0, 1, "Total", is_header=True)]),
        TableRowJSON(cells=[_cell(1, 0, "Rohtak"), _cell(1, 1, "45189")]),
    ]
    page = _table_page(1, rows, heading_text="District-wise Totals")

    wb_bytes = render_document_excel([page], "book.pdf")
    wb = load_workbook(__import__("io").BytesIO(wb_bytes))

    assert "Document_Info" in wb.sheetnames
    assert "Table_Index" in wb.sheetnames
    assert "Table_001" in wb.sheetnames

    index = wb["Table_Index"]
    assert index.cell(row=2, column=3).value == "District-wise Totals"
    assert index.cell(row=2, column=4).value == 1  # source page


def test_render_document_excel_preserves_numeric_text_exactly():
    """Regression guard: '33,541' and '0.25' must never be reparsed into a
    different string (e.g. dropping the comma, or Excel's own numeric
    auto-formatting silently truncating '0.25' to '.25')."""
    rows = [
        TableRowJSON(cells=[_cell(0, 0, "Year", is_header=True), _cell(0, 1, "Value", is_header=True)]),
        TableRowJSON(cells=[_cell(1, 0, "1995"), _cell(1, 1, "33,541")]),
        TableRowJSON(cells=[_cell(2, 0, "1996"), _cell(2, 1, "0.25")]),
    ]
    page = _table_page(1, rows)

    wb_bytes = render_document_excel([page], "book.pdf")
    wb = load_workbook(__import__("io").BytesIO(wb_bytes))
    sheet = wb["Table_001"]

    values = [cell.value for row in sheet.iter_rows(min_row=4, max_row=6, max_col=2) for cell in row]
    assert "1995" in values
    assert "33,541" in values
    assert "1996" in values
    assert "0.25" in values


def test_render_document_excel_handles_multiple_tables_across_pages():
    rows_a = [TableRowJSON(cells=[_cell(0, 0, "A", is_header=True)]), TableRowJSON(cells=[_cell(1, 0, "1")])]
    rows_b = [TableRowJSON(cells=[_cell(0, 0, "B", is_header=True)]), TableRowJSON(cells=[_cell(1, 0, "2")])]
    page1 = _table_page(1, rows_a)
    page2 = _table_page(2, rows_b)

    wb_bytes = render_document_excel([page1, page2], "book.pdf")
    wb = load_workbook(__import__("io").BytesIO(wb_bytes))

    assert "Table_001" in wb.sheetnames
    assert "Table_002" in wb.sheetnames
    assert wb["Table_Index"].cell(row=3, column=4).value == 2  # table 2's source page


def test_render_document_excel_skips_pages_with_no_tables():
    text_only_page = PageJSON(
        document_id="d0", page_id="p1", page_number=1, page_width=800, page_height=1200, dpi=150,
        blocks=[
            DocumentBlockJSON(
                id="p", type=LayoutBlockType.PARAGRAPH, bbox=BBox(x1=0, y1=0, x2=100, y2=20),
                bbox_norm=NormBBox(x1=0, y1=0, x2=1, y2=0.02), confidence=0.9, z_order=0,
                content=[TextRunJSON(text="Just a paragraph, no tables here.")],
            )
        ],
    )

    wb_bytes = render_document_excel([text_only_page], "book.pdf")  # must not raise
    wb = load_workbook(__import__("io").BytesIO(wb_bytes))
    assert "Table_001" not in wb.sheetnames
    assert wb["Document_Info"].cell(row=5, column=2).value == 0  # "Tables extracted"
