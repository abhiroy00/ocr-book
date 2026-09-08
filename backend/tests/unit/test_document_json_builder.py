from app.layout.detector import LayoutBlockResult
from app.models.enums import LayoutBlockType, TextAlign
from app.reconstruction.document_json_builder import build_page_json
from app.schemas.geometry import BBox
from app.tables.models import DetectedCell, DetectedTable
from app.models.enums import TableDetectionMethod


def test_build_page_json_includes_text_block():
    results = [
        LayoutBlockResult(
            block_type=LayoutBlockType.PARAGRAPH,
            bbox=BBox(x1=10, y1=10, x2=200, y2=50),
            confidence=0.9,
            z_order=0,
            text="Hello world",
            style={"font_size": 10.0},
        )
    ]
    page_json = build_page_json("doc1", "page1", 1, 1000, 1400, 300, 0.0, results, tables=[])

    assert page_json.page_number == 1
    assert len(page_json.blocks) == 1
    block = page_json.blocks[0]
    assert block.type == LayoutBlockType.PARAGRAPH
    assert block.content[0].text == "Hello world"
    assert 0 <= block.bbox_norm.x1 <= 1


def test_build_page_json_embeds_table_structure():
    table = DetectedTable(
        bbox=BBox(x1=0, y1=0, x2=200, y2=100),
        n_rows=1,
        n_cols=2,
        confidence=0.8,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=[
            DetectedCell(row=0, column=0, rowspan=1, colspan=1, bbox=BBox(x1=0, y1=0, x2=100, y2=100), text="A", align_h=TextAlign.LEFT),
            DetectedCell(row=0, column=1, rowspan=1, colspan=1, bbox=BBox(x1=100, y1=0, x2=200, y2=100), text="B", align_h=TextAlign.LEFT),
        ],
        column_widths=[100, 100],
        row_heights=[100],
    )
    results = [
        LayoutBlockResult(block_type=LayoutBlockType.TABLE, bbox=table.bbox, confidence=0.8, z_order=0, table_ref=0)
    ]
    page_json = build_page_json("doc1", "page1", 1, 1000, 1400, 300, 0.0, results, tables=[table])

    block = page_json.blocks[0]
    assert block.type == LayoutBlockType.TABLE
    assert block.table is not None
    assert len(block.table.rows) == 1
    assert [c.text for c in block.table.rows[0].cells] == ["A", "B"]


def test_document_json_is_json_serializable():
    results = [
        LayoutBlockResult(block_type=LayoutBlockType.HEADING, bbox=BBox(x1=0, y1=0, x2=100, y2=30), confidence=0.7, z_order=0, text="Title")
    ]
    page_json = build_page_json("doc1", "page1", 1, 1000, 1400, 300, 0.0, results, tables=[])
    dumped = page_json.model_dump(mode="json")
    assert dumped["page_number"] == 1
    assert isinstance(dumped["blocks"], list)
