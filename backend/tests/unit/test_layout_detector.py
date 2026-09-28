import cv2
import numpy as np

from app.layout.detector import (
    LayoutDetector,
    _drop_text_blocks_inside_graphics,
    _suppress_nested_graphic_regions,
)
from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult


def _word(text, x1, y1, x2, y2, block_id, line_id, font_size=10.0) -> OCRWordResult:
    return OCRWordResult(
        text=text,
        confidence=0.9,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
        page_number=1,
        block_id=block_id,
        line_id=line_id,
        font_size_estimate=font_size,
    )


def test_detects_page_number_in_bottom_margin():
    page_w, page_h = 1000, 1400
    words = [
        _word("Body text here", 100, 200, 400, 220, "b0", "l0"),
        _word("19", 480, 1350, 520, 1370, "b1", "l1"),
    ]
    image = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    results = LayoutDetector().detect(image, image, words, tables=[], page_width=page_w, page_height=page_h)

    types = {r.text: r.block_type for r in results}
    assert types.get("19") == LayoutBlockType.PAGE_NUMBER


def test_larger_font_near_top_classified_as_heading_or_title():
    page_w, page_h = 1000, 1400
    words = [
        _word("DISTRICT SUMMARY", 100, 60, 600, 100, "b0", "l0", font_size=20.0),
        _word("Body text", 100, 200, 400, 220, "b1", "l1", font_size=10.0),
        _word("more body text here", 100, 230, 450, 250, "b1", "l2", font_size=10.0),
    ]
    image = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    results = LayoutDetector().detect(image, image, words, tables=[], page_width=page_w, page_height=page_h)

    heading_block = next(r for r in results if "DISTRICT SUMMARY" in r.text)
    assert heading_block.block_type in (LayoutBlockType.TITLE, LayoutBlockType.HEADING)


def test_paragraph_default_classification():
    page_w, page_h = 1000, 1400
    words = [
        _word("This is a normal paragraph of body text", 100, 300, 700, 320, "b0", "l0", font_size=10.0),
        _word("that continues onto a second line of text.", 100, 330, 650, 350, "b0", "l1", font_size=10.0),
    ]
    image = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    results = LayoutDetector().detect(image, image, words, tables=[], page_width=page_w, page_height=page_h)

    para = next(r for r in results if "normal paragraph" in r.text)
    assert para.block_type == LayoutBlockType.PARAGRAPH


def test_z_order_follows_reading_order():
    page_w, page_h = 1000, 1400
    words = [
        _word("Second block", 100, 400, 400, 420, "b1", "l0"),
        _word("First block", 100, 100, 400, 120, "b0", "l0"),
    ]
    image = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    results = LayoutDetector().detect(image, image, words, tables=[], page_width=page_w, page_height=page_h)
    sorted_by_z = sorted(results, key=lambda r: r.z_order)
    assert sorted_by_z[0].bbox.y1 <= sorted_by_z[-1].bbox.y1


def test_ocr_text_inside_a_detected_graphic_region_is_not_duplicated_as_a_paragraph():
    """Regression for a confirmed real-world defect: a village-name label
    hand-lettered ON a map is already part of that map's own image crop —
    OCR still partially reads it, but keeping that reading as a SEPARATE
    paragraph block draws a garbled duplicate of it on top of the correct
    picture in the reconstructed PDF/DOCX (found by rendering a real
    processed document: the map itself was intact, but its village list
    had a second, jumbled copy of the same text layered over it)."""
    page_w, page_h = 1000, 1400
    image = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    # A solid ink blob standing in for a map/photo -- big enough to pass
    # the graphic-region area filter, far from any table.
    cv2.rectangle(image, (100, 100), (500, 400), (0, 0, 0), thickness=-1)

    words = [
        # Falls entirely inside the blob -- OCR's (partial, low-quality)
        # reading of a label that's physically part of the picture.
        _word("Village", 200, 200, 280, 220, "b0", "l0"),
        # A normal paragraph elsewhere on the page, untouched by the filter.
        _word("Ordinary body paragraph below the figure.", 100, 600, 600, 620, "b1", "l0"),
    ]

    results = LayoutDetector().detect(image, image, words, tables=[], page_width=page_w, page_height=page_h)

    assert any(r.block_type == LayoutBlockType.IMAGE for r in results)
    assert not any(r.text == "Village" for r in results)
    assert any("Ordinary body paragraph" in r.text for r in results)


def _result(block_type, x1, y1, x2, y2, text="") -> "LayoutBlockResult":
    from app.layout.detector import LayoutBlockResult

    return LayoutBlockResult(block_type=block_type, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), confidence=0.9, text=text)


def _region(x1, y1, x2, y2):
    from app.layout.graphics import GraphicRegion

    return GraphicRegion(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), block_type=LayoutBlockType.IMAGE, confidence=0.5)


def test_drop_text_blocks_inside_graphics_removes_a_label_fully_inside_the_image():
    image_region = _region(0, 0, 500, 500)
    label_inside = _result(LayoutBlockType.PARAGRAPH, 100, 100, 200, 130, text="Village")
    paragraph_outside = _result(LayoutBlockType.PARAGRAPH, 0, 600, 300, 630, text="Body text")

    kept = _drop_text_blocks_inside_graphics([label_inside, paragraph_outside], [image_region])

    assert kept == [paragraph_outside]


def test_drop_text_blocks_inside_graphics_leaves_tables_and_lines_alone():
    image_region = _region(0, 0, 500, 500)
    table_inside = _result(LayoutBlockType.TABLE, 100, 100, 200, 130)

    kept = _drop_text_blocks_inside_graphics([table_inside], [image_region])

    assert kept == [table_inside]


def test_drop_text_blocks_inside_graphics_is_a_noop_with_no_graphic_regions():
    paragraph = _result(LayoutBlockType.PARAGRAPH, 100, 100, 200, 130, text="Village")
    assert _drop_text_blocks_inside_graphics([paragraph], []) == [paragraph]


def test_suppress_nested_graphic_regions_keeps_only_the_larger_one():
    big = _region(0, 0, 500, 500)
    nested = _region(100, 100, 200, 200)  # entirely inside `big`

    kept = _suppress_nested_graphic_regions([nested, big])

    assert kept == [big]


def test_suppress_nested_graphic_regions_keeps_non_overlapping_regions():
    left = _region(0, 0, 100, 100)
    right = _region(900, 900, 1000, 1000)

    kept = _suppress_nested_graphic_regions([left, right])

    assert {r.bbox.x1 for r in kept} == {0.0, 900.0}
