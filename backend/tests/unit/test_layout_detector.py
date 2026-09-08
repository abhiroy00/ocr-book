import numpy as np

from app.layout.detector import LayoutDetector
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
