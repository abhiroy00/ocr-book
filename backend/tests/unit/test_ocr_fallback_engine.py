from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.tables.ocr_fallback_engine import detect_borderless_tables


def _word(text, x1, y1, x2, y2) -> OCRWordResult:
    return OCRWordResult(
        text=text, confidence=0.9, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
        page_number=1, block_id="b", line_id="l",
    )


def _make_borderless_grid(rows=4, cols=3):
    words = []
    col_x = [50, 250, 450]
    for r in range(rows):
        y1 = 100 + r * 40
        for c in range(cols):
            words.append(_word(f"r{r}c{c}", col_x[c], y1, col_x[c] + 80, y1 + 20))
    return words


def test_detects_borderless_table_from_aligned_words():
    words = _make_borderless_grid()
    tables = detect_borderless_tables(words, page_width=1000, page_height=1400)
    assert len(tables) == 1
    assert tables[0].n_cols == 3
    assert tables[0].n_rows == 4


def test_no_table_detected_for_scattered_words():
    words = [_word("a", 50, 100, 100, 120), _word("b", 400, 300, 450, 320), _word("c", 200, 600, 250, 620)]
    tables = detect_borderless_tables(words, page_width=1000, page_height=1400)
    assert tables == []


def test_no_table_for_empty_input():
    assert detect_borderless_tables([], 1000, 1400) == []


def test_wrapped_paragraph_is_not_misdetected_as_a_table():
    """Regression test for a live false positive: a title + a few
    left-aligned body/paragraph lines of varying length were detected as a
    borderless table because a couple of words happened to line up across
    lines by coincidence. Real prose should not trip the table detector."""
    lines = [
        [("HARYANA", 100, 60, 220, 90), ("AGRICULTURE", 230, 60, 400, 90), ("REPORT", 410, 60, 500, 90)],
        [("District", 100, 110, 190, 130), ("wise", 200, 110, 240, 130), ("rainfall", 250, 110, 330, 130), ("summary", 340, 110, 420, 130)],
        [("This", 100, 130, 140, 150), ("is", 150, 130, 170, 150), ("a", 180, 130, 195, 150), ("sample", 205, 130, 270, 150), ("paragraph", 280, 130, 380, 150)],
        [("and", 100, 150, 130, 170), ("layout", 140, 150, 200, 170), ("reconstruction", 210, 150, 340, 170), ("pipeline", 350, 150, 430, 170)],
    ]
    words = [_word(text, x1, y1, x2, y2) for line in lines for (text, x1, y1, x2, y2) in line]

    tables = detect_borderless_tables(words, page_width=1000, page_height=1400)
    assert tables == []
