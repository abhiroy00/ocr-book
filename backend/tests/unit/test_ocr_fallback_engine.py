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
    # Realistic-ish: a district-name column plus numeric data columns, since
    # real borderless data tables (the actual target of this engine) always
    # have at least one predominantly-numeric column -- see
    # `_has_numeric_column`.
    words = []
    col_x = [50, 250, 450]
    districts = ["Rohtak", "Hisar", "Sirsa", "Karnal"]
    for r in range(rows):
        y1 = 100 + r * 40
        for c in range(cols):
            text = districts[r % len(districts)] if c == 0 else f"{1000 + r * 111 + c * 17}"
            words.append(_word(text, col_x[c], y1, col_x[c] + 80, y1 + 20))
    return words


def test_detects_borderless_table_from_aligned_words():
    words = _make_borderless_grid()
    tables = detect_borderless_tables(words, page_width=1000, page_height=1400)
    assert len(tables) == 1
    assert tables[0].n_cols == 3
    assert tables[0].n_rows == 4


def test_all_text_grid_without_any_numeric_column_is_not_detected_as_a_table():
    """Regression test for a real, confirmed false positive: a plain
    "Explanatory Note" prose page (justified paragraph text in an old
    typewriter-style scan) got split into several fake borderless tables
    purely because enough lines' words coincidentally lined up into 3+
    x-position bands -- the row/column-alignment checks alone were
    satisfied, but the "cells" were prose fragments like "in India." and
    "There is," -- not tabular data. A real district/crop/statistics table
    (this engine's actual target, per its module docstring) always has at
    least one predominantly-numeric column; requiring one directly targets
    this failure mode."""
    words = []
    col_x = [50, 250, 450]
    # 5 rows x 3 "columns" that align by pure coincidence, all pure text.
    phrases = [
        ["in India.", "Vol. I,", "the report"],
        ["There is,", "however,", "some doubt"],
        ["This publication", "keeping in", "mind changes"],
        ["Production and", "yield levels", "are consistent"],
        ["for the", "estimates given", "in the text"],
    ]
    for r, row_texts in enumerate(phrases):
        y1 = 100 + r * 30
        for c, text in enumerate(row_texts):
            words.append(_word(text, col_x[c], y1, col_x[c] + 150, y1 + 18))

    tables = detect_borderless_tables(words, page_width=1000, page_height=1400)
    assert tables == []


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
