import cv2
import numpy as np

from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.tables.opencv_engine import extract_table_structure, find_candidate_table_regions


def _make_grid_image(rows=3, cols=3, cell_w=100, cell_h=50, margin=50) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    w = margin * 2 + cell_w * cols
    h = margin * 2 + cell_h * rows
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    for r in range(rows + 1):
        y = margin + r * cell_h
        cv2.line(img, (margin, y), (margin + cell_w * cols, y), (0, 0, 0), 2)
    for c in range(cols + 1):
        x = margin + c * cell_w
        cv2.line(img, (x, margin), (x, margin + cell_h * rows), (0, 0, 0), 2)

    cell_boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * cell_w
            y1 = margin + r * cell_h
            cell_boxes.append((x1, y1, x1 + cell_w, y1 + cell_h))
    return img, cell_boxes


def _tight_grid_bbox(cell_boxes, pad: int = 5) -> BBox:
    """A bbox around the grid with a small pad (a few px) — mirroring what
    `find_candidate_table_regions`'s dilated-connected-component boxes look
    like in real usage, as opposed to a box cut exactly through the line
    centers (which would clip a couple of pixels off the outer lines)."""
    return BBox(
        x1=min(b[0] for b in cell_boxes) - pad,
        y1=min(b[1] for b in cell_boxes) - pad,
        x2=max(b[2] for b in cell_boxes) + pad,
        y2=max(b[3] for b in cell_boxes) + pad,
    )


def _word_at(text: str, x1, y1, x2, y2) -> OCRWordResult:
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return OCRWordResult(
        text=text, confidence=0.95, bbox=bbox,
        polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
        page_number=1, block_id="b0", line_id="l0",
    )


def test_find_candidate_table_regions_detects_grid():
    img, _ = _make_grid_image()
    candidates = find_candidate_table_regions(img)
    assert len(candidates) >= 1


def test_extract_table_structure_recovers_row_col_count():
    img, cell_boxes = _make_grid_image(rows=3, cols=3)
    # Tight bbox around the grid itself, as `find_candidate_table_regions`
    # would produce from the connected-components of the line mask (not
    # the full padded image).
    grid_bbox = _tight_grid_bbox(cell_boxes)
    words = [_word_at(f"R{i}", *box) for i, box in enumerate(cell_boxes)]

    table = extract_table_structure(img, grid_bbox, words)
    assert table is not None
    assert table.n_rows == 3
    assert table.n_cols == 3
    assert len(table.cells) == 9


def test_extract_table_structure_assigns_text_to_correct_cells():
    img, cell_boxes = _make_grid_image(rows=2, cols=2)
    grid_bbox = _tight_grid_bbox(cell_boxes)
    labels = ["A1", "B1", "A2", "B2"]
    words = [_word_at(labels[i], *box) for i, box in enumerate(cell_boxes)]

    table = extract_table_structure(img, grid_bbox, words)
    assert table is not None
    texts = {(c.row, c.column): c.text for c in table.cells}
    assert set(texts.values()) == set(labels)


def test_extract_table_structure_returns_none_without_grid():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    bbox = BBox(x1=0, y1=0, x2=200, y2=200)
    assert extract_table_structure(blank, bbox, []) is None


def _make_vertical_lines_only_table(n_cols=3, col_w=150, row_h=40, n_rows=4, margin=20):
    """Simulates a real, observed print style (old typewriter-era Indian
    government reports, e.g. the 1967 'Area, Production and Yield' Table I)
    that rules an outer border and vertical column dividers only -- rows
    are separated by whitespace alone, with no horizontal line between
    them. Returns (image, words) where words are placed as if OCR read
    real content into each (row, col) cell, with no ruling lines to mark
    row boundaries beyond the outer border."""
    w = margin * 2 + col_w * n_cols
    h = margin * 2 + row_h * n_rows
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Outer border only.
    cv2.rectangle(img, (margin, margin), (margin + col_w * n_cols, margin + row_h * n_rows), (0, 0, 0), 2)
    # Vertical column dividers (interior only, full height).
    for c in range(1, n_cols):
        x = margin + c * col_w
        cv2.line(img, (x, margin), (x, margin + row_h * n_rows), (0, 0, 0), 2)
    # Deliberately NO horizontal lines between rows.

    words = []
    for r in range(n_rows):
        for c in range(n_cols):
            x1 = margin + c * col_w + 10
            y1 = margin + r * row_h + 8
            words.append(_word_at(f"r{r}c{c}", x1, y1, x1 + 60, y1 + 20))
    return img, words


def test_extract_table_structure_recovers_rows_with_no_horizontal_lines():
    """Regression test for a real bug found by comparing a live document's
    original scan against its reconstructed output: a table ruled with
    only vertical column lines (rows separated by whitespace, common in
    old typewriter-era print) was detected as a single giant row spanning
    the whole table, so every real data row's values landed in the same
    row band and got concatenated/overwritten -- the "numbers shifted or
    duplicated" defect. Row boundaries must be recovered from OCR word
    clustering when horizontal-line evidence is this sparse."""
    img, words = _make_vertical_lines_only_table(n_rows=4, n_cols=3)
    bbox = BBox(x1=0, y1=0, x2=img.shape[1], y2=img.shape[0])

    table = extract_table_structure(img, bbox, words)
    assert table is not None
    assert table.n_rows == 4
    assert table.n_cols == 3

    texts = {(c.row, c.column): c.text for c in table.cells}
    for r in range(4):
        for c in range(3):
            assert texts.get((r, c)) == f"r{r}c{c}", f"cell ({r},{c}) got {texts.get((r, c))!r}"


def _make_horizontal_lines_only_table(n_rows=4, col_w=150, row_h=40, n_cols=3, margin=20):
    """Symmetric counterpart of `_make_vertical_lines_only_table`: rules an
    outer border and horizontal row dividers only, with NO vertical lines
    between columns -- columns must be recovered from text-clustering."""
    w = margin * 2 + col_w * n_cols
    h = margin * 2 + row_h * n_rows
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.rectangle(img, (margin, margin), (margin + col_w * n_cols, margin + row_h * n_rows), (0, 0, 0), 2)
    for r in range(1, n_rows):
        y = margin + r * row_h
        cv2.line(img, (margin, y), (margin + col_w * n_cols, y), (0, 0, 0), 2)
    # Deliberately NO vertical lines between columns.

    words = []
    for r in range(n_rows):
        for c in range(n_cols):
            # Text spans most of the column width (small, roughly even
            # padding each side) -- as in real printed tables, where the
            # gap between two columns' text is clearly bigger than the gap
            # from the outermost column to the table's outer border. A
            # narrow fixed-width word regardless of column width leaves an
            # unrealistically large, inconsistent edge gap instead.
            x1 = margin + c * col_w + 15
            y1 = margin + r * row_h + 8
            words.append(_word_at(f"r{r}c{c}", x1, y1, margin + (c + 1) * col_w - 15, y1 + 20))
    return img, words


def test_extract_table_structure_recovers_columns_with_no_vertical_lines():
    """Symmetric case of the no-horizontal-lines regression: a table ruled
    with only row dividers, columns separated by whitespace alone, must
    still recover the correct column count and cell assignment."""
    img, words = _make_horizontal_lines_only_table(n_rows=4, n_cols=3)
    bbox = BBox(x1=0, y1=0, x2=img.shape[1], y2=img.shape[0])

    table = extract_table_structure(img, bbox, words)
    assert table is not None
    assert table.n_rows == 4
    assert table.n_cols == 3

    texts = {(c.row, c.column): c.text for c in table.cells}
    for r in range(4):
        for c in range(3):
            assert texts.get((r, c)) == f"r{r}c{c}", f"cell ({r},{c}) got {texts.get((r, c))!r}"


def test_supplement_sparse_axis_leaves_well_lined_table_untouched():
    """A table with real horizontal AND vertical lines on every row/column
    must not be altered by the text-clustering supplement -- it should
    only fill in an axis that line detection under-counted, never override
    good line evidence with a different row/column count."""
    img, cell_boxes = _make_grid_image(rows=3, cols=3, margin=20)
    grid_bbox = _tight_grid_bbox(cell_boxes)
    words = [_word_at(f"v{i}", *box) for i, box in enumerate(cell_boxes)]

    table = extract_table_structure(img, grid_bbox, words)
    assert table is not None
    assert table.n_rows == 3
    assert table.n_cols == 3
    assert len(table.cells) == 9


def test_column_supplementation_does_not_split_a_paragraph_into_one_word_per_column():
    """Regression test for a real bug found by re-running the hybrid
    row/column recovery fix against the live 446-page document: an
    ordinary wrapped Hindi sentence (several lines of naturally-varying
    word-start positions, no real column alignment) inside a region with
    only outer-border line evidence got chopped into one spurious column
    PER WORD (23 columns for a single sentence), because column-text-
    clustering was treating every inter-word gap in a sentence the same
    as a real inter-column gap. A column gap must recur across multiple
    text rows before it's trusted, and this sentence's word starts don't
    (each line wraps differently) -- so no column split should happen at
    all beyond whatever real line evidence exists (here, just the outer
    border -> 1 column)."""
    # An outer box (giving exactly 2 real vertical line positions: the left
    # and right border -> line-based n_cols=1) around 4 lines of a wrapped
    # sentence, each line's words starting at different x offsets, as real
    # wrapped prose does -- only the left margin recurs across all lines.
    margin = 20
    w, h = 500, 200
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 2)

    line_word_starts = [
        [30, 90, 170, 210],
        [30, 110, 170],
        [30, 90, 150, 190],
        [30, 110, 220],
    ]
    words = []
    for r, starts in enumerate(line_word_starts):
        y1 = margin + 15 + r * 35
        for c, x1 in enumerate(starts):
            words.append(_word_at(f"w{r}_{c}", x1, y1, x1 + 50, y1 + 20))

    bbox = BBox(x1=0, y1=0, x2=w, y2=h)
    table = extract_table_structure(img, bbox, words)

    # Either no table structure at all (further rejected downstream), or if
    # one comes back it must NOT have exploded into a column per word.
    if table is not None:
        assert table.n_cols <= 2, f"paragraph text was split into {table.n_cols} columns"


def test_extract_table_structure_rejects_mostly_empty_grid():
    """Regression test for a real, high-impact bug found by comparing a
    live document's original scan against its reconstructed output: scan
    artifacts / print-bleed on an old document can form a line grid over a
    region that is actually plain paragraph text. Previously, that false
    "table" claimed every OCR word whose center fell inside it (removing
    them from normal paragraph grouping) while its own cells stayed empty
    (words rarely land exactly inside a spurious cell), turning real
    paragraph text into rows of empty boxes with a few stray fragments.
    A detected grid whose cells are overwhelmingly empty must be rejected
    so those words are left for the layout detector's paragraph grouping."""
    img, cell_boxes = _make_grid_image(rows=4, cols=5, margin=20)
    grid_bbox = _tight_grid_bbox(cell_boxes)
    # Only 2 of the 20 cells actually have OCR text under them -- like a
    # paragraph's stray words that happen to fall inside a couple of the
    # spurious grid's cells.
    words = [_word_at("the", *cell_boxes[0]), _word_at("and", *cell_boxes[7])]

    table = extract_table_structure(img, grid_bbox, words)
    assert table is None


def test_full_pipeline_with_realistic_dilated_candidate_region_has_no_picture_frame_cell():
    """Regression test for a live bug: `find_candidate_table_regions`'s
    dilated connected-component boxes carry real margin (tens of px, not
    just a few) around the actual grid. Passing that looser candidate
    region straight into `extract_table_structure` (the real pipeline
    path) previously produced a spurious cell wrapping the entire table —
    a hairline "picture frame" band around the true grid with no line
    evidence, which the merged-cell union-find glued into one giant cell
    that then swallowed every other cell's text via bbox-overlap."""
    img, cell_boxes = _make_grid_image(rows=2, cols=3, margin=50)
    labels = ["District", "Rainfall", "Area", "Ambala", "16.5", "1574"]
    words = [_word_at(labels[i], *box) for i, box in enumerate(cell_boxes)]

    candidates = find_candidate_table_regions(img)
    assert len(candidates) >= 1
    region = candidates[0]

    table = extract_table_structure(img, region, words)
    assert table is not None
    assert table.n_rows == 2
    assert table.n_cols == 3
    assert len(table.cells) == 6

    # No cell should span the whole grid, and no cell's text should contain
    # more than one label (the tell-tale sign of the picture-frame bug).
    for cell in table.cells:
        assert not (cell.rowspan == table.n_rows and cell.colspan == table.n_cols)
        assert sum(1 for label in labels if label in cell.text) <= 1
