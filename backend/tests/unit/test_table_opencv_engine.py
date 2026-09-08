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
