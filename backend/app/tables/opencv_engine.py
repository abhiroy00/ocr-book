"""
OpenCV grid-line based table structure extraction (spec section 12/13).

Pipeline:
  1. Build a combined horizontal+vertical ruling-line mask for the page.
  2. Find connected components of that mask that look like table grids
     (enough crossings, plausible size) -> candidate table bounding boxes.
  3. Within each candidate, derive row/column boundaries from the line
     projection profile.
  4. Detect merged cells by checking whether a grid separator line is
     actually present between each pair of adjacent base cells; where a
     line is missing, cells are unioned (rowspan/colspan > 1) via
     union-find.
  5. Assign OCR words to the resulting cells by bbox-center containment.
"""
from __future__ import annotations

import numpy as np
import cv2

from app.layout.lines import build_grid_mask
from app.models.enums import TableDetectionMethod, TextAlign
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedCell, DetectedTable

MIN_TABLE_WIDTH_RATIO = 0.15
MIN_TABLE_HEIGHT_RATIO = 0.03
MIN_GRID_LINES = 3  # at least this many combined h+v lines to call it a "table" region
LINE_PRESENCE_COVERAGE = 0.55


def find_candidate_table_regions(image: np.ndarray) -> list[BBox]:
    grid = build_grid_mask(image)
    h, w = grid.shape[:2]

    dilated = cv2.dilate(grid, np.ones((15, 15), np.uint8), iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    candidates: list[BBox] = []
    for label in range(1, n_labels):
        x, y, cw, ch, area = stats[label]
        if cw < MIN_TABLE_WIDTH_RATIO * w or ch < MIN_TABLE_HEIGHT_RATIO * h:
            continue
        component_mask = (labels[y : y + ch, x : x + cw] == label).astype(np.uint8) * 255
        raw_lines_in_region = grid[y : y + ch, x : x + cw]
        line_pixels = cv2.countNonZero(cv2.bitwise_and(raw_lines_in_region, component_mask))
        if line_pixels < 200:
            continue
        candidates.append(BBox(x1=float(x), y1=float(y), x2=float(x + cw), y2=float(y + ch)))
    return candidates


def extract_table_structure(image: np.ndarray, bbox: BBox, words: list[OCRWordResult]) -> DetectedTable | None:
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ch, cw = binary.shape[:2]

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, cw // 15), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, ch // 15)))
    horiz_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    row_bounds = _projection_boundaries(horiz_mask, axis=1, length=ch)
    col_bounds = _projection_boundaries(vert_mask, axis=0, length=cw)

    if len(row_bounds) < 2 or len(col_bounds) < 2:
        return None  # not enough grid structure to be confident this is a table

    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    if n_rows < 1 or n_cols < 1 or n_rows * n_cols > 2000:
        return None

    # Union-find over base grid cells; merge where the separating line is absent.
    parent = list(range(n_rows * n_cols))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols - 1):
            if not _line_present(vert_mask, col_bounds[c + 1], row_bounds[r], row_bounds[r + 1], orientation="v"):
                union(idx(r, c), idx(r, c + 1))
    for c in range(n_cols):
        for r in range(n_rows - 1):
            if not _line_present(horiz_mask, row_bounds[r + 1], col_bounds[c], col_bounds[c + 1], orientation="h"):
                union(idx(r, c), idx(r + 1, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            groups.setdefault(find(idx(r, c)), []).append((r, c))

    cells: list[DetectedCell] = []
    for members in groups.values():
        rows = [m[0] for m in members]
        cols = [m[1] for m in members]
        r0, r1c = min(rows), max(rows)
        c0, c1c = min(cols), max(cols)
        cell_bbox = BBox(
            x1=x1 + col_bounds[c0],
            y1=y1 + row_bounds[r0],
            x2=x1 + col_bounds[c1c + 1],
            y2=y1 + row_bounds[r1c + 1],
        )
        cell_words = [w for w in words if _center_in(w.bbox, cell_bbox)]
        cell_words.sort(key=lambda w: (round(w.bbox.y1 / 8), w.bbox.x1))
        text = " ".join(w.text for w in cell_words).strip()
        confidence = float(np.mean([w.confidence for w in cell_words])) if cell_words else 0.0

        cells.append(
            DetectedCell(
                row=r0,
                column=c0,
                rowspan=(r1c - r0 + 1),
                colspan=(c1c - c0 + 1),
                bbox=cell_bbox,
                text=text,
                confidence=confidence,
                align_h=TextAlign.CENTER if r0 == 0 else TextAlign.LEFT,
                is_header=(r0 == 0),
            )
        )

    all_conf = [c.confidence for c in cells if c.confidence > 0]
    table_confidence = float(np.mean(all_conf)) if all_conf else 0.5

    column_widths = [col_bounds[i + 1] - col_bounds[i] for i in range(n_cols)]
    row_heights = [row_bounds[i + 1] - row_bounds[i] for i in range(n_rows)]

    return DetectedTable(
        bbox=bbox,
        n_rows=n_rows,
        n_cols=n_cols,
        confidence=table_confidence,
        detection_method=TableDetectionMethod.OPENCV_LINES,
        cells=cells,
        column_widths=column_widths,
        row_heights=row_heights,
        border_style={"style": "grid", "source": "opencv_lines"},
    )


def _projection_boundaries(mask: np.ndarray, axis: int, length: int, min_gap: int = 8) -> list[int]:
    """Finds line positions from a binary ruling-line mask's projection
    profile, returning sorted boundary coordinates.

    Deliberately does NOT pad the result with synthetic 0/`length` edge
    boundaries: a real table's outer border is itself a detected line (it
    shows up in `positions` like any interior line), so padding was only
    ever needed when a boundary got clipped by a too-tight crop. In
    practice the candidate region always carries a few pixels of slack
    (from `find_candidate_table_regions`'s dilation), so the outer border
    is fully visible — and adding a synthetic edge boundary anyway created
    a hairline pseudo-row/column with no line evidence, which the
    merged-cell union-find (correctly) treats as "no separator" and glues
    into a `picture-frame' cell wrapping the whole table. Boundaries here
    span only the extent actually bounded by real detected lines, which is
    exactly the table's interior grid — the correct extent to build cells
    from."""
    profile = mask.sum(axis=axis) if axis == 1 else mask.sum(axis=axis)
    # axis=1 -> sum across columns for each row (horizontal lines -> row y positions)
    # axis=0 -> sum across rows for each column (vertical lines -> column x positions)
    threshold = profile.max() * 0.4 if profile.max() > 0 else 0
    positions = np.where(profile > max(threshold, 1))[0]
    if len(positions) == 0:
        return []

    # Cluster consecutive positions into single boundary lines.
    boundaries: list[int] = []
    cluster = [int(positions[0])]
    for p in positions[1:]:
        if p - cluster[-1] <= min_gap:
            cluster.append(int(p))
        else:
            boundaries.append(int(np.mean(cluster)))
            cluster = [int(p)]
    boundaries.append(int(np.mean(cluster)))
    return boundaries


def _line_present(mask: np.ndarray, at: int, span_start: int, span_end: int, orientation: str, band: int = 3) -> bool:
    h, w = mask.shape[:2]
    # A vertical separator's span runs along y (row range); a horizontal
    # separator's span runs along x (column range) — clip against the
    # matching axis, not the perpendicular one.
    span_limit = h if orientation == "v" else w
    span_start, span_end = max(0, span_start), min(span_limit, span_end)
    if span_end <= span_start:
        return False
    if orientation == "v":
        c0, c1 = max(0, at - band), min(w, at + band)
        region = mask[span_start:span_end, c0:c1]
    else:
        r0, r1 = max(0, at - band), min(h, at + band)
        region = mask[r0:r1, span_start:span_end]
    if region.size == 0:
        return False
    coverage = cv2.countNonZero(region) / region.size
    return coverage >= LINE_PRESENCE_COVERAGE * 0.2  # band dilutes coverage; low absolute threshold is fine here


def _center_in(bbox: BBox, container: BBox) -> bool:
    cx, cy = (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
    return container.x1 <= cx <= container.x2 and container.y1 <= cy <= container.y2
