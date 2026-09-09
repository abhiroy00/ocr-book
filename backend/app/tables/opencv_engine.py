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

    # Some real tables (common in typewriter-era documents like old Indian
    # government reports) rule only ONE axis -- e.g. vertical lines between
    # year columns, with rows separated by whitespace alone, no horizontal
    # rule between them. Relying solely on `row_bounds` from horizontal-line
    # detection there under-counts rows, which then forces multiple real
    # data rows into one detected row band -- their values get concatenated
    # or one overwrites another, which is exactly the "numbers shifted /
    # duplicated" defect this was built to fix. `find_candidate_table_regions`
    # already required real combined line evidence to treat this region as
    # a table candidate at all, so supplementing one sparse axis from text
    # clustering here is refining an already-confirmed table, not inventing
    # one from plain paragraph text.
    row_bounds, forced_row_seps = _supplement_sparse_axis(row_bounds, words, axis="row", offset=y1, length=ch)
    col_bounds, forced_col_seps = _supplement_sparse_axis(col_bounds, words, axis="col", offset=x1, length=cw)

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

    # A boundary that came purely from OCR-text clustering (no ruling line
    # backs it -- see `_supplement_sparse_axis`) has to be treated as an
    # unconditional separator here: `_line_present` checking the pixel mask
    # for ink at that position will correctly find none (there IS no line
    # there by construction), and the "no ink -> merge these cells" rule
    # that correctly detects a real table's merged/spanned cells would
    # otherwise glue every text-derived row/column straight back together,
    # undoing the whole point of having recovered them.
    for r in range(n_rows):
        for c in range(n_cols - 1):
            boundary = col_bounds[c + 1]
            if boundary not in forced_col_seps and not _line_present(
                vert_mask, boundary, row_bounds[r], row_bounds[r + 1], orientation="v"
            ):
                union(idx(r, c), idx(r, c + 1))
    for c in range(n_cols):
        for r in range(n_rows - 1):
            boundary = row_bounds[r + 1]
            if boundary not in forced_row_seps and not _line_present(
                horiz_mask, boundary, col_bounds[c], col_bounds[c + 1], orientation="h"
            ):
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

    # A grid of ruling lines can come from things that aren't tables at all:
    # scan artifacts/dirt streaks on an old document, print bleed-through,
    # or a stray long underline in running paragraph text. Those produce a
    # "table" whose cells are mostly EMPTY of OCR text (a real table almost
    # never is). Without this check, that false table still steals every
    # word whose bbox center falls inside its region away from normal
    # paragraph grouping — the words that don't land in a cell are simply
    # dropped — turning real paragraph text into rows of empty boxes with a
    # few stray word fragments. Reject it here instead, so those words stay
    # available for the layout detector's normal text grouping.
    non_empty_cells = sum(1 for c in cells if c.text.strip())
    if cells and (non_empty_cells / len(cells)) < 0.35:
        return None

    all_conf = [c.confidence for c in cells if c.confidence > 0]
    table_confidence = float(np.mean(all_conf)) if all_conf else 0.5

    column_widths = [col_bounds[i + 1] - col_bounds[i] for i in range(n_cols)]
    row_heights = [row_bounds[i + 1] - row_bounds[i] for i in range(n_rows)]

    # Use the extent actually covered by detected grid lines, not the
    # (possibly looser/padded, from the candidate region's dilation) input
    # `bbox` -- this is also the region the layout detector excludes from
    # normal paragraph grouping. If that exclusion zone is bigger than what
    # actually got captured into cells, a real value just outside the
    # captured columns gets silently dropped: excluded from paragraph
    # grouping (its center is still inside the old, looser bbox) but never
    # assigned to any cell either. Keeping the two in sync means every OCR
    # word is either in a cell or available to the paragraph grouper --
    # never neither.
    tight_bbox = BBox(
        x1=x1 + col_bounds[0], y1=y1 + row_bounds[0],
        x2=x1 + col_bounds[-1], y2=y1 + row_bounds[-1],
    )

    return DetectedTable(
        bbox=tight_bbox,
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
    return _cluster_positions([int(p) for p in positions], min_gap)


def _cluster_positions(positions: list[int], min_gap: int) -> list[int]:
    """Collapses a sorted-or-unsorted list of nearby integer positions into
    one representative (mean) position per cluster, where consecutive
    positions within `min_gap` of each other are treated as the same
    boundary line."""
    if not positions:
        return []
    ordered = sorted(positions)
    boundaries: list[int] = []
    cluster = [ordered[0]]
    for p in ordered[1:]:
        if p - cluster[-1] <= min_gap:
            cluster.append(p)
        else:
            boundaries.append(int(round(sum(cluster) / len(cluster))))
            cluster = [p]
    boundaries.append(int(round(sum(cluster) / len(cluster))))
    return boundaries


def _supplement_sparse_axis(
    line_bounds: list[int], words: list[OCRWordResult], axis: str, offset: float, length: int, min_gap: int = 8
) -> tuple[list[int], set[int]]:
    """If ruling-line detection on `axis` found far fewer bands than the
    OCR words visibly cluster into along that axis, merge in text-derived
    boundaries (union, then re-collapse near-duplicates) rather than
    letting several real rows/columns of data get forced into one
    detected band. Leaves `line_bounds` untouched when line evidence
    already looks adequate, or when there's no text to derive bounds
    from.

    Returns (bounds, forced_separator_positions): the second set flags
    which boundary values are NOT backed by any real ruling line -- the
    caller's merged-cell detection (which unions cells when it finds no
    ink at a boundary) must treat those as hard separators regardless,
    or it will just glue the newly-recovered rows/columns back together."""
    # The caller's `words` can include ones that only partially overlap this
    # region (engine.py's candidate-region matching tolerates ~30% overlap),
    # so a stray word centered well outside the crop could seed its own
    # cluster right at the crop edge -- exactly the kind of unsupported
    # edge boundary that previously produced an empty "picture frame" cell
    # (see the opencv_lines picture-frame regression test). Restrict
    # clustering to words actually centered inside this crop.
    if axis == "row":
        in_crop = [w for w in words if offset <= (w.bbox.y1 + w.bbox.y2) / 2 <= offset + length]
    else:
        in_crop = [w for w in words if offset <= (w.bbox.x1 + w.bbox.x2) / 2 <= offset + length]

    text_bounds = _text_cluster_boundaries(in_crop, axis, offset, length, min_gap)
    line_count = max(0, len(line_bounds) - 1)
    text_count = max(0, len(text_bounds) - 1)
    if text_count <= line_count + 1:
        return line_bounds, set()

    # A real ruling line and its nearest text-derived boundary rarely land
    # on the exact same pixel -- there's normally a few px of padding
    # between a printed rule and the text above/below it, often more than
    # this module's tight `min_gap` (tuned for distinguishing genuinely
    # separate *lines* from each other). Merging at that tight tolerance
    # left the outer border line un-collapsed with the nearest text
    # boundary, producing a spurious near-empty sliver row/column right at
    # the edge. Scale the merge tolerance to a fraction of the typical
    # spacing between boundaries instead of a fixed pixel count, so it
    # adapts across DPIs/font sizes while staying well short of merging
    # two genuinely distinct rows/columns into one.
    merged_positions = sorted(set(line_bounds) | set(text_bounds))
    gaps = [b - a for a, b in zip(merged_positions, merged_positions[1:])]
    typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else min_gap
    merge_gap = max(min_gap, int(typical_gap * 0.4))
    final_bounds = _cluster_positions(merged_positions, merge_gap)

    forced_separators = {b for b in final_bounds if not any(abs(b - lb) <= merge_gap for lb in line_bounds)}
    return final_bounds, forced_separators


def _text_cluster_boundaries(
    words: list[OCRWordResult], axis: str, offset: float, length: int, min_gap: int
) -> list[int]:
    """Derives boundary positions purely from OCR word clustering along
    `axis` -- for tables that rule only the OTHER axis (e.g. vertical
    column dividers with no horizontal rule between rows, common in
    typewriter-era documents). Returned positions are crop-relative (i.e.
    already offset by the crop's x1/y1), matching
    `_projection_boundaries`'s coordinate space, and clamped to [0, length].

    Rows and columns are NOT symmetric here. A normal sentence's words are
    reliably row-like (grouping by y-proximity within one text line is
    safe), but the *whitespace between two words in a sentence* looks
    identical, position-wise, to a real column gap -- clustering by
    x-proximity alone previously chopped an ordinary Hindi sentence into
    one spurious column per word (a real regression this function used to
    cause). A column gap is only trustworthy if it recurs at roughly the
    same x position across multiple text rows, which is what
    `_cross_row_column_boundaries` requires."""
    if not words:
        return []
    if axis == "row":
        return _single_axis_cluster_boundaries(words, offset, length)
    return _cross_row_column_boundaries(words, offset, length, min_gap)


def _single_axis_cluster_boundaries(words: list[OCRWordResult], offset: float, length: int) -> list[int]:
    ordered = sorted(words, key=lambda w: (w.bbox.y1 + w.bbox.y2) / 2)
    clusters: list[list[OCRWordResult]] = []
    current: list[OCRWordResult] = []
    current_mid = None
    for w in ordered:
        m = (w.bbox.y1 + w.bbox.y2) / 2
        if current_mid is None or abs(m - current_mid) <= (w.bbox.height or 1) * 0.7:
            current.append(w)
            current_mid = sum((x.bbox.y1 + x.bbox.y2) / 2 for x in current) / len(current)
        else:
            clusters.append(current)
            current = [w]
            current_mid = m
    if current:
        clusters.append(current)

    if len(clusters) < 2:
        return []

    raw_boundaries = [min(w.bbox.y1 for w in clusters[0])]
    for i in range(len(clusters) - 1):
        gap_start = max(w.bbox.y2 for w in clusters[i])
        gap_end = min(w.bbox.y1 for w in clusters[i + 1])
        raw_boundaries.append((gap_start + gap_end) / 2 if gap_end > gap_start else gap_start)
    raw_boundaries.append(max(w.bbox.y2 for w in clusters[-1]))

    return [int(round(max(0, min(length, b - offset)))) for b in raw_boundaries]


def _cross_row_column_boundaries(
    words: list[OCRWordResult], offset: float, length: int, min_gap: int
) -> list[int]:
    """Column boundaries derived from OCR text, requiring the same gap to
    recur across several text rows before it's trusted -- mirrors
    `app.tables.ocr_fallback_engine._shared_column_bands`'s safeguard
    against mistaking one sentence's natural word-spacing for real table
    columns. Needs at least 3 distinct text rows to establish a pattern;
    fewer than that returns no boundaries rather than guessing."""
    ordered = sorted(words, key=lambda w: (w.bbox.y1 + w.bbox.y2) / 2)
    rows: list[list[OCRWordResult]] = []
    current: list[OCRWordResult] = []
    current_mid = None
    for w in ordered:
        m = (w.bbox.y1 + w.bbox.y2) / 2
        if current_mid is None or abs(m - current_mid) <= (w.bbox.height or 1) * 0.6:
            current.append(w)
            current_mid = sum((x.bbox.y1 + x.bbox.y2) / 2 for x in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda x: x.bbox.x1))
            current = [w]
            current_mid = m
    if current:
        rows.append(sorted(current, key=lambda x: x.bbox.x1))

    # Same reasoning as the borderless (OCR-alignment) table engine's
    # MIN_ROWS/row-hit-ratio: a couple of words coincidentally starting at
    # similar x-positions across just 2 of a handful of rows is common in
    # ordinary prose (indentation, short words) and is not evidence of a
    # real column. Require at least 4 rows to establish a pattern at all,
    # and the gap to recur in a strong majority (75%) of them.
    if len(rows) < 4:
        return []

    tolerance = max(min_gap, length * 0.03)
    all_starts = sorted(w.bbox.x1 for row in rows for w in row)
    bands: list[list[float]] = []
    for x in all_starts:
        if bands and x - bands[-1][-1] <= tolerance:
            bands[-1].append(x)
        else:
            bands.append([x])

    good_starts = []
    for band in bands:
        center = sum(band) / len(band)
        rows_hit = sum(1 for row in rows if any(abs(w.bbox.x1 - center) <= tolerance for w in row))
        if rows_hit >= max(3, int(len(rows) * 0.75)):
            good_starts.append(center)
    good_starts = sorted(good_starts)

    if len(good_starts) < 2:
        return []

    raw_boundaries = [good_starts[0]]
    for a, b in zip(good_starts, good_starts[1:]):
        raw_boundaries.append((a + b) / 2)
    raw_boundaries.append(max(w.bbox.x2 for row in rows for w in row))

    return [int(round(max(0, min(length, b - offset)))) for b in raw_boundaries]


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
