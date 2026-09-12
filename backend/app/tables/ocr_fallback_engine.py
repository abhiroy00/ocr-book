"""
Borderless-table fallback (spec section 29: "If table detection fails,
fallback to OCR text positioning"). Detects tabular structure purely from
OCR word alignment when no ruling-line grid is present:

  1. Cluster OCR words into rows by y-overlap.
  2. Within a run of consecutive, similarly-structured rows, cluster word
     x-start positions across rows into column bands.
  3. If at least MIN_ROWS rows share at least MIN_COLS consistent column
     bands, treat that region as a table.

This never rasterizes the table into an image — cells remain real
text-positioned data, just without explicit border geometry.
"""
from __future__ import annotations

import re

import numpy as np

from app.models.enums import TableDetectionMethod, TextAlign
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedCell, DetectedTable

_NUMERIC_TOKEN_RE = re.compile(r"^[+-]?[\d,./%-]*\d[\d,./%-]*$")


# A short paragraph (as few as 3 lines, all sharing the same left margin
# plus one more accidental word-position match) can still coincidentally
# pass a 3-row/3-column check. Real tabular data — the actual target here
# (district/rainfall-style data tables) — reliably has more rows than a
# short block of prose does; requiring 4 lowers the false-positive rate on
# paragraphs substantially without meaningfully hurting real tables.
MIN_ROWS = 4
# Regular paragraph text can coincidentally line up 2 x-positions across a
# few lines (e.g. every line happens to start at the same left margin, plus
# one more accidental alignment) — requiring 3 recurring columns is a much
# stronger signal that this is actually tabular data, not prose.
MIN_COLS = 3
COLUMN_BAND_TOLERANCE_RATIO = 0.02  # relative to page width
# A real table column repeats in nearly every row; prose rarely keeps 3+
# words starting at matching x-positions across most of a paragraph.
MIN_ROW_HIT_RATIO = 0.85


def detect_borderless_tables(words: list[OCRWordResult], page_width: int, page_height: int) -> list[DetectedTable]:
    if not words:
        return []

    rows = _cluster_rows(words)
    if len(rows) < MIN_ROWS:
        return []

    tolerance = page_width * COLUMN_BAND_TOLERANCE_RATIO
    tables: list[DetectedTable] = []

    i = 0
    while i < len(rows):
        run = [rows[i]]
        j = i + 1
        while j < len(rows):
            bands = _shared_column_bands(run + [rows[j]], tolerance)
            if len(bands) >= MIN_COLS:
                run.append(rows[j])
                j += 1
            else:
                break

        if len(run) >= MIN_ROWS:
            bands = _shared_column_bands(run, tolerance)
            if len(bands) >= MIN_COLS and _has_numeric_column(run, bands, tolerance):
                tables.append(_build_table_from_rows(run, bands))
        i = j if j > i + 1 else i + 1

    return tables


def _has_numeric_column(rows: list[list[OCRWordResult]], bands: list[float], tolerance: float) -> bool:
    """Real statistical/data tables (the actual target of this fallback --
    district/crop/rainfall/yield figures) reliably have at least one column
    of mostly numbers; justified prose essentially never does, even when
    (rarely) its line starts coincidentally line up into 3+ recurring x-
    positions the way `_shared_column_bands` requires. This was a real,
    confirmed false positive: a plain "Explanatory Note" paragraph page
    got split into several fake tables (row/column checks alone were
    satisfied) whose "cells" were prose fragments like "in India." and
    "There is," -- not tabular data. Requiring at least one predominantly-
    numeric column directly targets that failure mode without tightening
    the row/column thresholds further (which would risk losing real
    borderless tables that happen to have fewer rows)."""
    for band_center in bands:
        hits = 0
        numeric = 0
        for row in rows:
            band_words = [w for w in row if abs(w.bbox.x1 - band_center) <= tolerance * 2]
            if not band_words:
                continue
            hits += 1
            cell_text = " ".join(w.text for w in band_words).strip()
            if _NUMERIC_TOKEN_RE.match(cell_text.replace(" ", "")):
                numeric += 1
        if hits >= max(2, int(len(rows) * 0.6)) and numeric >= hits * 0.6:
            return True
    return False


def _cluster_rows(words: list[OCRWordResult]) -> list[list[OCRWordResult]]:
    ordered = sorted(words, key=lambda w: (w.bbox.y1 + w.bbox.y2) / 2)
    rows: list[list[OCRWordResult]] = []
    current: list[OCRWordResult] = []
    current_mid = None
    for w in ordered:
        mid = (w.bbox.y1 + w.bbox.y2) / 2
        height = w.bbox.height or 1
        if current_mid is None or abs(mid - current_mid) <= height * 0.6:
            current.append(w)
            mids = [(x.bbox.y1 + x.bbox.y2) / 2 for x in current]
            current_mid = float(np.mean(mids))
        else:
            rows.append(sorted(current, key=lambda x: x.bbox.x1))
            current = [w]
            current_mid = mid
    if current:
        rows.append(sorted(current, key=lambda x: x.bbox.x1))
    return [r for r in rows if len(r) >= MIN_COLS]


def _shared_column_bands(rows: list[list[OCRWordResult]], tolerance: float) -> list[float]:
    all_x_starts = sorted(w.bbox.x1 for row in rows for w in row)
    if not all_x_starts:
        return []
    bands: list[list[float]] = []
    for x in all_x_starts:
        if bands and x - bands[-1][-1] <= tolerance:
            bands[-1].append(x)
        else:
            bands.append([x])

    # Keep only bands that appear in at least half the rows (a real column,
    # not incidental alignment of a couple of words).
    good_bands = []
    for band in bands:
        band_center = float(np.mean(band))
        rows_hit = sum(
            1 for row in rows if any(abs(w.bbox.x1 - band_center) <= tolerance for w in row)
        )
        if rows_hit >= max(2, int(len(rows) * MIN_ROW_HIT_RATIO)):
            good_bands.append(band_center)
    return sorted(good_bands)


def _build_table_from_rows(rows: list[list[OCRWordResult]], bands: list[float]) -> DetectedTable:
    n_cols = len(bands)
    n_rows = len(rows)

    all_words = [w for row in rows for w in row]
    x1 = min(w.bbox.x1 for w in all_words)
    x2 = max(w.bbox.x2 for w in all_words)
    y1 = min(w.bbox.y1 for w in all_words)
    y2 = max(w.bbox.y2 for w in all_words)
    table_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    col_edges = bands + [x2 + 1]
    cells: list[DetectedCell] = []
    row_heights: list[float] = []
    for r_idx, row in enumerate(rows):
        row_top = min(w.bbox.y1 for w in row)
        row_bottom = max(w.bbox.y2 for w in row)
        row_heights.append(row_bottom - row_top)
        for c_idx in range(n_cols):
            band_left = bands[c_idx]
            band_right = col_edges[c_idx + 1]
            cell_words = [w for w in row if band_left - 5 <= w.bbox.x1 < band_right - 5]
            text = " ".join(w.text for w in sorted(cell_words, key=lambda w: w.bbox.x1)).strip()
            confidence = float(np.mean([w.confidence for w in cell_words])) if cell_words else 0.0
            cell_bbox = BBox(
                x1=band_left,
                y1=row_top,
                x2=band_right,
                y2=row_bottom,
            )
            cells.append(
                DetectedCell(
                    row=r_idx,
                    column=c_idx,
                    rowspan=1,
                    colspan=1,
                    bbox=cell_bbox,
                    text=text,
                    confidence=confidence,
                    align_h=TextAlign.CENTER if r_idx == 0 else TextAlign.LEFT,
                    is_header=(r_idx == 0),
                )
            )

    all_conf = [c.confidence for c in cells if c.confidence > 0]
    return DetectedTable(
        bbox=table_bbox,
        n_rows=n_rows,
        n_cols=n_cols,
        confidence=float(np.mean(all_conf)) * 0.85 if all_conf else 0.4,  # discount vs. bordered detection
        detection_method=TableDetectionMethod.OCR_FALLBACK,
        cells=cells,
        column_widths=[col_edges[i + 1] - bands[i] for i in range(n_cols)],
        row_heights=row_heights,
        border_style={"style": "none", "source": "ocr_alignment"},
    )
