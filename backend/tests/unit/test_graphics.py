"""
Tests for `app.layout.graphics` -- in particular the excess-stamp cap
(`_drop_excess_stamp_regions`). Regression for a confirmed real-world
defect: a badly-degraded/typewriter-font statistical table page where OCR
recognized only a handful of words left behind dozens of small, nearly
identical (one table-cell-sized) leftover ink blobs, each independently
classified as its own "stamp" -- 37 of them, on one real page. Rendering
that many meaningless single-cell image crops in place of the table's
actual numbers made the reconstructed output look far worse (and far more
"cut/broken") than a real document with 37 genuine ink stamps ever would.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.layout.graphics import MAX_PLAUSIBLE_STAMPS_PER_PAGE, detect_graphic_regions
from app.models.enums import LayoutBlockType

_PAGE_W, _PAGE_H = 900, 1200


def _blank_page() -> np.ndarray:
    return np.full((_PAGE_H, _PAGE_W, 3), 255, dtype=np.uint8)


def _draw_blob(image: np.ndarray, x: int, y: int, w: int = 42, h: int = 62) -> None:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)


def test_a_couple_of_genuine_looking_stamps_are_kept():
    image = _blank_page()
    _draw_blob(image, 100, 100, w=60, h=60)
    _draw_blob(image, 700, 900, w=55, h=65)

    regions = detect_graphic_regions(image, words=[], excluded_boxes=[], page_width=_PAGE_W, page_height=_PAGE_H)

    stamps = [r for r in regions if r.block_type == LayoutBlockType.STAMP]
    assert len(stamps) == 2


def test_a_dense_grid_of_near_identical_blobs_is_dropped_not_kept_as_stamps():
    """Simulates the real defect: a table's worth of unrecognized-digit
    ink blobs, all roughly the same (one row's) size."""
    image = _blank_page()
    n_cols, n_rows = 4, 4  # 16 blobs -- comfortably past the cap
    for row in range(n_rows):
        for col in range(n_cols):
            _draw_blob(image, x=60 + col * 150, y=60 + row * 150)

    regions = detect_graphic_regions(image, words=[], excluded_boxes=[], page_width=_PAGE_W, page_height=_PAGE_H)

    assert not any(r.block_type == LayoutBlockType.STAMP for r in regions)


def test_dropping_excess_stamps_does_not_touch_a_genuine_large_image():
    """A real full-page figure/map/photo on the SAME page as a garbled
    table must still come through -- the cap only targets the small,
    stamp-classified clutter, never a large IMAGE-classified region."""
    image = _blank_page()
    for row in range(4):
        for col in range(4):
            _draw_blob(image, x=60 + col * 150, y=60 + row * 150)
    cv2.rectangle(image, (600, 700), (880, 1150), (0, 0, 0), thickness=-1)  # a large, separate figure

    regions = detect_graphic_regions(image, words=[], excluded_boxes=[], page_width=_PAGE_W, page_height=_PAGE_H)

    assert not any(r.block_type == LayoutBlockType.STAMP for r in regions)
    assert any(r.block_type == LayoutBlockType.IMAGE for r in regions)


def test_cap_boundary_is_inclusive():
    image = _blank_page()
    positions = [(60 + col * 100, 60) for col in range(MAX_PLAUSIBLE_STAMPS_PER_PAGE)]
    for x, y in positions:
        _draw_blob(image, x, y, w=40, h=60)

    regions = detect_graphic_regions(image, words=[], excluded_boxes=[], page_width=_PAGE_W, page_height=_PAGE_H)

    stamps = [r for r in regions if r.block_type == LayoutBlockType.STAMP]
    assert len(stamps) == MAX_PLAUSIBLE_STAMPS_PER_PAGE
