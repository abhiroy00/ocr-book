"""
Non-text region detection: image / chart / signature / stamp / handwritten
(spec section 11). These are regions with visual ink content that OCR
either didn't recognize, or recognized with very low confidence.

Distinguishing these categories from pixels alone is inherently heuristic
(no trained classifier is used here — see README known limitations); the
thresholds below are deliberately conservative and geometry/position based,
using the *original* (non-binarized) image so color/edge cues survive.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult

MIN_REGION_AREA_RATIO = 0.002
MAX_REGION_AREA_RATIO = 0.35

# A real page essentially never carries more than a handful of genuine ink
# rubber-stamps (confirmed against this project's own real corpus: ~146
# STAMP blocks across a 206-page document, well under 1 per page on
# average). Crossing this on a single page is a confirmed real failure
# mode instead: a dense numeric table on badly-degraded/typewriter-font
# scans where OCR recognizes only a handful of words, and the small ink
# blob left behind by EACH unrecognized digit cluster/table cell —
# near-identical in size, since they're all one row height — gets
# classified as its own "stamp" one at a time. Rendering dozens of
# meaningless single-cell image crops in place of a table's numbers makes
# the reconstructed page actively worse than leaving that area blank, so
# past this count they are dropped instead of kept (see
# `_drop_excess_stamp_regions`).
MAX_PLAUSIBLE_STAMPS_PER_PAGE = 8


@dataclass
class GraphicRegion:
    bbox: BBox
    block_type: LayoutBlockType
    confidence: float


def detect_graphic_regions(
    original_image: np.ndarray,
    words: list[OCRWordResult],
    excluded_boxes: list[BBox],
    page_width: int,
    page_height: int,
) -> list[GraphicRegion]:
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove thin text-stroke content so what remains is blob-like ink:
    # a small opening kills isolated glyph strokes but preserves solid
    # marks (stamps, signatures, photos, rule art).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(opened, np.ones((9, 9), np.uint8), iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    page_area = page_width * page_height

    regions: list[GraphicRegion] = []
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        bbox = BBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
        area_ratio = area / page_area
        if area_ratio < MIN_REGION_AREA_RATIO or area_ratio > MAX_REGION_AREA_RATIO:
            continue
        if any(_overlap_ratio(bbox, ex) > 0.5 for ex in excluded_boxes):
            continue

        region_words = [w2 for w2 in words if _center_in(w2.bbox, bbox)]
        text_coverage = _text_area_coverage(region_words, bbox)
        if text_coverage > 0.35:
            # Mostly recognizable text (e.g. a dense heading) — not a graphic.
            continue

        block_type, confidence = _classify_region(bbox, region_words, page_width, page_height)
        regions.append(GraphicRegion(bbox=bbox, block_type=block_type, confidence=confidence))

    return _drop_excess_stamp_regions(regions)


def _drop_excess_stamp_regions(regions: list[GraphicRegion]) -> list[GraphicRegion]:
    stamps = [r for r in regions if r.block_type == LayoutBlockType.STAMP]
    if len(stamps) <= MAX_PLAUSIBLE_STAMPS_PER_PAGE:
        return regions
    excess = {id(r) for r in stamps}
    return [r for r in regions if id(r) not in excess]


def _classify_region(
    bbox: BBox, region_words: list[OCRWordResult], page_width: int, page_height: int
) -> tuple[LayoutBlockType, float]:
    aspect = bbox.width / max(1.0, bbox.height)
    rel_y = bbox.y1 / max(1, page_height)
    avg_conf = float(np.mean([w.confidence for w in region_words])) if region_words else 1.0
    has_low_conf_text = bool(region_words) and avg_conf < 0.45

    if has_low_conf_text:
        return LayoutBlockType.HANDWRITTEN, 0.55

    if rel_y > 0.65 and aspect >= 2.2 and bbox.height < 0.08 * page_height:
        return LayoutBlockType.SIGNATURE, 0.6

    if 0.6 <= aspect <= 1.6 and bbox.width < 0.18 * page_width:
        return LayoutBlockType.STAMP, 0.55

    return LayoutBlockType.IMAGE, 0.5


def _overlap_ratio(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    return inter / a_area


def _center_in(bbox: BBox, container: BBox) -> bool:
    cx, cy = (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
    return container.x1 <= cx <= container.x2 and container.y1 <= cy <= container.y2


def _text_area_coverage(words: list[OCRWordResult], bbox: BBox) -> float:
    region_area = max(1.0, bbox.width * bbox.height)
    text_area = sum(w.bbox.width * w.bbox.height for w in words)
    return min(1.0, text_area / region_area)
