"""
A4 page coordinate transformation for document reconstruction.

Transforms blocks from original page coordinates to A4 coordinate system.
Ensures all output pages are exactly A4 size (595.2756 × 841.8898 points)
regardless of the original scanned page dimensions.

Transformation pipeline:
  1. Compute content bounding box from OCR words
  2. Determine optimal scaling to fit content on A4
  3. Compute offset to center content on A4
  4. Transform each block's bbox through the same transform
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult


# Standard A4 dimensions in PDF points (1/72 inch)
A4_PORTRAIT_PT = (595.2756, 841.8898)
A4_LANDSCAPE_PT = (841.8898, 595.2756)

# Minimum margin around content on A4 page (in mm)
_MIN_MARGIN_MM = 15.0
_PT_PER_MM = 72.0 / 25.4
_MIN_MARGIN_PT = _MIN_MARGIN_MM * _PT_PER_MM


@dataclass
class PageTransform:
    """Complete transformation from original page to A4 canvas."""

    # A4 page dimensions used
    a4_width_pt: float
    a4_height_pt: float

    # Content area (before scaling) in original pixel coordinates
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int

    # Scaling factor applied to content
    scale: float

    # Offset to center content on A4 page (in PDF points)
    offset_x_pt: float
    offset_y_pt: float

    # Transformed page dimensions in PDF points
    page_width_pt: float
    page_height_pt: float

    # DPI of the original rendering
    dpi: int

    def transform_bbox(self, bbox: BBox) -> BBox:
        """Transform a bounding box from original pixel coordinates to A4 PDF points.

        The transform includes:
        1. Crop to content region
        2. Scale to fit A4
        3. Translate to center on A4 page
        """
        pt_per_px = 72.0 / self.dpi

        # Step 1: Crop - remove the crop region offset
        x1_cropped = max(0, min(bbox.x2, self.crop_x2) - self.crop_x1)
        y1_cropped = max(0, min(bbox.y2, self.crop_y2) - self.crop_y1)
        x2_cropped = max(0, min(bbox.x2, self.crop_x2) - self.crop_x1)
        y2_cropped = max(0, min(bbox.y2, self.crop_y2) - self.crop_y1)

        # Actually: use original bbox coordinates, crop them to content region
        # then scale and offset
        x1 = max(0, min(bbox.x1, self.crop_x2) - self.crop_x1)
        y1 = max(0, min(bbox.y1, self.crop_y2) - self.crop_y1)
        x2 = max(0, min(bbox.x2, self.crop_x2) - self.crop_x1)
        y2 = max(0, min(bbox.y2, self.crop_y2) - self.crop_y1)

        # Step 2: Scale to A4
        x1_scaled = x1 * pt_per_px * self.scale
        y1_scaled = y1 * pt_per_px * self.scale
        x2_scaled = x2 * pt_per_px * self.scale
        y2_scaled = y2 * pt_per_px * self.scale

        # Step 3: Center on A4 page
        x1_final = x1_scaled + self.offset_x_pt
        y1_final = y1_scaled + self.offset_y_pt
        x2_final = x2_scaled + self.offset_x_pt
        y2_final = y2_scaled + self.offset_y_pt

        # Clip to A4 boundaries
        x1_final = max(0, min(x1_final, self.a4_width_pt))
        y1_final = max(0, min(y1_final, self.a4_height_pt))
        x2_final = max(0, min(x2_final, self.a4_width_pt))
        y2_final = max(0, min(y2_final, self.a4_height_pt))

        return BBox(x1=x1_final, y1=y1_final, x2=x2_final, y2=y2_final)


def compute_content_bbox(
    width: int, height: int, words: list[OCRWordResult], margin_px: int = 20
) -> tuple[int, int, int, int]:
    """Compute the content bounding box from OCR word boxes.

    Includes a margin around the content to ensure nothing is clipped.
    Falls back to full page if no words or content is too sparse.
    """
    real_words = [w for w in words if w.text.strip()]
    if not real_words:
        return 0, 0, width, height

    x1 = min(w.bbox.x1 for w in real_words)
    y1 = min(w.bbox.y1 for w in real_words)
    x2 = max(w.bbox.x2 for w in real_words)
    y2 = max(w.bbox.y2 for w in real_words)

    # Add margin
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(width, x2 + margin_px)
    y2 = min(height, y2 + margin_px)

    # Safety check: if content is too sparse, use full page
    if (x2 - x1) < width * 0.1 or (y2 - y1) < height * 0.1:
        return 0, 0, width, height

    return x1, y1, x2, y2


def compute_page_transform(
    width: int, height: int, words: list[OCRWordResult], dpi: int = 300
) -> PageTransform:
    """Compute the deterministic A4 normalization transform for a page.

    Returns a PageTransform that can be applied to both images and
    bounding boxes to normalize content to A4.

    The transform pipeline:
    1. Crop to content region (union of OCR word boxes + margin)
    2. Scale content to fill A4 area (preserving aspect ratio)
    3. Center content on A4 page
    """
    # Content bounding box with margin
    x1, y1, x2, y2 = compute_content_bbox(width, height, words, margin_px=20)
    content_w_px = x2 - x1
    content_h_px = y2 - y1

    # If content is too small, use full page
    if content_w_px < 10 or content_h_px < 10:
        x1, y1, x2, y2 = 0, 0, width, height

    pt_per_px = 72.0 / dpi
    content_w_pt = content_w_px * pt_per_px
    content_h_pt = content_h_px * pt_per_px

    # Determine portrait vs landscape based on content aspect ratio
    # Landscape only if content is significantly wider than tall
    is_landscape = content_w_pt > content_h_pt * 1.3

    a4_width_pt, a4_height_pt = A4_LANDSCAPE_PT if is_landscape else A4_PORTRAIT_PT

    # Available area on A4 page (with margins)
    available_w = a4_width_pt - 2 * _MIN_MARGIN_PT
    available_h = a4_height_pt - 2 * _MIN_MARGIN_PT

    # Scale content to fill available A4 area (always scale up, no cap)
    # Width determines scale if landscape, height determines if portrait
    if is_landscape:
        scale = available_w / max(content_w_pt, 1.0)
    else:
        scale = available_h / max(content_h_pt, 1.0)

    # Also scale the other dimension to check
    scaled_other = content_w_pt * scale if is_landscape else content_h_pt * scale
    if is_landscape and scaled_other > available_h * 0.95:
        scale = available_h / max(scaled_other, 1.0)
    elif not is_landscape and scaled_w > available_w * 0.95:
        scale = available_w / max(content_w_pt, 1.0)

    # Compute offsets to center content
    offset_x = (a4_width_pt - content_w_pt * scale) / 2
    offset_y = (a4_height_pt - content_h_pt * scale) / 2

    return PageTransform(
        a4_width_pt=a4_width_pt,
        a4_height_pt=a4_height_pt,
        crop_x1=x1,
        crop_y1=y1,
        crop_x2=x2,
        crop_y2=y2,
        scale=scale,
        offset_x_pt=offset_x,
        offset_y_pt=offset_y,
        page_width_pt=a4_width_pt,
        page_height_pt=a4_height_pt,
        dpi=dpi,
    )


def transform_block_to_a4(
    block: DocumentBlockJSON, page_json: PageJSON
) -> BBox:
    """Transform a document block's bbox from original page coordinates to A4.

    This is the main entry point for A4 normalization during PDF reconstruction.
    It uses the page's OCR words to compute the content boundary, then transforms
    the block's bbox through the same A4 normalization pipeline.

    Args:
        block: The document block with original pixel bbox
        page_json: The page JSON with DPI, dimensions, and OCR words

    Returns:
        BBox in A4 PDF point coordinates (0-595.2756 x 0-841.8898 for portrait)
    """
    # Compute the A4 transformation based on this page's content
    transform = compute_page_transform(
        page_json.page_width,
        page_json.page_height,
        page_json.words if hasattr(page_json, "words") else [],
        page_json.dpi,
    )

    # Transform the block's bbox
    a4_bbox = transform.transform_bbox(block.bbox)

    return a4_bbox