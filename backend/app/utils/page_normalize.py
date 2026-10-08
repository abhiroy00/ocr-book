"""
A4 page normalization for the searchable PDF.

Computes a content-aware crop from OCR word boxes -- never raw pixel/ink
analysis (a fixed crop percentage or `image.getbbox()` would risk cutting
real content or leaving artifact-driven margins) -- then a deterministic
crop -> scale -> center transform onto a standard A4 canvas. Every OCR
word's bounding box is run through the *exact same* transform used to
place the image, which is what keeps the invisible searchable text layer
aligned with the now-repositioned visible content.

This is pure geometry (no OCR, no AI, no re-processing) so it's safe to
apply on every export/regenerate without re-running the expensive parts
of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult

# Standard A4 in PDF points (1/72 inch), the same values PyMuPDF's own
# `fitz.paper_size("a4")` returns -- kept as explicit constants so this
# module has no hidden dependency on a particular PyMuPDF version's table.
A4_PORTRAIT_PT = (595.2756, 841.8898)
A4_LANDSCAPE_PT = (841.8898, 595.2756)

_MARGIN_MM = 18.0
_PT_PER_MM = 72.0 / 25.4
_MARGIN_PT = _MARGIN_MM * _PT_PER_MM

# Content must be at least this many times wider than tall before we place
# it on a landscape A4 page. A page only "slightly" wider than tall (a
# common side-effect of a loose content bbox) should stay portrait --
# genuinely landscape content (a wide statistical table, a fold-out map)
# clears this by a wide margin.
_LANDSCAPE_ASPECT_THRESHOLD = 1.25

# Content bbox must cover at least this fraction of the original page in
# both dimensions to be trusted; anything tighter is treated as a
# degenerate/unreliable detection (e.g. one stray OCR misfire) and the
# full original page is kept uncropped rather than risking an aggressive,
# wrong crop.
_MIN_CONTENT_FRACTION = 0.10


@dataclass
class PageTransform:
    """Everything needed to place a page's cropped content onto an A4
    canvas, and to carry any pixel-space bounding box through the same
    transform so the OCR text layer stays aligned with the image."""

    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    scale: float
    offset_x_pt: float
    offset_y_pt: float
    page_width_pt: float
    page_height_pt: float
    dpi: int

    @property
    def is_landscape(self) -> bool:
        return self.page_width_pt > self.page_height_pt

    def transform_bbox(self, bbox: BBox) -> BBox:
        """Original-image-pixel bbox -> crop -> scale -> A4-placement, in
        PDF points. Clipped to the crop region first so a word that
        straddles the crop boundary (shouldn't happen given the crop is
        built to contain every word, but a defensive clip costs nothing)
        never produces a negative or out-of-page coordinate."""
        x1 = min(max(bbox.x1, self.crop_x1), self.crop_x2) - self.crop_x1
        y1 = min(max(bbox.y1, self.crop_y1), self.crop_y2) - self.crop_y1
        x2 = min(max(bbox.x2, self.crop_x1), self.crop_x2) - self.crop_x1
        y2 = min(max(bbox.y2, self.crop_y1), self.crop_y2) - self.crop_y1

        pt_per_px = 72.0 / self.dpi
        return BBox(
            x1=x1 * pt_per_px * self.scale + self.offset_x_pt,
            y1=y1 * pt_per_px * self.scale + self.offset_y_pt,
            x2=x2 * pt_per_px * self.scale + self.offset_x_pt,
            y2=y2 * pt_per_px * self.scale + self.offset_y_pt,
        )


def compute_content_bbox(
    width: int, height: int, words: list[OCRWordResult], margin_px: int, extra_boxes: list[BBox] | None = None
) -> tuple[int, int, int, int]:
    """Union of every non-blank OCR word's bounding box plus any extra
    content boxes (detected graphic/painting regions, which carry no OCR
    words of their own), expanded by `margin_px`, clamped to the page.
    The graphic boxes are what keeps a cover painting/photo from being
    cropped out of the searchable PDF just because the OCR text sits in a
    small title region -- text-only pages (no graphics) behave exactly as
    before, since there is nothing extra to union.

    Falls back to the full, uncropped page for a genuinely sparse page
    (near-blank, or a content region suspiciously smaller than a
    trustworthy detection) rather than risk an aggressive wrong crop --
    a title page with three lines of text should still look like a
    proper book page, not a tiny cropped sliver.
    """
    real_words = [w for w in words if w.text.strip()]
    boxes = [w.bbox for w in real_words]
    for b in extra_boxes or []:
        if b.width > 0 and b.height > 0:
            boxes.append(b)
    if not boxes:
        return 0, 0, width, height

    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)

    x1 = max(0, int(x1 - margin_px))
    y1 = max(0, int(y1 - margin_px))
    x2 = min(width, int(x2 + margin_px))
    y2 = min(height, int(y2 + margin_px))

    if (x2 - x1) < width * _MIN_CONTENT_FRACTION or (y2 - y1) < height * _MIN_CONTENT_FRACTION:
        return 0, 0, width, height

    return x1, y1, x2, y2


def compute_page_transform(
    width: int, height: int, words: list[OCRWordResult], dpi: int, extra_boxes: list[BBox] | None = None
) -> PageTransform:
    """The single source of truth for how a page's image AND its OCR
    words both get placed on the output A4 canvas -- callers must use the
    same `PageTransform` instance for both, or the invisible text layer
    will drift away from the visible content."""
    # ~2mm safety margin around detected content, scaled to the actual
    # render DPI so it behaves consistently regardless of input DPI
    # (150/200/300/400/600) -- a fixed pixel margin would be too tight at
    # high DPI and too loose at low DPI.
    margin_px = max(3, int(dpi * (2.0 / 25.4)))
    x1, y1, x2, y2 = compute_content_bbox(width, height, words, margin_px, extra_boxes)
    content_w_px, content_h_px = x2 - x1, y2 - y1

    pt_per_px = 72.0 / dpi
    content_w_pt = content_w_px * pt_per_px
    content_h_pt = content_h_px * pt_per_px

    is_landscape = content_w_pt > content_h_pt * _LANDSCAPE_ASPECT_THRESHOLD
    page_w_pt, page_h_pt = A4_LANDSCAPE_PT if is_landscape else A4_PORTRAIT_PT

    available_w = page_w_pt - 2 * _MARGIN_PT
    available_h = page_h_pt - 2 * _MARGIN_PT

    # No 1.0 cap: always scale content to fill the available A4 area
    # (aspect ratio preserved -- whichever dimension is tighter, width or
    # height, determines the scale). A sparse page (e.g. a title page)
    # previously stayed at its small original size, surrounded by a large
    # white margin -- confirmed as an unwanted result on a real title page
    # (2026-09-13): the content should fill the page like the rest of the
    # book's pages, not sit tiny in the middle of mostly blank space.
    scale = min(available_w / content_w_pt, available_h / content_h_pt)

    scaled_w = content_w_pt * scale
    scaled_h = content_h_pt * scale
    offset_x = (page_w_pt - scaled_w) / 2
    offset_y = (page_h_pt - scaled_h) / 2

    return PageTransform(
        crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2,
        scale=scale, offset_x_pt=offset_x, offset_y_pt=offset_y,
        page_width_pt=page_w_pt, page_height_pt=page_h_pt, dpi=dpi,
    )
