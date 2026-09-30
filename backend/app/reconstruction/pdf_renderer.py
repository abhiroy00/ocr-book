"""
PDF reconstruction (spec section 16): renders a DocumentJSON back into a
PDF using ABSOLUTE positioning derived from each block's normalized bbox —
not a flowing/auto-reflow layout — so the output visually matches the
original scan as closely as technically possible.

All pages are normalized to A4 size (595.2756 × 841.8898 points for portrait,
841.8898 × 595.2756 for landscape) regardless of the original scanned page
size. Block positions are transformed from original coordinates to the A4
coordinate system using the same transform applied to the page content.
"""
from __future__ import annotations

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import LayoutBlockType, TextAlign
from app.reconstruction.fonts import resolve_body_font_path
from app.reconstruction.page_transform import (
    compute_page_transform,
    transform_block_to_a4,
    A4_PORTRAIT_PT,
    A4_LANDSCAPE_PT,
)
from app.schemas.document_json import DocumentBlockJSON, PageJSON
from app.services.storage import get_storage
from app.utils.coordinates import px_to_pt

logger = get_logger(__name__)

_ALIGN_MAP = {
    TextAlign.LEFT: fitz.TEXT_ALIGN_LEFT,
    TextAlign.CENTER: fitz.TEXT_ALIGN_CENTER,
    TextAlign.RIGHT: fitz.TEXT_ALIGN_RIGHT,
}

_IMAGE_BLOCK_TYPES = {
    LayoutBlockType.IMAGE,
    LayoutBlockType.CHART,
    LayoutBlockType.SIGNATURE,
    LayoutBlockType.STAMP,
    LayoutBlockType.HANDWRITTEN,
}


def render_document_pdf(pages: list[PageJSON], preserve_original: bool | None = None) -> bytes:
    """Render all pages as PDF.

    If preserve_original is True (default), output pages preserve the
    original scanned page dimensions (A4, Letter, Legal, custom sizes).
    If False, output pages are normalized to A4 size.

    Every page is normalized to A4 size (595.2756 × 841.8898 points for
    portrait, 841.8898 × 595.2756 for landscape) regardless of the original
    scanned page dimensions. Block positions are transformed from original
    coordinates to the A4 coordinate system so that content is properly placed
    regardless of the original scan dimensions.
    """
    settings = get_settings()
    if preserve_original is None:
        preserve_original = settings.preserve_original_page_size

    doc = fitz.open()
    body_font = resolve_body_font_path(bold=False)
    bold_font = resolve_body_font_path(bold=True) or body_font

    for page_json in pages:
        if preserve_original:
            # Preserve original page dimensions - do NOT normalize to A4
            a4_width_pt, a4_height_pt = page_json.page_width, page_json.page_height
            page = doc.new_page(width=a4_width_pt, height=a4_height_pt)
            # Render blocks using original coordinates
            for block in page_json.sorted_blocks():
                try:
                    # Use original bbox (not A4-transformed), preserve original mode
                    _render_block(doc, page, block, page_json, body_font, bold_font,
                                  block.bbox, preserve_original=True)
                except Exception as exc:  # noqa: BLE001 - one bad block must not fail the whole page
                    logger.warning(
                        "pdf_block_render_failed",
                        block_id=block.id,
                        block_type=block.type.value,
                        error=str(exc),
                    )
        else:
            # Normalize to A4 size (existing behavior)
            # Compute A4 normalization transform for this page
            transform = compute_page_transform(
                page_json.page_width,
                page_json.page_height,
                getattr(page_json, "words", []) or [],
                page_json.dpi,
            )

            # Determine page orientation based on content aspect ratio
            is_landscape = transform.is_landscape
            a4_width, a4_height = A4_LANDSCAPE_PT if is_landscape else A4_PORTRAIT_PT
            page = doc.new_page(width=a4_width, height=a4_height)

            for block in page_json.sorted_blocks():
                try:
                    # Transform block bbox from original to A4 coordinates,
                    # normalize to A4 mode (existing behavior)
                    a4_bbox = transform_block_to_a4(block, page_json)

                    _render_block(doc, page, block, page_json, body_font, bold_font, a4_bbox,
                                  preserve_original=False)
                except Exception as exc:  # noqa: BLE001 - one bad block must not fail the whole page
                    logger.warning(
                        "pdf_block_render_failed",
                        block_id=block.id,
                        block_type=block.type.value,
                        error=str(exc),
                    )

    pdf_bytes = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return pdf_bytes


def _bbox_to_rect_from_pt(a4_bbox: BBox, dpi: int, page_width_px: int, page_height_px: int) -> "fitz.Rect":
    """Convert a bounding box from pixel coordinates at given DPI to PDF points,
    and create a fitz.Rect. This handles both A4-transformed and original coordinate systems.
    """
    # Convert pixel bbox to PDF points
    pt_per_px = 72.0 / dpi
    x1 = a4_bbox.x1 * pt_per_px
    y1 = a4_bbox.y1 * pt_per_px
    x2 = a4_bbox.x2 * pt_per_px
    y2 = a4_bbox.y2 * pt_per_px
    return fitz.Rect(x1, y1, x2, y2)


def _render_block(doc, page, block: DocumentBlockJSON, page_json: PageJSON,
                  body_font, bold_font, a4_bbox, preserve_original: bool = False) -> None:
    """Render a single block on a page using the provided bbox.

    If preserve_original is True, a4_bbox is in original pixel coordinates
    (converted to PDF points internally). If False, a4_bbox is already in
    A4 PDF point coordinates (existing behavior).
    """
    if preserve_original:
        rect = _bbox_to_rect_from_pt(a4_bbox, page_json.dpi, page_json.page_width, page_json.page_height)
    else:
        rect = _bbox_to_rect_from_a4(a4_bbox, page_json)

    if block.type == LayoutBlockType.TABLE and block.table:
        _render_table(page, block, page_json, body_font)
        return

    if block.type in (LayoutBlockType.HLINE, LayoutBlockType.VLINE):
        mid_y = (rect.y0 + rect.y1) / 2
        mid_x = (rect.x0 + rect.x1) / 2
        if block.type == LayoutBlockType.HLINE:
            start, end = fitz.Point(rect.x0, mid_y), fitz.Point(rect.x1, mid_y)
        else:
            start, end = fitz.Point(mid_x, rect.y0), fitz.Point(mid_x, rect.y1)
        page.draw_line(start, end, color=(0, 0, 0), width=0.75)
        return

    if block.type in _IMAGE_BLOCK_TYPES:
        if block.image_ref:
            _insert_image(doc, page, rect, block.image_ref)
        return

    _render_text(page, rect, block, body_font, bold_font)


def _bbox_to_rect_from_a4(a4_bbox: BBox, page_json: PageJSON) -> "fitz.Rect":
    """Convert an A4-transformed bbox to a fitz.Rect.

    The a4_bbox is already in PDF point coordinates (A4 coordinate system),
    so we just directly use it.
    """
    return fitz.Rect(a4_bbox.x1, a4_bbox.y1, a4_bbox.x2, a4_bbox.y2)


def _render_text(page, rect: "fitz.Rect", block: DocumentBlockJSON,
                 body_font, bold_font) -> None:
    text = "\n".join(run.text for run in block.content) if block.content else ""
    if not text.strip():
        return

    is_bold = block.type in (LayoutBlockType.TITLE, LayoutBlockType.HEADING)
    fontsize = float(block.style.get("font_size") or _default_font_size(block.type))
    align_name = (block.style.get("alignment") or "LEFT").upper()
    align = _ALIGN_MAP.get(TextAlign(align_name) if align_name in TextAlign.__members__ else TextAlign.LEFT, fitz.TEXT_ALIGN_LEFT)

    fontfile = bold_font if is_bold else body_font
    kwargs = {"fontsize": fontsize, "align": align, "color": (0, 0, 0)}
    if fontfile:
        kwargs["fontfile"] = fontfile
        kwargs["fontname"] = "F0"
    else:
        kwargs["fontname"] = "helv"

    remainder = page.insert_textbox(rect, text, **kwargs)
    if remainder < 0:
        # Text didn't fit the box at this size — shrink until it fits or we
        # hit a sane minimum, rather than silently truncating content.
        size = fontsize
        while remainder < 0 and size > 5:
            size -= 0.5
            kwargs["fontsize"] = size
            remainder = page.insert_textbox(rect, text, **kwargs)

    if remainder < 0:
        # Still doesn't fit even at the smallest legible size -- insert
        # anyway (unpadded, at the floor size) rather than silently dropping
        # real content; a hairline overflow is a far smaller defect than losing
        # the data entirely.
        kwargs["fontsize"] = 2.5
        page.insert_textbox(rect, text, **kwargs)


def _default_font_size(block_type: LayoutBlockType) -> float:
    return {
        LayoutBlockType.TITLE: 18.0,
        LayoutBlockType.HEADING: 14.0,
        LayoutBlockType.SUBHEADING: 12.0,
        LayoutBlockType.PAGE_NUMBER: 9.0,
        LayoutBlockType.FOOTNOTE: 8.0,
        LayoutBlockType.HEADER: 9.0,
        LayoutBlockType.FOOTER: 9.0,
    }.get(block_type, 10.0)


def _render_table(page, block: DocumentBlockJSON, page_json: PageJSON, body_font) -> None:
    """Draws each cell's border, then its text sized to actually fit.

    Real-world dense statistical tables (the kind this project's primary
    test document is full of) routinely have row heights well under the
    previous fixed 9pt -- confirmed on real data: several rows only 2.7-
    10.9pt tall after the fixed 2pt inset. `insert_textbox` does not raise
    when text can't fit vertically at the given size; it silently inserts
    nothing and returns a negative remainder. Because that return value was
    never checked here (unlike _render_text, which already retries at a
    smaller size), every cell in a table with any short row silently lost
    its text in the exported PDF -- while its (now text-less) border
    rectangle was still drawn unconditionally just above, since that call
    doesn't depend on the text fitting. This produced exactly the
    "empty grid of boxes" defect confirmed by rendering a real exported
    page and reading it back with `get_text()`.
    """
    table = block.table
    dpi = page_json.dpi

    for row in table.rows:
        for cell in row.cells:
            # Transform cell bbox to A4 coordinates
            a4_cell_bbox = _transform_bbox_to_a4(cell.bbox, page_json.dpi, page_json.page_width, page_json.page_height)
            cell_rect = fitz.Rect(
                a4_cell_bbox.x1, a4_cell_bbox.y1, a4_cell_bbox.x2, a4_cell_bbox.y2,
            )
            page.draw_rect(cell_rect, color=(0, 0, 0), width=0.6)
            if not cell.text.strip():
                continue

            align = _ALIGN_MAP.get(cell.align_h, fitz.TEXT_ALIGN_LEFT)
            kwargs = {"align": align, "color": (0, 0, 0)}
            if body_font:
                kwargs["fontfile"] = body_font
                kwargs["fontname"] = "F0"
            else:
                kwargs["fontname"] = "helv"

            # Scale font based on cell height in A4 coordinates
            pad = min(2.0, max(0.0, cell_rect.height * 0.15))
            inset = cell_rect + (pad, pad, -pad, -pad)
            if inset.width <= 0 or inset.height <= 0:
                inset = cell_rect

            fontsize = min(9.0, max(3.0, inset.height * 0.85))
            remainder = -1.0
            while remainder < 0 and fontsize > 2.5:
                kwargs["fontsize"] = fontsize
                remainder = page.insert_textbox(inset, cell.text, **kwargs)
                fontsize -= 0.5

            if remainder < 0:
                # Still doesn't fit even at the smallest legible size --
                # insert anyway (unpadded, at the floor size) rather than
                # silently dropping real content; a hairline overflow is a
                # far smaller defect than losing the data entirely.
                kwargs["fontsize"] = 2.5
                page.insert_textbox(cell_rect, cell.text, **kwargs)


def _transform_bbox_to_a4(cell_bbox, dpi, page_width, page_height):
    """Transform a cell bbox from original pixel coordinates to A4 PDF points."""
    # Calculate scale factor from original to A4
    pt_per_px = 72.0 / dpi
    
    # Simple proportion: original width/height -> A4 width/height
    # This is a placeholder - in a full implementation, we'd use the same
    # compute_page_transform logic as for other blocks
    # For now, just convert basic px to pt
    x1 = px_to_pt(cell_bbox.x1, dpi)
    y1 = px_to_pt(cell_bbox.y1, dpi)
    x2 = px_to_pt(cell_bbox.x2, dpi)
    y2 = px_to_pt(cell_bbox.y2, dpi)
    
    return type('BBox', (), {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})()


def _insert_image(doc, page, rect: "fitz.Rect", image_ref: str) -> None:
    storage = get_storage()
    if not storage.exists(image_ref):
        return
    data = storage.read(image_ref)
    page.insert_image(rect, stream=data)