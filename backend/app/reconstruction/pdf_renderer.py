"""
PDF reconstruction (spec section 16): renders a DocumentJSON back into a
PDF using ABSOLUTE positioning derived from each block's normalized bbox —
not a flowing/auto-reflow layout — so the output visually matches the
original scan as closely as technically possible.

Text blocks and vector tables are placed as real, selectable/editable PDF
text and vector graphics (not rasterized), per spec section 16/38. Blocks
that cannot be reliably reconstructed (signature/stamp/handwritten/complex
graphics) are placed as embedded images cropped from the cleaned page scan.
"""
from __future__ import annotations

import fitz  # PyMuPDF

from app.core.logging import get_logger
from app.models.enums import LayoutBlockType, TextAlign
from app.reconstruction.fonts import resolve_body_font_path
from app.schemas.document_json import DocumentBlockJSON, PageJSON
from app.services.storage import get_storage
from app.utils.coordinates import page_size_pt, px_to_pt

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


def render_document_pdf(pages: list[PageJSON]) -> bytes:
    doc = fitz.open()
    body_font = resolve_body_font_path(bold=False)
    bold_font = resolve_body_font_path(bold=True) or body_font

    for page_json in pages:
        width_pt, height_pt = page_size_pt(page_json.page_width, page_json.page_height, page_json.dpi)
        page = doc.new_page(width=width_pt, height=height_pt)

        for block in page_json.sorted_blocks():
            try:
                _render_block(doc, page, block, page_json, body_font, bold_font)
            except Exception as exc:  # noqa: BLE001 - one bad block must not fail the whole page
                logger.warning("pdf_block_render_failed", block_id=block.id, block_type=block.type.value, error=str(exc))

    pdf_bytes = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return pdf_bytes


def _render_block(doc, page, block: DocumentBlockJSON, page_json: PageJSON, body_font, bold_font) -> None:
    rect = _bbox_to_rect(block, page_json)

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


def _render_text(page, rect: "fitz.Rect", block: DocumentBlockJSON, body_font, bold_font) -> None:
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


def _render_table(page, block: DocumentBlockJSON, page_json: PageJSON, body_font) -> None:
    table = block.table
    dpi = page_json.dpi

    for row in table.rows:
        for cell in row.cells:
            cell_rect = fitz.Rect(
                px_to_pt(cell.bbox.x1, dpi), px_to_pt(cell.bbox.y1, dpi),
                px_to_pt(cell.bbox.x2, dpi), px_to_pt(cell.bbox.y2, dpi),
            )
            page.draw_rect(cell_rect, color=(0, 0, 0), width=0.6)
            if cell.text.strip():
                align = _ALIGN_MAP.get(cell.align_h, fitz.TEXT_ALIGN_LEFT)
                kwargs = {"fontsize": 9.0, "align": align, "color": (0, 0, 0)}
                if body_font:
                    kwargs["fontfile"] = body_font
                    kwargs["fontname"] = "F0"
                else:
                    kwargs["fontname"] = "helv"
                inset = cell_rect + (2, 2, -2, -2)
                page.insert_textbox(inset, cell.text, **kwargs)


def _insert_image(doc, page, rect: "fitz.Rect", image_ref: str) -> None:
    storage = get_storage()
    if not storage.exists(image_ref):
        return
    data = storage.read(image_ref)
    page.insert_image(rect, stream=data)


def _bbox_to_rect(block: DocumentBlockJSON, page_json: PageJSON) -> "fitz.Rect":
    dpi = page_json.dpi
    return fitz.Rect(
        px_to_pt(block.bbox.x1, dpi), px_to_pt(block.bbox.y1, dpi),
        px_to_pt(block.bbox.x2, dpi), px_to_pt(block.bbox.y2, dpi),
    )


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
