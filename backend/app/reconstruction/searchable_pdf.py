"""
Searchable PDF: a content-aware-cropped, A4-normalized page image, with an
invisible OCR text layer positioned exactly over the recognized words --
the classic "image + hidden text" searchable-scan technique. This is the
safety net when perfect visual reconstruction isn't possible: the page
still looks like the (cleaned, now A4-normalized) scan, but is fully
searchable and copy-pasteable.

A4 normalization (crop -> scale -> center) is pure, deterministic geometry
computed once per page via `app.utils.page_normalize.compute_page_transform`
and applied identically to the image and to every OCR word's bounding box
-- this is what keeps the invisible text aligned with the now-repositioned
visible content, and it never re-runs OCR (see that module's docstring for
why the content boundary comes from OCR word boxes, not pixel analysis).
"""
from __future__ import annotations

import os

import fitz  # PyMuPDF
import numpy as np

from app.reconstruction.fonts import resolve_body_font_path, resolve_devanagari_fallback_font_paths
from app.models.enums import LayoutBlockType
from app.schemas.document_json import DocumentBlockJSON
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.utils.page_normalize import compute_page_transform


# Block types that carry no searchable text of their own (pure graphics or
# rule lines). Everything else with text -- paragraphs, headings, table
# cells -- contributes to the fallback text layer below.
_NON_TEXT_BLOCK_TYPES = frozenset({
    LayoutBlockType.IMAGE,
    LayoutBlockType.CHART,
    LayoutBlockType.SIGNATURE,
    LayoutBlockType.STAMP,
    LayoutBlockType.HANDWRITTEN,
    LayoutBlockType.HLINE,
    LayoutBlockType.VLINE,
})


def _bbox_polygon(bbox: BBox) -> Polygon:
    return Polygon.from_xy_list([[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]])


def fallback_words_from_blocks(blocks: list[DocumentBlockJSON], page_number: int) -> list[OCRWordResult]:
    """Word-level pseudo-results from Document JSON text/table blocks, used
    ONLY when a page's word-level OCR rows are missing. It can never
    duplicate the primary text layer (callers use it solely when `words`
    is empty) -- it guarantees a page with visible text is never exported
    as an image-only, unsearchable page just because its word rows are
    unavailable. Table cells are kept per-cell so numbers stay searchable.
    Block bboxes are already in the processed-image pixel frame (see
    `build_page_json`), exactly the coordinate space `add_searchable_page`
    expects, so no conversion is needed.

    Block/cell text is split into whitespace-separated tokens sharing the
    block's bbox (position-approximate, like the rest of this invisible
    layer) rather than inserted as one multi-word string: some fallback
    fonts encode the space character as a non-breaking space, which would
    silently break phrase search, while separate tokens stay searchable
    under every font."""
    out: list[OCRWordResult] = []
    idx = 0

    def _add(text: str, bbox: BBox, confidence: float, block_id: str, line_id: str) -> None:
        nonlocal idx
        for token in text.split():
            out.append(OCRWordResult(
                text=token, confidence=confidence, bbox=bbox,
                polygon=_bbox_polygon(bbox), page_number=page_number,
                block_id=f"fb_{block_id}_{idx}", line_id=line_id, language="und",
            ))
            idx += 1

    for b in blocks:
        if b.type in _NON_TEXT_BLOCK_TYPES:
            continue
        if b.type == LayoutBlockType.TABLE and b.table:
            for row in b.table.rows:
                for cell in row.cells:
                    text = (cell.text or "").strip()
                    if text:
                        _add(text, cell.bbox, cell.confidence or 0.0, b.id, f"fb_{b.id}_r{cell.row}")
            continue
        text = " ".join(run.text for run in (b.content or []) if run.text and run.text.strip()).strip()
        if text:
            _add(text, b.bbox, b.confidence, b.id, f"fb_{b.id}")
    return out


def _font_covers_text(font: "fitz.Font", text: str) -> bool:
    """True when `font` has a glyph for every non-space character in
    `text`. A single bad glyph must never fail the page, so any lookup
    error reads as "not covered" and the caller simply tries the next
    candidate font."""
    try:
        return all(font.has_glyph(ord(ch)) for ch in text if not ch.isspace())
    except Exception:  # noqa: BLE001
        return False


def _resolve_text_layer_fonts(primary_path: str | None) -> list[tuple[str, str | None, "fitz.Font"]]:
    """(resource name, fontfile path or None for base-14, measuring Font)
    for every font file usable by this page's invisible text layer, in
    priority order: the pipeline's body font first, then the bundled/OS
    Devanagari fallbacks. Broken files are skipped, never fatal -- worst
    case the page falls back to plain Helvetica, exactly today's behavior
    when no Unicode font is available.

    Each distinct file gets its own resource name ("F0", "F1", ...) because
    PyMuPDF reuses an already-registered name's *first* file and would
    otherwise silently draw a different font than the one measured."""
    candidates: list[str] = []
    if primary_path and os.path.exists(primary_path):
        candidates.append(primary_path)
    for fallback in resolve_devanagari_fallback_font_paths():
        if fallback not in candidates:
            candidates.append(fallback)

    fonts: list[tuple[str, str | None, "fitz.Font"]] = []
    for i, path in enumerate(candidates):
        try:
            fonts.append((f"F{i}", path, fitz.Font(fontfile=path)))
        except Exception:  # noqa: BLE001 - a corrupt font file must not fail the page
            continue
    if not fonts:
        fonts.append(("helv", None, fitz.Font(fontname="helv")))
    return fonts


def add_searchable_page(doc: "fitz.Document", image, dpi: int, words: list[OCRWordResult], font_path: str | None) -> None:
    """Appends one A4-normalized, image+invisible-text-layer page to an
    already-open fitz.Document (incremental, memory-safe for large
    documents).

    Uses `page.insert_text()` at each word's baseline point, NOT
    `insert_textbox()` (a fitting/word-wrapping container). This was a
    real, confirmed bug: real OCR word boxes are tight around the glyphs,
    and `insert_textbox`'s line-height reservation needs slightly more
    vertical room than a box sized to `bbox.height * 0.85`, well within
    what a human would call "fits" -- confirmed on real production OCR
    data where `insert_textbox` returned a small negative (did-not-fit)
    remainder and silently inserted NOTHING for every single word on the
    page (37/37), producing a "searchable" PDF with a completely empty
    text layer despite good underlying OCR. `insert_text` has no such
    fit-rejection: it just draws at a point, which is exactly what an
    invisible, position-approximate selection/search layer needs -- unlike
    visible text, there is no requirement that it look right, only that
    each word's invisible glyphs sit close enough to their word for
    click-to-select and Ctrl+F highlighting to land in the right place.
    """
    import cv2

    height_px, width_px = image.shape[:2]
    transform = compute_page_transform(width_px, height_px, words, dpi)
    page = doc.new_page(width=transform.page_width_pt, height=transform.page_height_pt)

    cropped = image[transform.crop_y1 : transform.crop_y2, transform.crop_x1 : transform.crop_x2]
    ok, encoded = cv2.imencode(".png", cropped)
    if ok:
        scaled_w_pt = (transform.crop_x2 - transform.crop_x1) * (72.0 / dpi) * transform.scale
        scaled_h_pt = (transform.crop_y2 - transform.crop_y1) * (72.0 / dpi) * transform.scale
        placement = fitz.Rect(
            transform.offset_x_pt, transform.offset_y_pt,
            transform.offset_x_pt + scaled_w_pt, transform.offset_y_pt + scaled_h_pt,
        )
        page.insert_image(placement, stream=encoded.tobytes())

    # `fitz.Font(fontfile=...)` is the one API that measures text width for
    # a *custom* font correctly (the module-level `fitz.get_text_length`
    # only knows the built-in fonts) -- needed so a long Devanagari/English
    # word's measured width isn't silently wrong, which would make the
    # overflow-shrink below either fire when it shouldn't or not fire when
    # it should. Loaded once per page, not per word.
    #
    # The invisible layer is only ever searched/selected, never seen -- so
    # the font for a word is chosen by *coverage*, not looks. The single
    # body font is not guaranteed to encode every script on the page (it
    # is the Latin Noto Sans whenever that file is present, and plain
    # Helvetica -- Latin-only -- whenever the bundled fonts are absent).
    # A Hindi/Devanagari word drawn with a font that has no Devanagari
    # glyphs is inserted as unmapped glyphs and extracts as garbage, so
    # Ctrl+F can never find it despite correct underlying OCR. Per word,
    # use the first available font that actually covers every character in
    # that word: the body font wins whenever it covers the word, so
    # Latin/numbers (and Devanagari when the bundled variable Noto Sans,
    # which covers both scripts, is present) render exactly as before.
    # Visible output is byte-identical either way -- this text is invisible.
    text_fonts = _resolve_text_layer_fonts(font_path)

    for w in words:
        if not w.text.strip():
            continue
        transformed = transform.transform_bbox(w.bbox)
        box_width = transformed.width
        box_height = transformed.height
        if box_width <= 0 or box_height <= 0:
            continue  # fell entirely outside the crop -- nothing to place
        x1 = transformed.x1
        y2 = transformed.y2  # baseline approximated at the box bottom
        fontsize = max(3.0, box_height * 0.85)
        fontname, fontfile, measure_font = text_fonts[0]
        if not _font_covers_text(measure_font, w.text):
            # The primary font cannot encode this word -- try the
            # Devanagari fallbacks before giving up and drawing it with an
            # unsuitable font (which is what made such words unsearchable).
            for candidate in text_fonts[1:]:
                if _font_covers_text(candidate[2], w.text):
                    fontname, fontfile, measure_font = candidate
                    break
        kwargs = {"fontsize": fontsize, "render_mode": 3, "color": (0, 0, 0)}  # render_mode 3 = invisible
        kwargs["fontname"] = fontname
        if fontfile:
            kwargs["fontfile"] = fontfile
        try:
            # insert_text draws at nominal fontsize with no box constraint,
            # so a long word in a narrow box can overshoot noticeably --
            # measure first (draws nothing) and squeeze the fontsize down
            # before the one real draw call, so the invisible glyphs stay
            # roughly over their word rather than spilling into the next
            # one and confusing text selection/search-result highlighting.
            # Cosmetically irrelevant since this text is never seen, only
            # searched/selected.
            measured_width = measure_font.text_length(w.text, fontsize=fontsize)
            if box_width > 0 and measured_width > box_width * 1.15:
                fontsize *= max(0.3, box_width / measured_width)
                kwargs["fontsize"] = fontsize
            page.insert_text(fitz.Point(x1, y2), w.text, **kwargs)
        except Exception:  # noqa: BLE001 - a single bad glyph must not fail the page
            continue


def render_searchable_pdf(pages: list[tuple[np.ndarray, int, list[OCRWordResult]]]) -> bytes:
    """Convenience one-shot wrapper (small documents / tests). The pipeline
    itself uses `add_searchable_page` incrementally instead."""
    doc = fitz.open()
    font_path = resolve_body_font_path()
    for image, dpi, words in pages:
        add_searchable_page(doc, image, dpi, words, font_path)
    pdf_bytes = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return pdf_bytes
