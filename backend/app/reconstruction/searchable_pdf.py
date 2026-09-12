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

import fitz  # PyMuPDF
import numpy as np

from app.reconstruction.fonts import resolve_body_font_path
from app.schemas.ocr import OCRWordResult
from app.utils.page_normalize import compute_page_transform


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
    measure_font = fitz.Font(fontfile=font_path) if font_path else fitz.Font(fontname="helv")

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
        kwargs = {"fontsize": fontsize, "render_mode": 3, "color": (0, 0, 0)}  # render_mode 3 = invisible
        if font_path:
            kwargs["fontfile"] = font_path
            kwargs["fontname"] = "F0"
        else:
            kwargs["fontname"] = "helv"
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
