"""
Searchable PDF (spec section 17): the cleaned page image as a background,
with an invisible OCR text layer positioned exactly over the recognized
words — the classic "image + hidden text" searchable-scan technique. This
is the safety net when perfect visual reconstruction isn't possible: the
page still looks exactly like the (cleaned) scan, but is fully searchable
and copy-pasteable.
"""
from __future__ import annotations

import fitz  # PyMuPDF
import numpy as np

from app.reconstruction.fonts import resolve_body_font_path
from app.schemas.ocr import OCRWordResult
from app.utils.coordinates import page_size_pt, px_to_pt


def add_searchable_page(doc: "fitz.Document", image, dpi: int, words: list[OCRWordResult], font_path: str | None) -> None:
    """Appends one image+invisible-text-layer page to an already-open
    fitz.Document (incremental, memory-safe for large documents)."""
    import cv2

    height_px, width_px = image.shape[:2]
    width_pt, height_pt = page_size_pt(width_px, height_px, dpi)
    page = doc.new_page(width=width_pt, height=height_pt)

    ok, encoded = cv2.imencode(".png", image)
    if ok:
        page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=encoded.tobytes())

    for w in words:
        if not w.text.strip():
            continue
        rect = fitz.Rect(
            px_to_pt(w.bbox.x1, dpi), px_to_pt(w.bbox.y1, dpi),
            px_to_pt(w.bbox.x2, dpi), px_to_pt(w.bbox.y2, dpi),
        )
        fontsize = max(4.0, px_to_pt(w.bbox.height, dpi) * 0.85)
        kwargs = {"fontsize": fontsize, "render_mode": 3, "color": (0, 0, 0)}  # render_mode 3 = invisible
        if font_path:
            kwargs["fontfile"] = font_path
            kwargs["fontname"] = "F0"
        else:
            kwargs["fontname"] = "helv"
        try:
            page.insert_textbox(rect, w.text, **kwargs)
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
