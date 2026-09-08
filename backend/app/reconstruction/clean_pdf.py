"""
"Clean PDF" (spec section 17): the cleaned/preprocessed page images
assembled into a PDF, image-only — no text layer, no reconstruction. This
is the fastest export and a faithful visual copy of the cleaned scan.
"""
from __future__ import annotations

import cv2
import fitz  # PyMuPDF
import numpy as np

from app.utils.coordinates import page_size_pt


def add_clean_page(doc: "fitz.Document", image: np.ndarray, dpi: int) -> None:
    """Appends one page to an already-open fitz.Document. Used by the
    pipeline to build the clean PDF incrementally, one rendered page at a
    time, so a 500-page document never needs all page images in RAM at
    once (spec section 28)."""
    height_px, width_px = image.shape[:2]
    width_pt, height_pt = page_size_pt(width_px, height_px, dpi)
    page = doc.new_page(width=width_pt, height=height_pt)
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=encoded.tobytes())


def render_clean_pdf(pages: list[tuple[np.ndarray, int]]) -> bytes:
    """Convenience one-shot wrapper (small documents / tests). The pipeline
    itself uses `add_clean_page` incrementally instead."""
    doc = fitz.open()
    for image, dpi in pages:
        add_clean_page(doc, image, dpi)
    pdf_bytes = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return pdf_bytes
