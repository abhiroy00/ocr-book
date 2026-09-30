"""
"Clean PDF" (spec section 17): the cleaned/preprocessed page images
assembled into a PDF, image-only — no text layer, no reconstruction. This
is the fastest export and a faithful visual copy of the cleaned scan.

All pages are normalized to A4 size regardless of the original scanned page
dimensions.
"""
from __future__ import annotations

import cv2
import fitz  # PyMuPDF
import numpy as np

from app.utils.coordinates import page_size_pt
from app.reconstruction.page_transform import compute_page_transform


def add_clean_page(doc: "fitz.Document", image: np.ndarray, dpi: int, words: list = None) -> None:
    """Appends one page to an already-open fitz.Document. Used by the
    pipeline to build the clean PDF incrementally, one rendered page at a
    time, so a 500-page document never needs all page images in RAM at
    once (spec section 28).

    The page is normalized to A4 size with proper content-aware cropping.
    """
    height_px, width_px = image.shape[:2]

    # Get or compute words for content bbox calculation
    if words is None:
        words = []

    # Compute A4 normalization transform
    transform = compute_page_transform(width_px, height_px, words, dpi)

    # Create A4 page
    page = doc.new_page(width=transform.a4_width_pt, height=transform.a4_height_pt)

    # Crop and scale the image according to the transform
    cropped = image[transform.crop_y1 : transform.crop_y2, transform.crop_x1 : transform.crop_x2]
    ok, encoded = cv2.imencode(".png", cropped)
    if ok:
        # Scale the cropped content to fit the A4 page with margins
        scaled_w_pt = (transform.crop_x2 - transform.crop_x1) * (72.0 / dpi) * transform.scale
        scaled_h_pt = (transform.crop_y2 - transform.crop_y1) * (72.0 / dpi) * transform.scale
        placement = fitz.Rect(
            transform.offset_x_pt, transform.offset_y_pt,
            transform.offset_x_pt + scaled_w_pt, transform.offset_y_pt + scaled_h_pt,
        )
        page.insert_image(placement, stream=encoded.tobytes())


def render_clean_pdf(pages: list[tuple[np.ndarray, int, list]]) -> bytes:
    """Convenience one-shot wrapper (small documents / tests).

    Each entry is (image, dpi, words) where words are OCR word results
    used for A4 normalization.
    """
    doc = fitz.open()
    for image, dpi, words in pages:
        add_clean_page(doc, image, dpi, words)
    pdf_bytes = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return pdf_bytes