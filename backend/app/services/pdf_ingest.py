"""
PDF/image ingestion: page-by-page rendering to numpy images at a configured
DPI (spec section 6). Pages are yielded one at a time — a 500-page PDF is
never fully materialized in memory (acceptance criterion #24 / section 28).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


@dataclass
class RenderedPage:
    page_number: int  # 1-indexed
    image: np.ndarray  # BGR uint8
    width: int
    height: int
    dpi: int
    rotation: float


def get_page_count(file_bytes: bytes, is_pdf: bool) -> int:
    if not is_pdf:
        return 1
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return doc.page_count


def iter_render_pages(file_bytes: bytes, is_pdf: bool, dpi: int) -> Iterator[RenderedPage]:
    """Streams rendered pages. For images, yields a single page (converted to
    the standard BGR array PaddleOCR/OpenCV expect)."""
    if not is_pdf:
        yield _render_image_bytes(file_bytes, dpi)
        return

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.is_encrypted:
            doc.authenticate("")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            rotation = float(page.rotation or 0)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = _pixmap_to_bgr(pix)
            yield RenderedPage(
                page_number=i + 1,
                image=image,
                width=image.shape[1],
                height=image.shape[0],
                dpi=dpi,
                rotation=rotation,
            )
            # Free page resources promptly for large documents.
            page = None
    finally:
        doc.close()


def _pixmap_to_bgr(pix: "fitz.Pixmap") -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _render_image_bytes(file_bytes: bytes, dpi: int) -> RenderedPage:
    with Image.open(__import__("io").BytesIO(file_bytes)) as pil_img:
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return RenderedPage(page_number=1, image=image, width=image.shape[1], height=image.shape[0], dpi=dpi, rotation=0.0)
