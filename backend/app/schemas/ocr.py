from __future__ import annotations

from pydantic import BaseModel

from app.schemas.geometry import BBox, Polygon


class OCRWordResult(BaseModel):
    """
    One OCR recognition unit as emitted by an OCRProvider — always carries
    geometry, never just text. This is the contract every engine
    (PaddleOCR/Tesseract/NVIDIA/Ollama) must normalize its output into.
    """

    text: str
    confidence: float
    bbox: BBox
    polygon: Polygon
    page_number: int
    block_id: str
    line_id: str
    word_id: str | None = None
    language: str = "und"
    font_size_estimate: float | None = None


class OCRPageResult(BaseModel):
    page_number: int
    provider: str
    words: list[OCRWordResult]
    mean_confidence: float
