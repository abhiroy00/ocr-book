"""
OCRProvider interface (spec section 8). Every engine — PaddleOCR, Tesseract,
NVIDIA VLM, Ollama — implements this and returns the same geometry-carrying
result contract (`OCRPageResult` / `OCRWordResult`), so nothing downstream
(layout, table, reconstruction) needs to know which engine produced it.

The application MUST work locally with PaddleOCR alone, with zero external
API keys. NVIDIA/Ollama are optional enhancers selected explicitly.
"""
from __future__ import annotations

import abc
import uuid

import numpy as np

from app.schemas.ocr import OCRPageResult


class OCREngineUnavailableError(RuntimeError):
    """Raised when a provider cannot run (missing credentials/model/binary).
    Callers (the pipeline) must catch this and fall back rather than crash
    the whole job — see spec section 29."""


class OCRProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def is_available(self) -> bool: ...

    @abc.abstractmethod
    def recognize_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult: ...

    def new_block_id(self) -> str:
        return f"blk_{uuid.uuid4().hex[:10]}"
