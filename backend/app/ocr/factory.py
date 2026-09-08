"""
OCRProvider factory with automatic fallback (spec section 29: never let the
whole pipeline depend on one AI provider).

Resolution order when a specific provider is requested but unavailable:
    requested -> paddleocr -> tesseract
PaddleOCR is always the final guaranteed-available fallback in a standard
deployment (it ships in requirements.txt and needs no external service).
"""
from __future__ import annotations

from app.core.config import OCRProviderName, get_settings
from app.core.logging import get_logger
from app.ocr.base import OCREngineUnavailableError, OCRProvider
from app.ocr.nvidia_engine import NvidiaVLMEngine
from app.ocr.ollama_engine import OllamaEngine
from app.ocr.paddle_engine import PaddleOCREngine
from app.ocr.tesseract_engine import TesseractEngine

logger = get_logger(__name__)

_REGISTRY: dict[OCRProviderName, type[OCRProvider]] = {
    OCRProviderName.PADDLEOCR: PaddleOCREngine,
    OCRProviderName.TESSERACT: TesseractEngine,
    OCRProviderName.NVIDIA: NvidiaVLMEngine,
    OCRProviderName.OLLAMA: OllamaEngine,
}


def get_ocr_provider(requested: OCRProviderName | str | None = None) -> OCRProvider:
    settings = get_settings()
    requested_enum = OCRProviderName(requested) if requested else settings.ocr_provider

    tried: list[str] = []
    for candidate in _fallback_chain(requested_enum):
        tried.append(candidate.value)
        engine = _REGISTRY[candidate]()
        if engine.is_available():
            if candidate != requested_enum:
                logger.warning("ocr_provider_fallback", requested=requested_enum.value, using=candidate.value)
            return engine

    raise OCREngineUnavailableError(f"No OCR provider available. Tried: {tried}")


def _fallback_chain(requested: OCRProviderName) -> list[OCRProviderName]:
    chain = [requested]
    if OCRProviderName.PADDLEOCR not in chain:
        chain.append(OCRProviderName.PADDLEOCR)
    if OCRProviderName.TESSERACT not in chain:
        chain.append(OCRProviderName.TESSERACT)
    return chain
