"""
Lightweight script/language detection from recognized text — used to tag
OCR results and to route which PaddleOCR/Tesseract language model produced
the best reading for a region. This never translates or rewrites text
(spec section 10) — it only classifies script.
"""
from __future__ import annotations

import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9०-९]")


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "und"
    has_deva = bool(_DEVANAGARI_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_deva and has_latin:
        return "hi-en"
    if has_deva:
        return "hi"
    if has_latin:
        return "en"
    if _DIGIT_RE.search(text):
        return "num"
    return "und"
