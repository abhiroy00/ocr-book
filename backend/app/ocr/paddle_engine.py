"""
PaddleOCR provider — the default, offline, no-API-key OCR engine.

PaddleOCR's recognizer is per-language (a "lang" model bundle). For
Hindi+English mixed pages we run one detector+recognizer pass per configured
language (default `en` + `hi` from PADDLE_OCR_LANGS) over the *same* image
and merge results at the region level: for overlapping detections we keep
the higher-confidence recognition; non-overlapping detections from either
pass are both kept. This is a practical way to cover mixed-script pages
without a single joint multilingual recognizer.
"""
from __future__ import annotations

import threading

import numpy as np

from app.core.config import get_settings
from app.ocr.base import OCREngineUnavailableError, OCRProvider
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRPageResult, OCRWordResult
from app.utils.language import detect_language

_engine_cache: dict[str, "PaddleOCR"] = {}
_engine_lock = threading.Lock()


def _get_paddle_engine(lang: str):
    """Lazily instantiate (and cache) one PaddleOCR() per language — model
    loading is expensive, so this must happen once per worker process."""
    with _engine_lock:
        if lang not in _engine_cache:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise OCREngineUnavailableError(f"paddleocr package not installed: {exc}") from exc
            _engine_cache[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        return _engine_cache[lang]


class PaddleOCREngine(OCRProvider):
    name = "paddleocr"

    def __init__(self, langs: list[str] | None = None) -> None:
        settings = get_settings()
        self.langs = langs or settings.paddle_ocr_lang_list or ["en"]

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False

    def recognize_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult:
        if not self.is_available():
            raise OCREngineUnavailableError("PaddleOCR is not installed in this environment")

        all_words: list[OCRWordResult] = []
        for lang in self.langs:
            engine = _get_paddle_engine(_map_lang(lang))
            raw = engine.ocr(image, cls=True)
            words = _parse_paddle_result(raw, page_number, dpi, lang)
            all_words = _merge_by_iou(all_words, words)

        # Assign stable block/line ids grouped by reading order (top-to-bottom).
        all_words.sort(key=lambda w: (round(w.bbox.y1 / 10), w.bbox.x1))
        for idx, w in enumerate(all_words):
            w.block_id = f"blk_{idx:05d}"
            w.line_id = f"ln_{idx:05d}"

        mean_conf = float(np.mean([w.confidence for w in all_words])) if all_words else 0.0
        return OCRPageResult(page_number=page_number, provider=self.name, words=all_words, mean_confidence=mean_conf)


def _map_lang(lang: str) -> str:
    """Map our config language codes to PaddleOCR's model language codes."""
    return {"hi": "hi", "en": "en", "devanagari": "devanagari"}.get(lang, lang)


def _parse_paddle_result(raw, page_number: int, dpi: int, lang_hint: str) -> list[OCRWordResult]:
    words: list[OCRWordResult] = []
    if not raw:
        return words
    lines = raw[0] if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) else raw
    if lines is None:
        return words

    for i, item in enumerate(lines):
        try:
            box_pts, (text, confidence) = item
        except (TypeError, ValueError):
            continue
        if not text or not text.strip():
            continue
        polygon = Polygon.from_xy_list(box_pts)
        bbox = polygon.bbox()
        language = detect_language(text) or lang_hint
        font_size_estimate = _estimate_font_size(bbox, dpi)
        words.append(
            OCRWordResult(
                text=text,
                confidence=float(confidence),
                bbox=bbox,
                polygon=polygon,
                page_number=page_number,
                block_id=f"blk_{i:05d}",
                line_id=f"ln_{i:05d}",
                word_id=None,
                language=language,
                font_size_estimate=font_size_estimate,
            )
        )
    return words


def _estimate_font_size(bbox: BBox, dpi: int) -> float:
    """Rough pt font size from glyph bbox height at the render DPI."""
    return round((bbox.height / dpi) * 72.0, 1)


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(a: BBox, b: BBox) -> float:
    """Intersection area over the SMALLER of the two boxes' areas, rather
    than strict IoU. Different language passes often segment the same line
    of text into slightly different-sized boxes (different word/character
    grouping), which drags plain IoU down even when the two detections are
    really the same text — this containment-style ratio stays high in
    that case, which strict IoU (penalized by the size mismatch) misses."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    smaller_area = min(a.width * a.height, b.width * b.height)
    return inter / smaller_area if smaller_area > 0 else 0.0


def _merge_by_iou(existing: list[OCRWordResult], new: list[OCRWordResult], overlap_threshold: float = 0.5) -> list[OCRWordResult]:
    """Keeps existing detections; adds new ones that don't overlap an
    existing higher-or-equal-confidence detection, and replaces overlapping
    lower-confidence ones."""
    merged = list(existing)
    for cand in new:
        replaced = False
        for idx, cur in enumerate(merged):
            if _overlap_ratio(cur.bbox, cand.bbox) >= overlap_threshold:
                if cand.confidence > cur.confidence:
                    merged[idx] = cand
                replaced = True
                break
        if not replaced:
            merged.append(cand)
    return merged
