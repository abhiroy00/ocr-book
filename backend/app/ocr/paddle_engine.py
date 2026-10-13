"""
PaddleOCR provider — the default, offline, no-API-key OCR engine.

PaddleOCR's recognizer is per-language (a "lang" model bundle). With the
default PADDLE_OCR_LANGS=en,hi and OCR_HINDI_LINE_ENGINE=tesseract, a page
gets ONE English PaddleOCR pass, and only the lines the English model cannot
read go to Tesseract (hin+eng) -- see `_recognize_english_with_hindi_lines`
for the measurements behind this. Any other language set (or
OCR_HINDI_LINE_ENGINE=paddle) keeps the original behaviour: one full pass per
language over the same image, merged per region by confidence.
"""
from __future__ import annotations

import re
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
    loading is expensive, so this must happen once per worker process.

    `cpu_threads=1` is deliberate, not an oversight: PaddleOCR's own default
    is `cpu_threads=10` (it calls `config.set_cpu_math_library_num_threads`
    internally), which is only correct when a single PaddleOCR instance
    owns the whole machine. This engine now regularly runs inside one of
    several `app.workers.page_worker_pool` worker *processes* running at
    once -- with the library default, N pool workers each spun up their
    own internal 10-thread MKL/OpenBLAS pool, oversubscribing the CPU by
    up to Nx10 threads competing for the host's actual core count. This
    was confirmed as a real, severe production regression: with 2 pool
    workers at the library default, per-page OCR time degraded from ~4s to
    over 1150s (nearly 20 minutes) for a single page after a few pages,
    with PaddlePaddle's own internal scheduler logging "bvar is busy at
    sampling" -- a symptom of exactly this thread thrashing. Pinning each
    instance to 1 internal thread makes the pool's own process-level
    parallelism (one process per `OCR_WORKERS`) the sole source of
    concurrency, which is the correct place for it: no oversubscription is
    possible regardless of how many pool workers are configured.

    Revisited 2026-10-12 with measurements: real scanned books need ~7.8GB
    per worker, so on the 12GB host only ONE pool worker ever runs -- and at
    1 thread that left 13 of 14 cores idle. Thread count and MKLDNN are now
    settings (`OCR_PADDLE_THREADS`, `OCR_PADDLE_MKLDNN`). Measured on 8 real
    pages: MKLDNN + 4 threads made the English pass ~1.8x faster (3.6s ->
    2.0s/page) with identical output; 8 threads was slower and used more
    memory. Keep threads x OCR workers <= the cores you really have.
    """
    with _engine_lock:
        if lang not in _engine_cache:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise OCREngineUnavailableError(f"paddleocr package not installed: {exc}") from exc
            settings = get_settings()
            _engine_cache[lang] = PaddleOCR(
                use_angle_cls=True, lang=lang, show_log=False,
                cpu_threads=max(1, settings.ocr_paddle_threads), enable_mkldnn=settings.ocr_paddle_mkldnn,
            )
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

        if "en" in self.langs and "hi" in self.langs and get_settings().ocr_hindi_line_engine == "tesseract":
            return self._recognize_english_with_hindi_lines(image, page_number, dpi)

        all_words: list[OCRWordResult] = []
        for lang in self.langs:
            engine = _get_paddle_engine(_map_lang(lang))
            raw = engine.ocr(image, cls=True)
            words = _parse_paddle_result(raw, page_number, dpi, lang)
            all_words = _merge_by_iou(all_words, words)
        return self._finalize(all_words, page_number)

    def _recognize_english_with_hindi_lines(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult:
        """English PaddleOCR on the whole page, then Tesseract (hin+eng) on
        ONLY the lines the English model could not read. Replaces running a
        full Hindi Paddle pass on every page and merging by confidence.

        Why (measured 2026-10-12, real 402-page book + known-text pages):
          * the confidence merge let the Hindi model's output win on plain
            ENGLISH lines -- real pages came out as "t० the state", "s०
            distant", "Ieft Iargely" -- while the English model alone read
            the same lines correctly;
          * Paddle's Devanagari models (PP-OCRv4 and v3) broke conjuncts and
            dropped spaces: ~26% Hindi character error, ~6% of Hindi words
            exact; Tesseract `hin` on the same lines: 2.8% / 95%;
          * the full Hindi pass cost ~12s of a ~17s page; this path is ~3s.
        A Tesseract result replaces the English one only when it really is
        mostly Devanagari, so an English line can never be swapped for
        Latin guesses from the second engine."""
        engine = _get_paddle_engine("en")
        raw = engine.ocr(image, cls=True)
        lines = raw[0] if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) else raw
        words: list[OCRWordResult] = []
        for i, item in enumerate(lines or []):
            try:
                box_pts, (text, confidence) = item
            except (TypeError, ValueError):
                continue
            if _looks_unreadable_as_english(text, float(confidence)):
                hindi_text, hindi_conf = _tesseract_line(image, box_pts)
                if _is_mostly_devanagari(hindi_text):
                    text, confidence = hindi_text, hindi_conf
            if not text or not text.strip():
                continue
            polygon = Polygon.from_xy_list(box_pts)
            bbox = polygon.bbox()
            words.append(
                OCRWordResult(
                    text=text, confidence=float(confidence), bbox=bbox, polygon=polygon, page_number=page_number,
                    block_id=f"blk_{i:05d}", line_id=f"ln_{i:05d}", word_id=None,
                    language=detect_language(text) or "en", font_size_estimate=_estimate_font_size(bbox, dpi),
                )
            )
        return self._finalize(words, page_number)

    def _finalize(self, all_words: list[OCRWordResult], page_number: int) -> OCRPageResult:
        # Assign stable block/line ids grouped by reading order (top-to-bottom).
        all_words.sort(key=lambda w: (round(w.bbox.y1 / 10), w.bbox.x1))
        for idx, w in enumerate(all_words):
            w.block_id = f"blk_{idx:05d}"
            w.line_id = f"ln_{idx:05d}"

        mean_conf = float(np.mean([w.confidence for w in all_words])) if all_words else 0.0
        return OCRPageResult(page_number=page_number, provider=self.name, words=all_words, mean_confidence=mean_conf)


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _looks_unreadable_as_english(text: str, confidence: float) -> bool:
    """The English model is unsure, or produced the letter fragments it
    typically emits when it is shown Devanagari."""
    if confidence < get_settings().ocr_hindi_line_confidence:
        return True
    latin_words = _LATIN_WORD.findall(text)
    return bool(latin_words) and sum(len(w) for w in latin_words) / len(latin_words) < 2.0


def _is_mostly_devanagari(text: str) -> bool:
    chars = re.sub(r"\s", "", text or "")
    return bool(chars) and len(_DEVANAGARI.findall(chars)) >= 0.5 * len(chars)


def _tesseract_line(image: np.ndarray, box_pts) -> tuple[str, float]:
    """Single-line Tesseract (hin+eng) on the detected line's crop. Never
    raises: an unavailable/failing Tesseract just keeps the English result."""
    try:
        import pytesseract
        from paddleocr.tools.infer.utility import get_rotate_crop_image

        settings = get_settings()
        if settings.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
        crop = get_rotate_crop_image(image, np.array(box_pts, dtype=np.float32))
        data = pytesseract.image_to_data(crop, lang=settings.ocr_hindi_line_langs, config="--psm 7", output_type=pytesseract.Output.DICT)
        tokens = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w and w.strip() and float(c) >= 0]
        if not tokens:
            return "", 0.0
        return " ".join(w for w, _ in tokens), sum(c for _, c in tokens) / len(tokens) / 100.0
    except Exception:  # noqa: BLE001 - the English result stands
        return "", 0.0


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
