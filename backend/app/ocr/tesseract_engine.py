"""
Tesseract fallback provider (spec section 8). Used when PaddleOCR is
unavailable/fails, or when explicitly selected. Requires the `tesseract`
binary (+ `hin`/`eng` traineddata) on PATH or at TESSERACT_PATH.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.ocr.base import OCREngineUnavailableError, OCRProvider
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRPageResult, OCRWordResult
from app.utils.language import detect_language


class TesseractEngine(OCRProvider):
    name = "tesseract"

    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        try:
            import pytesseract

            if self.settings.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_path
            pytesseract.get_tesseract_version()
            return True
        except Exception:  # noqa: BLE001
            return False

    def recognize_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult:
        if not self.is_available():
            raise OCREngineUnavailableError("Tesseract binary not found or not runnable")

        import cv2
        import pytesseract

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb, lang=self.settings.tesseract_langs, output_type=pytesseract.Output.DICT
        )

        words: list[OCRWordResult] = []
        n = len(data.get("text", []))
        line_counter = -1
        last_line_key = None
        for i in range(n):
            text = data["text"][i]
            if not text or not text.strip():
                continue
            conf_raw = data["conf"][i]
            try:
                confidence = max(0.0, float(conf_raw)) / 100.0
            except (TypeError, ValueError):
                confidence = 0.0

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = BBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
            polygon = Polygon.from_xy_list([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

            line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if line_key != last_line_key:
                line_counter += 1
                last_line_key = line_key

            words.append(
                OCRWordResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    polygon=polygon,
                    page_number=page_number,
                    block_id=f"blk_{data['block_num'][i]:05d}",
                    line_id=f"ln_{line_counter:05d}",
                    word_id=f"w_{i:05d}",
                    language=detect_language(text),
                    font_size_estimate=round((h / dpi) * 72.0, 1),
                )
            )

        mean_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        return OCRPageResult(page_number=page_number, provider=self.name, words=words, mean_confidence=mean_conf)
