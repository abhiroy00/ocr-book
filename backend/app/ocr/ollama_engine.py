"""
Optional local Ollama vision-model OCR provider (spec section 27). Requires
a locally running Ollama with a vision-capable model (e.g. `llava`,
`llama3.2-vision`, `bakllava`) pulled. Entirely optional; absence never
breaks the app — `is_available()` just returns False and the pipeline
falls back to PaddleOCR/Tesseract.

Note: per spec, Ollama's primary intended role is block-level layout/
structure/validation suggestions (see app/ai/validator.py), not bulk OCR of
whole documents — but it is also wired in as a selectable OCRProvider so the
UI's engine picker (spec section 26) has a working local-LLM OCR option.
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import httpx
import numpy as np

from app.core.config import get_settings
from app.ocr.base import OCREngineUnavailableError, OCRProvider
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRPageResult, OCRWordResult
from app.utils.language import detect_language

_OLLAMA_CONFIDENCE = 0.7

_PROMPT = (
    "Transcribe every visible text region on this scanned page (Hindi and "
    "English, numbers, table cells) exactly as written, without translating "
    "or correcting anything. Respond ONLY with a JSON array of objects "
    '{"text": <string>, "bbox": [x1,y1,x2,y2]} where bbox is normalized to '
    "[0,1] of the image dimensions, one object per line of text, in reading order."
)


class OllamaEngine(OCRProvider):
    name = "ollama"

    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def recognize_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult:
        if not self.is_available():
            raise OCREngineUnavailableError(f"Ollama not reachable at {self.settings.ollama_base_url}")

        h, w = image.shape[:2]
        b64 = base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode("ascii")

        payload = {
            "model": self.settings.ollama_model,
            "messages": [{"role": "user", "content": _PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = httpx.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=180.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OCREngineUnavailableError(f"Ollama request failed: {exc}") from exc

        content = resp.json().get("message", {}).get("content", "")
        regions = _extract_json_array(content)

        words: list[OCRWordResult] = []
        for i, region in enumerate(regions):
            text = str(region.get("text", "")).strip()
            bbox_norm = region.get("bbox")
            if not text or not bbox_norm or len(bbox_norm) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox_norm)
            bbox = BBox(x1=x1 * w, y1=y1 * h, x2=x2 * w, y2=y2 * h)
            polygon = Polygon.from_xy_list(
                [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]]
            )
            words.append(
                OCRWordResult(
                    text=text,
                    confidence=_OLLAMA_CONFIDENCE,
                    bbox=bbox,
                    polygon=polygon,
                    page_number=page_number,
                    block_id=f"blk_{i:05d}",
                    line_id=f"ln_{i:05d}",
                    language=detect_language(text),
                    font_size_estimate=round((bbox.height / dpi) * 72.0, 1),
                )
            )

        mean_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        return OCRPageResult(page_number=page_number, provider=self.name, words=words, mean_confidence=mean_conf)


def _extract_json_array(content: str) -> list[dict]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []
