"""
Optional NVIDIA VLM OCR provider (spec section 26). Entirely optional — the
app must run without NVIDIA_API_KEY; this engine simply reports
`is_available() == False` in that case and the pipeline falls back.

Calls an OpenAI-compatible chat-completions endpoint (NVIDIA NIM /
integrate.api.nvidia.com) with the page image and asks the VLM to return a
JSON array of grounded text regions: [{"text": ..., "bbox": [x1,y1,x2,y2]}]
with bbox normalized to [0,1] of the image. Confidence is not natively
provided by VLM decoding, so a conservative fixed confidence is assigned and
surfaced to the reviewer UI (never silently treated as high-confidence OCR).
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

_VLM_CONFIDENCE = 0.75  # VLM grounding is not a calibrated probability; flag for review

_PROMPT = (
    "You are an OCR system. Read every piece of text visible in this scanned "
    "document page, including Hindi (Devanagari) and English text, numbers, "
    "and table contents. Return ONLY a JSON array, no prose, where each "
    "element is {\"text\": <exact text, unmodified>, \"bbox\": [x1,y1,x2,y2]} "
    "with bbox coordinates normalized to the [0,1] range of the image width/"
    "height. One element per line/row of text, in reading order. Do not "
    "translate, correct, or reformat any text — transcribe it exactly as it "
    "visually appears, including historical/unusual values."
)


class NvidiaVLMEngine(OCRProvider):
    name = "nvidia"

    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        return bool(self.settings.nvidia_api_key)

    def recognize_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult:
        if not self.is_available():
            raise OCREngineUnavailableError("NVIDIA_API_KEY is not configured")

        h, w = image.shape[:2]
        b64 = base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode("ascii")

        payload = {
            "model": self.settings.nvidia_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.settings.nvidia_api_key}", "Content-Type": "application/json"}

        try:
            resp = httpx.post(
                f"{self.settings.nvidia_base_url}/chat/completions", json=payload, headers=headers, timeout=90.0
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OCREngineUnavailableError(f"NVIDIA API request failed: {exc}") from exc

        content = resp.json()["choices"][0]["message"]["content"]
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
                    confidence=_VLM_CONFIDENCE,
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
