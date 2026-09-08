"""
AI-assisted SUGGESTIONS only (spec sections 25/27): layout/structure
reasoning, OCR-correction suggestions, ambiguous-table-structure hints.
These are ALWAYS surfaced to the user for approval — never auto-applied to
`OCRBlock.text` / `TableCell.text` (the pipeline writes suggestions to a
separate `corrected_text`/`edited_text`-adjacent "suggestion" field the API
returns, distinct from the source-of-truth text).

Uses Ollama (local, block-level, optional) when configured/reachable, and
otherwise reports unavailable — this module NEVER blocks or fails the main
pipeline (spec section 29): every method degrades to returning `None`.
"""
from __future__ import annotations

import json
import re

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_CORRECTION_PROMPT = (
    "You are assisting a human reviewer of OCR output from a scanned document. "
    "The OCR engine read the following text from one region of the page:\n\n"
    '"{text}"\n\n'
    "If you believe this contains an OCR misread (e.g. a character confusion "
    "like O/0, l/1, rn/m), suggest a corrected reading. Do NOT change "
    "historical facts, numbers, dates, or values that could plausibly be "
    "correct as-is even if unusual — only fix clear OCR artifacts. If the "
    "text looks correct, or you are not confident, respond with exactly the "
    'original text unchanged. Respond ONLY with JSON: {{"suggestion": "...", '
    '"changed": true|false, "reason": "..."}}'
)


class AISuggestionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_ollama_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def suggest_ocr_correction(self, text: str) -> dict | None:
        """Returns {"suggestion": str, "changed": bool, "reason": str} or None
        if no AI backend is available/reachable — callers must treat None as
        "no suggestion available", never as an error that breaks the flow."""
        if not text or not text.strip():
            return None
        if not self.is_ollama_available():
            logger.info("ai_suggestion_skipped", reason="ollama_unavailable")
            return None

        payload = {
            "model": self.settings.ollama_model,
            "messages": [{"role": "user", "content": _CORRECTION_PROMPT.format(text=text)}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = httpx.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("ai_suggestion_failed", error=str(exc))
            return None

        content = resp.json().get("message", {}).get("content", "")
        parsed = _extract_json_object(content)
        if not parsed or "suggestion" not in parsed:
            return None
        return {
            "suggestion": str(parsed.get("suggestion", text)),
            "changed": bool(parsed.get("changed", False)),
            "reason": str(parsed.get("reason", "")),
        }


def _extract_json_object(content: str) -> dict | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
