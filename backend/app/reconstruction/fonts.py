"""
Font resolution for PDF/DOCX reconstruction.

Hindi (Devanagari) + English mixed content requires a Unicode font that
covers both scripts — the built-in PDF base-14 fonts (Helvetica etc.) only
cover Latin. `docker/backend/Dockerfile` downloads Noto Sans + Noto Sans
Devanagari at image build time into this directory; if they're missing
(e.g. running outside Docker without having fetched them) we fall back to
the built-in Latin-only font and log a warning rather than failing the
whole pipeline — Latin/number-heavy pages still reconstruct correctly.
"""
from __future__ import annotations

import os

from app.core.logging import get_logger

logger = get_logger(__name__)

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

LATIN_FONT_PATH = os.path.join(_FONT_DIR, "NotoSans-Regular.ttf")
LATIN_BOLD_FONT_PATH = os.path.join(_FONT_DIR, "NotoSans-Bold.ttf")
DEVANAGARI_FONT_PATH = os.path.join(_FONT_DIR, "NotoSansDevanagari-Regular.ttf")

_warned = False


def resolve_body_font_path(bold: bool = False) -> str | None:
    """Returns a TTF path covering Latin + Devanagari (Noto Sans falls back
    to Devanagari-only coverage if the merged font isn't present), or None
    to signal "use the PDF base-14 font" when nothing is available."""
    global _warned
    candidate = LATIN_BOLD_FONT_PATH if bold else LATIN_FONT_PATH
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(DEVANAGARI_FONT_PATH):
        return DEVANAGARI_FONT_PATH
    if not _warned:
        logger.warning(
            "unicode_font_missing",
            message=(
                "No bundled Unicode font found in app/reconstruction/fonts/. "
                "Hindi text will not render correctly in exported PDF/DOCX. "
                "See docker/backend/Dockerfile for how fonts are fetched at build time."
            ),
        )
        _warned = True
    return None


def resolve_devanagari_font_path() -> str | None:
    return DEVANAGARI_FONT_PATH if os.path.exists(DEVANAGARI_FONT_PATH) else None
