"""
Coordinate-system conversions (spec section 15).

Internally every block is stored with BOTH a pixel bbox (at the page's
render DPI) and a normalized [0,1] bbox. Reconstruction targets (PDF points,
DOCX EMUs) are always derived from the *normalized* coordinates against the
target page size, so DPI never leaks into positioning errors between the
scan resolution and the output document.

Units:
    px      : pixels at a given DPI
    pt      : PDF points, 72 per inch
    emu     : DOCX English Metric Units, 914400 per inch
"""
from __future__ import annotations

PT_PER_INCH = 72.0
EMU_PER_INCH = 914_400


def px_to_inches(px: float, dpi: int) -> float:
    return px / float(dpi)


def px_to_pt(px: float, dpi: int) -> float:
    return px_to_inches(px, dpi) * PT_PER_INCH


def px_to_emu(px: float, dpi: int) -> int:
    return int(round(px_to_inches(px, dpi) * EMU_PER_INCH))


def norm_to_px(value_norm: float, page_dim_px: int) -> float:
    return value_norm * page_dim_px


def norm_bbox_to_pt(x1: float, y1: float, x2: float, y2: float, page_width_pt: float, page_height_pt: float):
    """Normalized [0,1] bbox -> PDF point bbox against a target page size."""
    return (
        x1 * page_width_pt,
        y1 * page_height_pt,
        x2 * page_width_pt,
        y2 * page_height_pt,
    )


def norm_bbox_to_emu(x1: float, y1: float, x2: float, y2: float, page_width_emu: int, page_height_emu: int):
    return (
        int(x1 * page_width_emu),
        int(y1 * page_height_emu),
        int(x2 * page_width_emu),
        int(y2 * page_height_emu),
    )


def page_size_pt(width_px: int, height_px: int, dpi: int) -> tuple[float, float]:
    return (px_to_pt(width_px, dpi), px_to_pt(height_px, dpi))


def page_size_emu(width_px: int, height_px: int, dpi: int) -> tuple[int, int]:
    return (px_to_emu(width_px, dpi), px_to_emu(height_px, dpi))
