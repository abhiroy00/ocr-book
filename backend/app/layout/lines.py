"""
Ruling-line detection (spec section 11: horizontal_line / vertical_line
blocks, and section 12's border/grid input for table detection).

Uses morphological opening with long thin structuring elements — the
standard, reliable technique for extracting straight ruling lines from a
scanned page independent of text content.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.geometry import BBox


@dataclass
class DetectedLine:
    bbox: BBox
    orientation: str  # "horizontal" | "vertical"
    thickness: float


def detect_lines(gray_or_bgr: np.ndarray) -> list[DetectedLine]:
    gray = gray_or_bgr if len(gray_or_bgr.shape) == 2 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape[:2]
    horiz_len = max(20, w // 40)
    vert_len = max(20, h // 40)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horiz_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vert_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    lines: list[DetectedLine] = []
    lines.extend(_mask_to_lines(horiz_mask, "horizontal"))
    lines.extend(_mask_to_lines(vert_mask, "vertical"))
    return lines


def _mask_to_lines(mask: np.ndarray, orientation: str) -> list[DetectedLine]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[DetectedLine] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if orientation == "horizontal":
            if w < 25 or h > max(6, w * 0.05):
                continue
            thickness = float(h)
        else:
            if h < 25 or w > max(6, h * 0.05):
                continue
            thickness = float(w)
        out.append(DetectedLine(bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h), orientation=orientation, thickness=thickness))
    return out


def build_grid_mask(gray_or_bgr: np.ndarray) -> np.ndarray:
    """Combined horizontal+vertical line mask — used by the table engine to
    find candidate table grids (spec section 12)."""
    gray = gray_or_bgr if len(gray_or_bgr.shape) == 2 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape[:2]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))

    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    grid = cv2.bitwise_or(horiz, vert)
    return grid
