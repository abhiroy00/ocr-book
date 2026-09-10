"""
OpenCV-based image cleaning pipeline (spec section 7).

Every function takes/returns a numpy BGR (or grayscale) image and never
mutates its input in place — callers keep both the original and the
processed image (see DocumentPage.original_image_path / processed_image_path).

`preprocess_page()` composes these into the FAST / BALANCED / HIGH_QUALITY
profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class PreprocessProfile(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_QUALITY = "HIGH_QUALITY"


@dataclass
class PreprocessResult:
    image: np.ndarray
    rotation_applied_deg: float = 0.0
    skew_angle_deg: float = 0.0
    boundary_found: bool = False
    steps_applied: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Rotation / deskew
# ----------------------------------------------------------------------------

_ROTATION_MARGIN_RATIO = 0.15  # a candidate orientation must beat 0deg by this much (relatively) to be chosen

# Coarse 90/180/270 auto-rotation is disabled by default -- see
# `detect_rotation`'s docstring for the real-production evidence behind
# this. Flip to True only for a document corpus you have separately
# verified doesn't trigger the same false-positive pattern (e.g. plain
# prose with no ruled tables), and even then treat it as experimental.
ENABLE_COARSE_ROTATION_DETECTION = False


def _score_rotation_candidates(binary: np.ndarray) -> dict[int, float]:
    """Row-sum variance for each of the 4 axis-aligned orientations of a
    THRESH_BINARY_INV mask (ink=255, background=0). `border_value=0` is
    required here: the default border_value=255 would fill the
    corner-expansion padding a 90/270-degree rotation introduces with fake
    "ink", creating a solid full-width/full-height border streak that
    dominates the real signal (confirmed as the cause of a 258/446 false
    -positive rate on real data before this was fixed)."""
    scores: dict[int, float] = {}
    for angle in (0, 90, 180, 270):
        rotated = _rotate_bound(binary, angle, border_value=0)
        row_sums = rotated.sum(axis=1)
        scores[angle] = float(np.var(row_sums))
    return scores


def detect_rotation(image: np.ndarray) -> float:
    """Coarse page-orientation detection (0/90/180/270) via text-line profile.

    DISABLED BY DEFAULT (returns 0.0 unconditionally unless
    `ENABLE_COARSE_ROTATION_DETECTION` is flipped on). This was originally a
    projection-profile variance heuristic: horizontal text lines create
    sharp peaks/troughs in the row-wise ink-density profile, so the upright
    orientation was expected to score higher than a sideways one.

    Verified against a real 446-page scanned government document (dense
    ruled statistical tables, mixed Hindi/English, 150 DPI): even after
    fixing an unrelated border-padding bug that made it worse (258/446
    pages wrongly flagged), the heuristic still produced false positives
    with no clean score threshold separating them from genuine rotations
    -- spot-checking pages across the full confidence range, INCLUDING the
    two most "confident" outliers (56x and 8x the baseline score), found
    every single one to be a normal upright page. Zero genuine rotations
    were found anywhere in the document. Long unbroken table ruling lines
    (common in this kind of content) concentrate ink into a few rows/
    columns in a way that mimics or exceeds the signal genuine sideways
    text produces, so no margin threshold can safely separate the two for
    this class of document.

    Because a false-positive flip actively destroys correct OCR content
    (turns a readable page into garbage) while a missed genuine rotation
    merely leaves that one page's OCR confidence low (recoverable, visible
    in the quality report), the safe default is to not auto-rotate at all.
    Fine-grained deskew (`deskew_image`, capped at +-20 degrees) is
    unaffected and still runs -- it corrects the small skew angles typical
    of flatbed scanning without ever attempting a 90-degree-class flip.
    Genuinely sideways/upside-down scans remain a known limitation (see
    README); reliably catching those needs an OCR-confidence-based check
    (actually run OCR at each candidate angle and compare confidence),
    not a pixel-projection heuristic -- which is a meaningfully more
    expensive per-page cost this codebase has not yet opted into.
    """
    if not ENABLE_COARSE_ROTATION_DETECTION:
        return 0.0

    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scores = _score_rotation_candidates(binary)

    baseline = scores[0]
    best_angle = 0.0
    best_score = baseline
    for angle in (90, 180, 270):
        if scores[angle] > best_score * (1.0 + _ROTATION_MARGIN_RATIO):
            best_score = scores[angle]
            best_angle = float(angle)
    return best_angle


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate by an arbitrary angle (degrees, counter-clockwise), expanding
    the canvas so no content is cropped."""
    if abs(angle_deg) < 1e-3:
        return image.copy()
    return _rotate_bound(image, angle_deg, border_value=_border_value(image))


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Fine-grained deskew (small angles, typically < 15deg) via minAreaRect
    over ink pixels. Returns (deskewed_image, angle_deg_applied)."""
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return image.copy(), 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angle in [-90, 0); normalize to a small signed
    # deviation from horizontal.
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 20:
        # Not a plausible skew for a scanned page — likely a false minAreaRect
        # fit on a sparse page; skip correction rather than risk a bad rotate.
        return image.copy(), 0.0

    corrected = _rotate_bound(image, -angle, border_value=_border_value(image))
    return corrected, float(-angle)


# ----------------------------------------------------------------------------
# Denoise / background / contrast
# ----------------------------------------------------------------------------

def remove_noise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Denoise while preserving text edges (fastNlMeansDenoising)."""
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, h=strength, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(image, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21)


def remove_background(image: np.ndarray) -> np.ndarray:
    """Flatten uneven scan background (aged paper, shadows, bleed-through)
    by estimating a large-kernel background surface and dividing it out."""
    gray = _to_gray(image)
    bg = cv2.medianBlur(gray, 51)
    bg = np.where(bg == 0, 1, bg).astype(np.float32)
    normalized = (gray.astype(np.float32) / bg) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    if len(image.shape) == 3:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement, applied on luminance only for color images."""
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def adaptive_threshold(image: np.ndarray, block_size: int = 35, c: int = 15) -> np.ndarray:
    gray = _to_gray(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size | 1, c
    )


# ----------------------------------------------------------------------------
# Page boundary / perspective / borders / dewarp
# ----------------------------------------------------------------------------

def detect_page_boundary(image: np.ndarray) -> np.ndarray | None:
    """Finds the largest 4-point contour that plausibly is the page edge.
    Returns an (4,2) float32 array of corners, or None if not found."""
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    page_area = h * w
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.3 * page_area:  # ignore anything not plausibly the full page
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    return best


def perspective_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a quadrilateral page region to a rectangle (removes camera-angle
    perspective distortion for photographed, as opposed to flat-bed-scanned,
    pages)."""
    rect = _order_corners(corners)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return image.copy()

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), borderValue=_border_value(image))


def detect_and_remove_black_borders(image: np.ndarray, threshold: int = 40, margin: int = 2) -> np.ndarray:
    """Crops the dark scanner-bed border frequently present around scanned
    pages (from flatbed scanner lids / book gutters)."""
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    mask = gray > threshold  # True = "not black"

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image.copy()

    y1, y2 = max(0, rows[0] - margin), min(h, rows[-1] + margin)
    x1, x2 = max(0, cols[0] - margin), min(w, cols[-1] + margin)
    if (y2 - y1) < 0.5 * h or (x2 - x1) < 0.5 * w:
        # Cropping would remove too much — likely a mostly-dark page, skip.
        return image.copy()
    return image[y1:y2, x1:x2].copy()


def dewarp_page(image: np.ndarray) -> np.ndarray:
    """Best-effort curved-page correction: detects the page boundary and, if
    found, applies perspective correction; this straightens bowed book
    spines/photographed pages but is not a full non-rigid unwarp network."""
    corners = detect_page_boundary(image)
    if corners is None:
        return image.copy()
    return perspective_correction(image, corners)


# ----------------------------------------------------------------------------
# Profile pipeline
# ----------------------------------------------------------------------------

def preprocess_page(image: np.ndarray, profile: PreprocessProfile) -> PreprocessResult:
    steps: list[str] = []
    working = image.copy()

    coarse_angle = detect_rotation(working)
    if coarse_angle != 0.0:
        working = _rotate_bound(working, coarse_angle, border_value=_border_value(working))
        steps.append(f"rotate({coarse_angle})")

    working, skew_angle = deskew_image(working)
    if skew_angle != 0.0:
        steps.append(f"deskew({skew_angle:.2f})")

    if profile == PreprocessProfile.FAST:
        working = _to_gray(working)
        steps.append("grayscale")
        working = cv2.fastNlMeansDenoising(working, h=5, templateWindowSize=7, searchWindowSize=21)
        steps.append("mild_denoise")
        return PreprocessResult(working, coarse_angle, skew_angle, False, steps)

    boundary_found = False
    if profile == PreprocessProfile.HIGH_QUALITY:
        working = dewarp_page(working)
        corners = detect_page_boundary(image)
        boundary_found = corners is not None
        steps.append("dewarp")

    working = detect_and_remove_black_borders(working)
    steps.append("border_removal")

    working = remove_noise(working, strength=7 if profile == PreprocessProfile.BALANCED else 10)
    steps.append("denoise")

    working = remove_background(working)
    steps.append("background_normalize")

    working = enhance_contrast(working)
    steps.append("contrast_enhance")

    if profile == PreprocessProfile.BALANCED:
        working = _to_gray(working)
        working = adaptive_threshold(working)
        steps.append("adaptive_threshold")

    return PreprocessResult(working, coarse_angle, skew_angle, boundary_found, steps)


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _border_value(image: np.ndarray):
    return (255, 255, 255) if len(image.shape) == 3 else 255


def _rotate_bound(image: np.ndarray, angle_deg: float, border_value=255) -> np.ndarray:
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - cx
    matrix[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=border_value)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect
