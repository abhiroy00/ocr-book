"""
Automated quality checks (spec section 24): visual similarity between the
original (cleaned) scan and the reconstructed page rendering, plus
aggregate OCR/layout/table confidence per page. Never modifies the
original image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

REVIEW_CONFIDENCE_THRESHOLD = 0.80
REVIEW_SIMILARITY_THRESHOLD = 0.85


@dataclass
class SimilarityResult:
    ssim_score: float
    mse: float
    pixel_diff_percentage: float


def compare_images(original: np.ndarray, reconstructed: np.ndarray) -> SimilarityResult:
    """Compares two page renderings (already the same approximate size —
    caller resizes the reconstructed render to the original's dimensions)."""
    a = _to_gray(original)
    b = _to_gray(reconstructed)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    score, diff = ssim(a, b, full=True)
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

    diff_mask = diff < 0.5
    pixel_diff_percentage = float(np.mean(diff_mask) * 100.0)

    return SimilarityResult(ssim_score=float(score), mse=mse, pixel_diff_percentage=pixel_diff_percentage)


def needs_review(ocr_confidence: float | None, similarity: float | None) -> bool:
    if ocr_confidence is not None and ocr_confidence < REVIEW_CONFIDENCE_THRESHOLD:
        return True
    if similarity is not None and similarity < REVIEW_SIMILARITY_THRESHOLD:
        return True
    return False


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
