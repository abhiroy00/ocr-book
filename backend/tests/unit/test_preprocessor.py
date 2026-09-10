import cv2
import numpy as np

from app.vision.preprocessor import (
    PreprocessProfile,
    _score_rotation_candidates,
    _to_gray,
    deskew_image,
    detect_and_remove_black_borders,
    detect_rotation,
    preprocess_page,
)


def _make_text_like_image(angle_deg: float = 0.0) -> np.ndarray:
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    for y in range(50, 350, 25):
        cv2.line(img, (50, y), (550, y), (0, 0, 0), 3)
    if angle_deg:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))
    return img


def _make_asymmetric_document_image() -> np.ndarray:
    """An upright page with a title (thick, short line) near the top, body
    paragraph lines (thin, varying width) in the middle, and a small table
    grid near the bottom -- deliberately asymmetric top-to-bottom, the kind
    of real content where a naive "pick the rotation with highest row-sum
    variance" heuristic can't distinguish 0deg from a 180deg flip (variance
    is invariant to reversing the row order) and ends up guessing."""
    img = np.full((1000, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 60), (500, 90), (0, 0, 0), -1)  # title block
    for i, y in enumerate(range(150, 500, 30)):
        width = 550 - (i * 37) % 300  # varying line lengths, like real text
        cv2.line(img, (100, y), (100 + width, y), (0, 0, 0), 4)
    for r in range(3):
        y = 700 + r * 40
        cv2.line(img, (100, y), (500, y), (0, 0, 0), 2)
    for c in range(4):
        x = 100 + c * 133
        cv2.line(img, (x, 700), (x, 780), (0, 0, 0), 2)
    return img


def _make_ruled_table_image() -> np.ndarray:
    """A page dominated by long, unbroken table ruling lines -- the real
    failure pattern found in production: a 446-page scanned government
    document of statistical tables had 258/446 pages (58%) wrongly
    auto-rotated because long ruling lines concentrate ink in a way the
    row-sum-variance heuristic mistook for (or preferred over) genuine
    upright text-line structure, on some pages by as much as 56x the
    baseline score with no genuine rotation present at all."""
    img = np.full((1000, 700, 3), 255, dtype=np.uint8)
    for i, y in enumerate(range(120, 900, 60)):
        width = 550 - (i * 41) % 300
        cv2.line(img, (100, y), (100 + width, y), (0, 0, 0), 2)  # sparse text rows
    for c in range(7):  # long, near-full-height column-separator rules
        x = 100 + c * 80
        cv2.line(img, (x, 100), (x, 920), (0, 0, 0), 2)
    return img


def test_detect_rotation_does_not_flip_a_normal_upright_page():
    page = _make_asymmetric_document_image()
    angle = detect_rotation(page)
    assert angle == 0.0


def test_detect_rotation_is_disabled_by_default_even_for_a_clear_90_degree_sideways_page():
    """Coarse 90/180/270 auto-rotation is disabled by default (see
    `detect_rotation`'s docstring for the real-production evidence): a
    false-positive flip actively destroys correct OCR content, while a
    missed genuine rotation just leaves that page's confidence low and
    recoverable. `detect_rotation` must therefore return 0.0 unconditionally
    -- even for content that IS genuinely sideways -- until a materially
    more robust (e.g. OCR-confidence-based) detector replaces it."""
    page = _make_asymmetric_document_image()
    sideways = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
    assert detect_rotation(sideways) == 0.0


def test_detect_rotation_is_disabled_by_default_for_a_dense_ruled_table_page():
    """Direct regression test for the real bug: previously, a page like
    this (long ruling lines, sparse text) could score a wrong orientation
    far higher than upright -- confirmed on real data up to a 56x margin.
    With auto-rotation disabled by default this must stay a no-op."""
    page = _make_ruled_table_image()
    assert detect_rotation(page) == 0.0


def test_rotation_scoring_mechanism_still_recognizes_a_clear_90_degree_flip():
    """The underlying row-sum-variance scoring (`_score_rotation_candidates`)
    is not broken math -- it correctly scores a genuinely-sideways page
    higher than upright for clean, non-tabular content. Disabling
    `detect_rotation` by default was a deliberate safety trade-off for
    ruled-table-heavy real-world documents (see its docstring), not
    evidence the scoring itself is wrong; this test documents that
    distinction and would catch a regression in the scoring function
    itself."""
    page = _make_asymmetric_document_image()
    sideways = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
    gray = _to_gray(sideways)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scores = _score_rotation_candidates(binary)
    best_angle = max(scores, key=scores.get)
    assert best_angle in (90, 270)
    assert scores[best_angle] > scores[0] * 1.15


def test_deskew_corrects_small_rotation():
    skewed = _make_text_like_image(angle_deg=5.0)
    corrected, angle = deskew_image(skewed)
    assert corrected.shape[0] > 0 and corrected.shape[1] > 0
    # Should detect a non-trivial correction in a plausible direction/magnitude.
    assert abs(angle) <= 20


def test_deskew_noop_on_already_straight_image():
    straight = _make_text_like_image(angle_deg=0.0)
    corrected, angle = deskew_image(straight)
    assert abs(angle) < 3.0


def test_black_border_removal_crops_dark_frame():
    page = np.full((400, 400, 3), 255, dtype=np.uint8)
    framed = np.zeros((440, 440, 3), dtype=np.uint8)
    framed[20:420, 20:420] = page
    cropped = detect_and_remove_black_borders(framed, threshold=40)
    assert cropped.shape[0] <= framed.shape[0]
    assert cropped.shape[1] <= framed.shape[1]
    # Center should remain white after crop.
    assert cropped[cropped.shape[0] // 2, cropped.shape[1] // 2].mean() > 200


def test_preprocess_page_fast_profile_returns_grayscale():
    img = _make_text_like_image()
    result = preprocess_page(img, PreprocessProfile.FAST)
    assert len(result.image.shape) == 2  # grayscale
    assert "grayscale" in result.steps_applied


def test_preprocess_page_balanced_profile_runs_without_error():
    img = _make_text_like_image()
    result = preprocess_page(img, PreprocessProfile.BALANCED)
    assert result.image is not None
    assert result.image.size > 0


def test_preprocess_never_mutates_input():
    img = _make_text_like_image()
    original_copy = img.copy()
    preprocess_page(img, PreprocessProfile.BALANCED)
    assert np.array_equal(img, original_copy)
