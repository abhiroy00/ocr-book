import numpy as np

from app.services.quality import compare_images, needs_review


def test_compare_images_identical_gives_high_similarity():
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = compare_images(img, img.copy())
    assert result.ssim_score > 0.99
    assert result.mse < 1.0


def test_compare_images_different_gives_lower_similarity():
    a = np.full((200, 200, 3), 255, dtype=np.uint8)
    b = np.zeros((200, 200, 3), dtype=np.uint8)
    result = compare_images(a, b)
    assert result.ssim_score < 0.5


def test_needs_review_flags_low_confidence():
    assert needs_review(0.5, 0.99) is True


def test_needs_review_flags_low_similarity():
    assert needs_review(0.99, 0.5) is True


def test_needs_review_false_when_both_good():
    assert needs_review(0.95, 0.95) is False


def test_needs_review_handles_missing_values():
    assert needs_review(None, None) is False
