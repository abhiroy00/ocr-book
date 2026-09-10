from unittest.mock import patch

from app.core.config import Settings


def test_resolved_ocr_workers_uses_explicit_value_when_set():
    s = Settings(ocr_workers=3)
    assert s.resolved_ocr_workers == 3


def test_resolved_ocr_workers_auto_picks_half_cpu_count_when_zero():
    s = Settings(ocr_workers=0)
    with patch("os.cpu_count", return_value=8):
        assert s.resolved_ocr_workers == 4


def test_resolved_ocr_workers_never_goes_below_one():
    s = Settings(ocr_workers=0)
    with patch("os.cpu_count", return_value=1):
        assert s.resolved_ocr_workers == 1


def test_resolved_ocr_workers_caps_at_six_even_on_a_huge_machine():
    """A bounded pool is the whole point -- OCR_WORKERS must never
    auto-scale to "one process per every core", let alone one per page,
    on a big server."""
    s = Settings(ocr_workers=0)
    with patch("os.cpu_count", return_value=64):
        assert s.resolved_ocr_workers == 6
