import fitz
import numpy as np
import pytest

from app.reconstruction.clean_pdf import render_clean_pdf
from app.reconstruction.page_transform import (
    A4_LANDSCAPE_PT,
    A4_PORTRAIT_PT,
    _MIN_MARGIN_PT,
    compute_page_transform,
)


def _fits(t, width_px, height_px):
    content_w = (t.crop_x2 - t.crop_x1) * (72.0 / t.dpi) * t.scale
    content_h = (t.crop_y2 - t.crop_y1) * (72.0 / t.dpi) * t.scale
    return (
        content_w <= t.a4_width_pt - 2 * _MIN_MARGIN_PT + 1e-6
        and content_h <= t.a4_height_pt - 2 * _MIN_MARGIN_PT + 1e-6
    )


@pytest.mark.parametrize(
    "width_px,height_px",
    [
        (1240, 1754),  # A4-ish portrait scan at 150 DPI -- previously a NameError
        (1500, 1754),  # wide-ish portrait: width is the tighter fit
        (800, 2400),  # tall, narrow portrait
        (2400, 1200),  # landscape
        (2400, 1700),  # landscape where height is the tighter fit
    ],
)
def test_compute_page_transform_fits_content_inside_margins(width_px, height_px):
    t = compute_page_transform(width_px, height_px, [], dpi=150)

    assert (t.a4_width_pt, t.a4_height_pt) in (A4_PORTRAIT_PT, A4_LANDSCAPE_PT)
    assert t.scale > 0
    assert _fits(t, width_px, height_px)
    assert t.offset_x_pt >= _MIN_MARGIN_PT - 1e-6
    assert t.offset_y_pt >= _MIN_MARGIN_PT - 1e-6


@pytest.mark.parametrize("width_px,height_px", [(1240, 1754), (800, 2400), (2400, 1200), (2400, 1700)])
def test_compute_page_transform_fills_the_limiting_dimension(width_px, height_px):
    t = compute_page_transform(width_px, height_px, [], dpi=150)
    content_w = width_px * (72.0 / 150) * t.scale
    content_h = height_px * (72.0 / 150) * t.scale
    fills_w = content_w == pytest.approx(t.a4_width_pt - 2 * _MIN_MARGIN_PT)
    fills_h = content_h == pytest.approx(t.a4_height_pt - 2 * _MIN_MARGIN_PT)
    assert fills_w or fills_h


def test_render_clean_pdf_builds_portrait_pages():
    image = np.full((1754, 1240, 3), 255, dtype=np.uint8)
    pdf = render_clean_pdf([(image, 150, []), (image, 150, [])])

    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert doc.page_count == 2
        assert doc[0].rect.width == pytest.approx(A4_PORTRAIT_PT[0], abs=0.01)
        assert doc[0].rect.height == pytest.approx(A4_PORTRAIT_PT[1], abs=0.01)
    finally:
        doc.close()
