from app.schemas.geometry import BBox, NormBBox
from app.utils.coordinates import page_size_emu, page_size_pt, px_to_emu, px_to_pt


def test_px_to_pt_at_72_dpi_is_identity():
    assert px_to_pt(100, 72) == 100.0


def test_px_to_pt_at_300_dpi():
    # 300px at 300dpi = 1 inch = 72pt
    assert abs(px_to_pt(300, 300) - 72.0) < 1e-6


def test_px_to_emu_one_inch():
    assert px_to_emu(300, 300) == 914400


def test_page_size_pt_matches_a4_like_ratio():
    # 2481x3508 px @ 300dpi ~ A4 (8.27x11.69 in) = 595.2 x 841.9 pt
    w_pt, h_pt = page_size_pt(2481, 3508, 300)
    assert abs(w_pt - 595.44) < 1.0
    assert abs(h_pt - 841.92) < 1.0


def test_page_size_emu_scales_with_dpi():
    w1, h1 = page_size_emu(300, 300, 300)
    w2, h2 = page_size_emu(600, 600, 600)
    assert w1 == w2
    assert h1 == h2


def test_bbox_to_norm_roundtrip():
    bbox = BBox(x1=100, y1=200, x2=300, y2=400)
    norm = bbox.to_norm(1000, 2000)
    back = norm.to_pixels(1000, 2000)
    assert abs(back.x1 - bbox.x1) < 1e-6
    assert abs(back.y2 - bbox.y2) < 1e-6


def test_norm_bbox_bounds():
    bbox = BBox(x1=0, y1=0, x2=500, y2=1000)
    norm = bbox.to_norm(1000, 1000)
    assert norm.x1 == 0.0
    assert norm.x2 == 0.5
    assert norm.y2 == 1.0
