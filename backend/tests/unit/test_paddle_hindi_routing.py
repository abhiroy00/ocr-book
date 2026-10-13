"""
English PaddleOCR + Tesseract-for-Hindi-lines routing
(`app.ocr.paddle_engine.PaddleOCREngine._recognize_english_with_hindi_lines`).

Regression for measurements on 2026-10-12: the old full Hindi Paddle pass,
merged by confidence, replaced correct English text on real book pages
("t० the state", "Ieft Iargely") and read Hindi with ~26% character error.
Paddle and Tesseract are faked here; the real-engine numbers live in the
benchmark notes in paddle_engine.py.
"""
import numpy as np
import pytest

import app.ocr.paddle_engine as pe
from app.core.config import Settings

BOX_A = [[10, 10], [300, 10], [300, 40], [10, 40]]
BOX_B = [[10, 60], [300, 60], [300, 90], [10, 90]]


class FakePaddle:
    def __init__(self, lines):
        self.lines = lines
        self.calls = 0

    def ocr(self, image, cls=True):
        self.calls += 1
        return [self.lines]


@pytest.fixture()
def setup(monkeypatch):
    def make(lines, tesseract=("", 0.0), engine="tesseract", langs="en,hi"):
        s = Settings(ocr_hindi_line_engine=engine, paddle_ocr_langs=langs)
        monkeypatch.setattr(pe, "get_settings", lambda: s)
        fakes = {"en": FakePaddle(lines), "hi": FakePaddle([[BOX_A, ("t० the state", 0.99)]])}
        monkeypatch.setattr(pe, "_get_paddle_engine", lambda lang: fakes[lang])
        calls = []

        def fake_tess(image, box):
            calls.append(box)
            return tesseract

        monkeypatch.setattr(pe, "_tesseract_line", fake_tess)
        monkeypatch.setattr(pe.PaddleOCREngine, "is_available", lambda self: True)
        return pe.PaddleOCREngine(), fakes, calls

    return make


IMG = np.zeros((100, 400, 3), dtype=np.uint8)


def test_confident_english_lines_are_kept_and_never_sent_to_tesseract(setup):
    engine, fakes, tess_calls = setup([[BOX_A, ("to the state", 0.98)], [BOX_B, ("so distant", 0.95)]])

    result = engine.recognize_page(IMG, 1, 150)

    assert [w.text for w in result.words] == ["to the state", "so distant"]
    assert tess_calls == []
    assert fakes["hi"].calls == 0  # no full-page Hindi pass any more


def test_an_unreadable_line_is_replaced_by_tesseract_hindi(setup):
    engine, _, tess_calls = setup(
        [[BOX_A, ("i l ii", 0.60)], [BOX_B, ("Agricultural production", 0.99)]],
        tesseract=("कृषि उत्पादन में वृद्धि", 0.91),
    )

    result = engine.recognize_page(IMG, 1, 150)

    texts = [w.text for w in result.words]
    assert "कृषि उत्पादन में वृद्धि" in texts and "Agricultural production" in texts
    assert len(tess_calls) == 1


def test_a_low_confidence_line_keeps_english_when_tesseract_does_not_return_devanagari(setup):
    engine, _, tess_calls = setup([[BOX_A, ("Manchuria 1932", 0.55)]], tesseract=("Manchurla 1932", 0.70))

    result = engine.recognize_page(IMG, 1, 150)

    assert [w.text for w in result.words] == ["Manchuria 1932"]
    assert len(tess_calls) == 1


def test_old_full_hindi_pass_is_still_available_by_setting(setup):
    engine, fakes, tess_calls = setup([[BOX_A, ("to the state", 0.98)]], engine="paddle")

    engine.recognize_page(IMG, 1, 150)

    assert fakes["hi"].calls == 1 and tess_calls == []


def test_english_only_configuration_does_not_route(setup):
    engine, fakes, tess_calls = setup([[BOX_A, ("i l", 0.30)]], langs="en")

    engine.recognize_page(IMG, 1, 150)

    assert tess_calls == [] and fakes["hi"].calls == 0


@pytest.mark.parametrize(
    "text,conf,expected",
    [("the state", 0.99, False), ("the state", 0.50, True), ("i l ii", 0.99, True), ("1932", 0.99, False)],
)
def test_unreadable_detection(monkeypatch, text, conf, expected):
    monkeypatch.setattr(pe, "get_settings", lambda: Settings())
    assert pe._looks_unreadable_as_english(text, conf) is expected


def test_devanagari_ratio():
    assert pe._is_mostly_devanagari("कृषि उत्पादन 38")
    assert not pe._is_mostly_devanagari("Manchuria")
    assert not pe._is_mostly_devanagari("")
