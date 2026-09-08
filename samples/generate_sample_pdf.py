"""
Generates a small, realistic synthetic test PDF that mirrors the real-world
sample document referenced in spec section 36 ("Hariyana Agriculture Year
1989-90") without hardcoding or shipping the actual scanned file: mixed
Hindi (Devanagari) + English text, a heading, a multi-row/multi-column data
table with numeric values, a page number, and a second page — enough to
exercise every stage of the pipeline (render, preprocess, OCR, layout,
table detection, reconstruction, export) quickly, deliberately kept to a
handful of pages so it runs in seconds rather than the hours a real
400+ page scan takes on CPU-only OCR.

Usage:
    python samples/generate_sample_pdf.py
    # writes samples/sample_test_document.pdf

Requires PyMuPDF (already a backend dependency: `pip install -r
backend/requirements.txt`, or run inside the backend container:
    docker compose exec backend python /app/backend/../samples/generate_sample_pdf.py
  — or simplest, copy this file's logic into a one-off `docker compose exec
  backend python -c "..."` call, which is how it was first prototyped).
"""
from __future__ import annotations

import os

import fitz  # PyMuPDF

OUTPUT_PATH = os.environ.get(
    "SAMPLE_OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "sample_test_document.pdf")
)

# A Devanagari-capable font is required to render real Hindi glyphs (not
# just placeholder boxes). Falls back to Latin-only content if unavailable
# (matches the same graceful-degradation the app itself uses). Overridable
# via SAMPLE_FONT_PATH for environments where this script's relative path
# to backend/app/reconstruction/fonts/ doesn't apply (e.g. run standalone
# inside a container that only has /app/backend mounted).
_FONT_CANDIDATES = [
    p
    for p in [
        os.environ.get("SAMPLE_FONT_PATH"),
        os.path.join(os.path.dirname(__file__), "..", "backend", "app", "reconstruction", "fonts", "NotoSansDevanagari-Regular.ttf"),
    ]
    if p
]


def _find_devanagari_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def build_page_1(doc: fitz.Document, hindi_font: str | None) -> None:
    page = doc.new_page(width=595, height=842)  # A4 in points

    page.insert_text((72, 72), "HARYANA AGRICULTURE REPORT", fontsize=18)
    page.insert_text((72, 96), "Annual Statistical Summary 1989-90", fontsize=11)

    if hindi_font:
        page.insert_font(fontname="devanagari", fontfile=hindi_font)
        page.insert_text((72, 130), "हरियाणा कृषि रिपोर्ट", fontsize=14, fontname="devanagari")
        page.insert_text((72, 150), "वर्ष 1989-90 के लिए वार्षिक सांख्यिकीय सारांश", fontsize=10, fontname="devanagari")

    body_lines = [
        "This report summarizes district-wise rainfall and cultivated area",
        "statistics for the agricultural year 1-7-89 to 30-6-90, compiled",
        "from field survey records across all districts of the state.",
    ]
    y = 190
    for line in body_lines:
        page.insert_text((72, y), line, fontsize=10)
        y += 18

    # District rainfall data table: header row + 4 district rows, 3 columns.
    x0, y0, col_w, row_h = 72, 280, 150, 28
    n_rows, n_cols = 5, 3
    for r in range(n_rows + 1):
        page.draw_line((x0, y0 + r * row_h), (x0 + n_cols * col_w, y0 + r * row_h))
    for c in range(n_cols + 1):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + n_rows * row_h))

    headers = ["District", "Rainfall (mm)", "Area (hectares)"]
    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 8, y0 + 18), h, fontsize=10)

    rows = [
        ("Ambala", "16.5", "1574"),
        ("Hisar", "12.8", "2103"),
        ("Karnal", "18.2", "1890"),
        ("Rohtak", "14.1", "1642"),
    ]
    for i, (district, rain, area) in enumerate(rows, start=1):
        yy = y0 + i * row_h + 18
        page.insert_text((x0 + 8, yy), district, fontsize=10)
        page.insert_text((x0 + col_w + 8, yy), rain, fontsize=10)
        page.insert_text((x0 + 2 * col_w + 8, yy), area, fontsize=10)

    footer_note = "Source: District Agriculture Offices, historical values preserved as recorded."
    page.insert_text((72, 760), footer_note, fontsize=8)
    page.insert_text((290, 800), "1", fontsize=10)


def build_page_2(doc: fitz.Document, hindi_font: str | None) -> None:
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 72), "SECTION 2: CROP-WISE SUMMARY", fontsize=16)
    if hindi_font:
        # PyMuPDF font registration is per-PAGE, not per-document — each
        # page that uses a custom font must call insert_font() itself, or
        # insert_text(..., fontname=...) raises "need font file or buffer".
        page.insert_font(fontname="devanagari", fontfile=hindi_font)
        page.insert_text((72, 100), "फसल-वार सारांश", fontsize=13, fontname="devanagari")

    page.insert_text((72, 140), "The following table lists total cultivated area and yield by crop", fontsize=10)
    page.insert_text((72, 158), "type across the reporting period, alongside the prior-year figures", fontsize=10)
    page.insert_text((72, 176), "for comparison.", fontsize=10)

    # Crop table with a merged header spanning two sub-columns (colspan test).
    x0, y0, col_w, row_h = 72, 220, 110, 26
    n_rows, n_cols = 4, 4
    for r in range(n_rows + 1):
        page.draw_line((x0, y0 + r * row_h), (x0 + n_cols * col_w, y0 + r * row_h))
    for c in range(n_cols + 1):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + n_rows * row_h))
    # Merge the two "Yield" sub-columns' header cell by erasing the divider
    # between them in the header row only (visual colspan).
    page.draw_line(
        (x0 + 2 * col_w, y0), (x0 + 2 * col_w, y0 + row_h), color=(1, 1, 1), width=2
    )

    page.insert_text((x0 + 8, y0 + 18), "Crop", fontsize=9)
    page.insert_text((x0 + col_w + 8, y0 + 18), "Area (ha)", fontsize=9)
    page.insert_text((x0 + 2 * col_w + 8, y0 + 18), "Yield 1988-89 / 1989-90", fontsize=9)

    crop_rows = [
        ("Wheat", "48210", "1820", "1905"),
        ("Rice", "31560", "1440", "1512"),
        ("Cotton", "12870", "610", "588"),
    ]
    for i, (crop, area, y1, y2) in enumerate(crop_rows, start=1):
        yy = y0 + i * row_h + 18
        page.insert_text((x0 + 8, yy), crop, fontsize=9)
        page.insert_text((x0 + col_w + 8, yy), area, fontsize=9)
        page.insert_text((x0 + 2 * col_w + 8, yy), y1, fontsize=9)
        page.insert_text((x0 + 3 * col_w + 8, yy), y2, fontsize=9)

    page.insert_text((290, 800), "2", fontsize=10)


def main() -> None:
    hindi_font = _find_devanagari_font()
    if not hindi_font:
        print("Warning: NotoSansDevanagari-Regular.ttf not found — generating Latin-only content.")
        print("Run: bash backend/scripts/fetch_fonts.sh")

    doc = fitz.open()
    build_page_1(doc, hindi_font)
    build_page_2(doc, hindi_font)
    doc.save(OUTPUT_PATH)
    doc.close()
    print(f"Wrote {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes, 2 pages)")


if __name__ == "__main__":
    main()
