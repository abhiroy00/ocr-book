"""
Coordinate primitives shared by OCR, layout, table and the Document JSON IR.

Two parallel representations are always kept in sync:
  - pixel bbox  : coordinates in the *original page* pixel space at the
                   page's render DPI (this is what OCR engines emit).
  - normalized  : coordinates in [0, 1] relative to page width/height.

Normalized coordinates are the DPI-independent source of truth (see
app/utils/coordinates.py); pixel/PDF-point/EMU (docx) values are derived from
them on demand so a page rendered at 150 DPI and one rendered at 600 DPI
reconstruct to the same physical layout.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in page pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_norm(self, page_width: float, page_height: float) -> "NormBBox":
        return NormBBox(
            x1=self.x1 / page_width,
            y1=self.y1 / page_height,
            x2=self.x2 / page_width,
            y2=self.y2 / page_height,
        )


class NormBBox(BaseModel):
    """Axis-aligned bounding box normalized to [0, 1] of page width/height."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixels(self, page_width: float, page_height: float) -> BBox:
        return BBox(
            x1=self.x1 * page_width,
            y1=self.y1 * page_height,
            x2=self.x2 * page_width,
            y2=self.y2 * page_height,
        )


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[Point] = Field(default_factory=list)

    @classmethod
    def from_xy_list(cls, pts: list[list[float]] | list[tuple[float, float]]) -> "Polygon":
        return cls(points=[Point(x=p[0], y=p[1]) for p in pts])

    def to_xy_list(self) -> list[list[float]]:
        return [[p.x, p.y] for p in self.points]

    def bbox(self) -> BBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
