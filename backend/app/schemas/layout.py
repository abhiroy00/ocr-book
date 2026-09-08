from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox


class LayoutBlockRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_id: str
    block_type: LayoutBlockType
    bbox: BBox
    confidence: float
    z_order: int
    content: dict
    style: dict
    is_edited: bool
    edited_text: Optional[str] = None


class LayoutBlockUpdate(BaseModel):
    text: Optional[str] = None
    bbox: Optional[BBox] = None
    style: Optional[dict] = None
    block_type: Optional[LayoutBlockType] = None
