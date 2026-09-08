"""
Postgres ENUM helper.

SQLAlchemy's `Enum(SomePythonEnum)` sends the Python enum member's `.name`
to the database by default — NOT `.value`. Every enum in `app.models.enums`
is a `str, enum.Enum` whose `.value` is a lowercase/snake_case string
(matching what the Alembic migration declared as the Postgres ENUM's
labels), while several members' `.name` differs in case (e.g.
`ProcessingStage.UPLOAD` -> name "UPLOAD", value "upload"). Without
`values_callable`, SQLAlchemy tries to insert "UPLOAD" against an enum
type that only accepts "upload", failing with
`InvalidTextRepresentation`. Every enum column must go through this helper
instead of calling `sqlalchemy.Enum(...)` directly.
"""
from __future__ import annotations

from typing import Type

from sqlalchemy import Enum as SAEnum


def pg_enum(enum_cls: Type, name: str, **kwargs) -> SAEnum:
    return SAEnum(enum_cls, name=name, values_callable=lambda obj: [e.value for e in obj], **kwargs)
