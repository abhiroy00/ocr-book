"""document_pages unique (document_id, page_number)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-13

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # DB-level backstop for the idempotency `upsert_page` already provides
    # in application code (see document_page.py's model docstring): a raw
    # race inserting the same document_id+page_number twice now fails
    # loudly with an IntegrityError instead of silently duplicating a row.
    op.create_unique_constraint(
        "uq_document_pages_document_id_page_number", "document_pages", ["document_id", "page_number"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_document_pages_document_id_page_number", "document_pages", type_="unique")
