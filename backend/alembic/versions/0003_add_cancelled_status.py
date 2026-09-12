"""add cancelled document status

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'CANCELLED'")


def downgrade() -> None:
    # Postgres has no ALTER TYPE ... DROP VALUE; removing an enum label
    # requires rebuilding the type, which isn't worth the risk/complexity
    # for a downgrade path. Leaving 'CANCELLED' in place on downgrade is
    # safe (an unused enum label is harmless).
    pass
