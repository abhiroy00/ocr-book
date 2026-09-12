"""add xlsx export type

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-11

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres requires ADD VALUE to run outside a transaction block in
    # older versions; modern Alembic/psycopg handles this, but keep the
    # statement minimal and idempotent-safe with IF NOT EXISTS (PG 12+).
    op.execute("ALTER TYPE export_type ADD VALUE IF NOT EXISTS 'xlsx'")


def downgrade() -> None:
    # Postgres has no ALTER TYPE ... DROP VALUE; removing an enum label
    # requires rebuilding the type, which isn't worth the risk/complexity
    # for a downgrade path. Leaving 'xlsx' in place on downgrade is safe
    # (an unused enum label is harmless).
    pass
