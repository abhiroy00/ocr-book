"""accession_records.author + accession_records.publisher

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("accession_records", sa.Column("author", sa.Text, nullable=True))
    op.add_column("accession_records", sa.Column("publisher", sa.Text, nullable=True))
    # Backfill from the existing combined `creator` column so pre-existing
    # rows aren't left blank in the Master Accession Register's new
    # Author/Publisher columns -- `creator` was always EITHER a personal
    # author or a publisher/issuing-body line (see
    # `accession_extractor._extract_creator`'s own docstring), never a
    # combination of both, so there is no ambiguity to resolve here; a
    # later `POST /accession-records/refresh` re-reads the real split
    # from stored OCR data for a more precise value.
    op.execute("UPDATE accession_records SET publisher = creator WHERE creator IS NOT NULL")


def downgrade() -> None:
    op.drop_column("accession_records", "publisher")
    op.drop_column("accession_records", "author")
