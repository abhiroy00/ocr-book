"""accession_records + documents.document_hash

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("document_hash", sa.String(64), nullable=True))
    op.create_index("ix_documents_document_hash", "documents", ["document_hash"])

    op.create_table(
        "accession_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("accession_number", sa.String(64), nullable=False),
        sa.Column("book_name", sa.Text, nullable=False),
        sa.Column("creator", sa.Text, nullable=True),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column("year_of_publication", sa.String(32), nullable=True),
        sa.Column("total_pages", sa.Integer, nullable=False),
        sa.Column("record_type", sa.String(64), nullable=True),
        sa.Column("record_date", sa.Date, nullable=False),
        sa.Column("needs_review", sa.Boolean, nullable=False, default=False),
        sa.Column("extraction_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_accession_records_document_id", "accession_records", ["document_id"])
    op.create_unique_constraint("uq_accession_records_document_id", "accession_records", ["document_id"])
    op.create_unique_constraint("uq_accession_records_accession_number", "accession_records", ["accession_number"])
    op.create_index("ix_accession_records_accession_number", "accession_records", ["accession_number"])
    op.create_index("ix_accession_records_record_date", "accession_records", ["record_date"])


def downgrade() -> None:
    op.drop_table("accession_records")
    op.drop_index("ix_documents_document_hash", table_name="documents")
    op.drop_column("documents", "document_hash")
