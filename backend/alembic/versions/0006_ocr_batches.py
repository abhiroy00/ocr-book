"""ocr_batches + ocr_batch_items (multiple-file OCR scheduling state)

Purely additive: two new tables, no change to documents/processing_jobs.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ocr_batches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ocr_provider", sa.String(32), nullable=False),
        sa.Column("dpi", sa.Integer, nullable=False),
        sa.Column("preprocess_profile", sa.String(32), nullable=False),
        sa.Column("uploaded_by", sa.String(256), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "ocr_batch_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("batch_id", sa.String(36), sa.ForeignKey("ocr_batches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("processing_job_id", sa.String(36), sa.ForeignKey("processing_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("priority_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("celery_task_id", sa.String(64), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ocr_batch_items_batch_id", "ocr_batch_items", ["batch_id"])
    op.create_index("ix_ocr_batch_items_document_id", "ocr_batch_items", ["document_id"])
    op.create_index("ix_ocr_batch_items_batch_id_state", "ocr_batch_items", ["batch_id", "state"])


def downgrade() -> None:
    op.drop_table("ocr_batch_items")
    op.drop_table("ocr_batches")
