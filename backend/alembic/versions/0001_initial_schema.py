"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

document_status = sa.Enum(
    "UPLOADED", "QUEUED", "PROCESSING", "OCR_PROCESSING", "LAYOUT_PROCESSING",
    "TABLE_PROCESSING", "RECONSTRUCTING", "EXPORTING", "COMPLETED", "FAILED",
    name="document_status",
)
processing_stage = sa.Enum(
    "upload", "render", "preprocess", "ocr", "layout", "table", "document_json",
    "reconstruct", "export", "quality", "done", name="processing_stage",
)
ocr_provider_enum = sa.Enum("paddleocr", "tesseract", "nvidia", "ollama", name="ocr_provider_enum")
preprocess_profile_enum = sa.Enum("FAST", "BALANCED", "HIGH_QUALITY", name="preprocess_profile_enum")
layout_block_type = sa.Enum(
    "title", "heading", "subheading", "paragraph", "table", "image", "chart",
    "signature", "stamp", "handwritten", "header", "footer", "page_number",
    "footnote", "horizontal_line", "vertical_line", name="layout_block_type",
)
table_detection_method = sa.Enum("opencv_lines", "paddle_structure", "ocr_fallback", name="table_detection_method")
export_type = sa.Enum("clean_pdf", "searchable_pdf", "reconstructed_pdf", "docx", name="export_type")
cell_align_h = sa.Enum("LEFT", "CENTER", "RIGHT", name="cell_align_h")
cell_align_v = sa.Enum("TOP", "MIDDLE", "BOTTOM", name="cell_align_v")


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("file_extension", sa.String(16), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False),
        sa.Column("page_count", sa.Integer, nullable=False, default=0),
        sa.Column("status", document_status, nullable=False),
        sa.Column("ocr_provider", ocr_provider_enum, nullable=False),
        sa.Column("dpi", sa.Integer, nullable=False),
        sa.Column("preprocess_profile", preprocess_profile_enum, nullable=False),
        sa.Column("storage_original_path", sa.String(1024), nullable=False),
        sa.Column("is_deleted", sa.Boolean, nullable=False, default=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("uploaded_by", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_pages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("dpi", sa.Integer, nullable=False),
        sa.Column("rotation", sa.Float, nullable=False, default=0.0),
        sa.Column("original_image_path", sa.String(1024), nullable=False),
        sa.Column("processed_image_path", sa.String(1024), nullable=True),
        sa.Column("ocr_status", sa.String(32), nullable=False, default="pending"),
        sa.Column("layout_status", sa.String(32), nullable=False, default="pending"),
        sa.Column("table_status", sa.String(32), nullable=False, default="pending"),
        sa.Column("ocr_confidence_avg", sa.Float, nullable=True),
        sa.Column("layout_confidence_avg", sa.Float, nullable=True),
        sa.Column("table_confidence_avg", sa.Float, nullable=True),
        sa.Column("visual_similarity_score", sa.Float, nullable=True),
        sa.Column("document_json", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_document_pages_document_id", "document_pages", ["document_id"])

    op.create_table(
        "ocr_blocks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("page_id", sa.String(36), sa.ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block_id", sa.String(64), nullable=False),
        sa.Column("line_id", sa.String(64), nullable=False),
        sa.Column("word_id", sa.String(64), nullable=True),
        sa.Column("text", sa.String(2048), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("bbox", sa.JSON, nullable=False),
        sa.Column("polygon", sa.JSON, nullable=False),
        sa.Column("language", sa.String(16), nullable=False, default="und"),
        sa.Column("font_size_estimate", sa.Float, nullable=True),
        sa.Column("reading_order_index", sa.Integer, nullable=False, default=0),
        sa.Column("corrected_text", sa.String(2048), nullable=True),
        sa.Column("is_reviewed", sa.Boolean, nullable=False, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ocr_blocks_document_id", "ocr_blocks", ["document_id"])
    op.create_index("ix_ocr_blocks_page_id", "ocr_blocks", ["page_id"])

    op.create_table(
        "layout_blocks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("page_id", sa.String(36), sa.ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block_type", layout_block_type, nullable=False),
        sa.Column("bbox", sa.JSON, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("z_order", sa.Integer, nullable=False, default=0),
        sa.Column("content", sa.JSON, nullable=False),
        sa.Column("style", sa.JSON, nullable=False),
        sa.Column("is_edited", sa.Boolean, nullable=False, default=False),
        sa.Column("edited_text", sa.String(8192), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_layout_blocks_document_id", "layout_blocks", ["document_id"])
    op.create_index("ix_layout_blocks_page_id", "layout_blocks", ["page_id"])

    op.create_table(
        "tables",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("page_id", sa.String(36), sa.ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("layout_block_id", sa.String(36), sa.ForeignKey("layout_blocks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("bbox", sa.JSON, nullable=False),
        sa.Column("n_rows", sa.Integer, nullable=False, default=0),
        sa.Column("n_cols", sa.Integer, nullable=False, default=0),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("detection_method", table_detection_method, nullable=False),
        sa.Column("column_widths", sa.JSON, nullable=False),
        sa.Column("row_heights", sa.JSON, nullable=False),
        sa.Column("border_style", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_tables_document_id", "tables", ["document_id"])
    op.create_index("ix_tables_page_id", "tables", ["page_id"])

    op.create_table(
        "table_cells",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("table_id", sa.String(36), sa.ForeignKey("tables.id", ondelete="CASCADE"), nullable=False),
        sa.Column("row", sa.Integer, nullable=False),
        sa.Column("column", sa.Integer, nullable=False),
        sa.Column("rowspan", sa.Integer, nullable=False, default=1),
        sa.Column("colspan", sa.Integer, nullable=False, default=1),
        sa.Column("text", sa.String(2048), nullable=False, default=""),
        sa.Column("confidence", sa.Float, nullable=False, default=0.0),
        sa.Column("bbox", sa.JSON, nullable=False),
        sa.Column("align_h", cell_align_h, nullable=False),
        sa.Column("align_v", cell_align_v, nullable=False),
        sa.Column("is_header", sa.Boolean, nullable=False, default=False),
        sa.Column("is_edited", sa.Boolean, nullable=False, default=False),
        sa.Column("edited_text", sa.String(2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_table_cells_table_id", "table_cells", ["table_id"])

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("status", document_status, nullable=False),
        sa.Column("stage", processing_stage, nullable=False),
        sa.Column("progress_percent", sa.Integer, nullable=False, default=0),
        sa.Column("celery_task_id", sa.String(64), nullable=True),
        sa.Column("ocr_provider", ocr_provider_enum, nullable=False),
        sa.Column("dpi", sa.Integer, nullable=False),
        sa.Column("preprocess_profile", preprocess_profile_enum, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_processing_jobs_document_id", "processing_jobs", ["document_id"])

    op.create_table(
        "export_files",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("export_type", export_type, nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_export_files_document_id", "export_files", ["document_id"])


def downgrade() -> None:
    op.drop_table("export_files")
    op.drop_table("processing_jobs")
    op.drop_table("table_cells")
    op.drop_table("tables")
    op.drop_table("layout_blocks")
    op.drop_table("ocr_blocks")
    op.drop_table("document_pages")
    op.drop_table("documents")

    bind = op.get_bind()
    for enum in (
        cell_align_v, cell_align_h, export_type, table_detection_method, layout_block_type,
        preprocess_profile_enum, ocr_provider_enum, processing_stage, document_status,
    ):
        enum.drop(bind, checkfirst=True)
