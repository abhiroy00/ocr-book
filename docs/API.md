# API Reference

Base URL: `http://localhost:8000/api`. Interactive docs at `/docs` (Swagger) and `/redoc`.
All routes are versioned under `/api` (v1 implicit today; see `app/api/v1`).

## Documents

| Method | Path | Description |
|---|---|---|
| POST | `/documents/upload` | Upload a PDF/JPG/JPEG/PNG/WebP. Validates MIME, extension, size, corruption, encryption. Creates `Document` + `ProcessingJob` (status `UPLOADED`), returns `document_id`, `job_id`. |
| GET | `/documents` | List documents (paginated, filter by status). |
| GET | `/documents/{id}` | Document detail incl. status, page count, timestamps. |
| DELETE | `/documents/{id}` | Soft-delete + schedule storage cleanup. |
| POST | `/documents/{id}/process` | Enqueue (or re-enqueue) the processing pipeline with a chosen OCR provider / DPI / preprocessing profile. |
| GET | `/documents/{id}/progress` | Current stage + percent (polling fallback for the WS). |
| GET | `/documents/{id}/pages` | List of `DocumentPage` summaries. |
| GET | `/documents/{id}/pages/{page}` | Single page: image URLs (original/processed), dimensions, document JSON for that page. |
| GET | `/documents/{id}/ocr` | OCR blocks (optionally `?page=`). |
| GET | `/documents/{id}/layout` | Layout blocks (optionally `?page=`). |
| GET | `/documents/{id}/tables` | Tables + cells (optionally `?page=`). |
| PUT | `/documents/{id}/blocks/{block_id}` | Edit a layout/text block's content/bbox/style. |
| PUT | `/documents/{id}/tables/{table_id}` | Edit table-level properties (add/remove row/col handled here). |
| PUT | `/documents/{id}/cells/{cell_id}` | Edit one cell's text/span/alignment. |
| POST | `/documents/{id}/reconstruct` | Re-run reconstruction from the (possibly edited) Document JSON. |
| POST | `/documents/{id}/export/pdf` | Generate clean + searchable + reconstructed PDF. |
| POST | `/documents/{id}/export/docx` | Generate DOCX. |
| GET | `/documents/{id}/download/pdf` | Download latest reconstructed PDF. |
| GET | `/documents/{id}/download/docx` | Download latest DOCX. |
| GET | `/documents/{id}/quality` | Per-page OCR/layout/table confidence + SSIM similarity score. |

## WebSocket

`ws://.../ws/documents/{id}` — pushes `{stage, percent, page, message}` events as
the Celery pipeline progresses (backed by Redis pub/sub so any API replica can
relay worker progress).

## Status values

`UPLOADED → QUEUED → PROCESSING → OCR_PROCESSING → LAYOUT_PROCESSING →
TABLE_PROCESSING → RECONSTRUCTING → EXPORTING → COMPLETED | FAILED`
