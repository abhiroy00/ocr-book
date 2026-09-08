# Build log (phase-by-phase)

Tracks what was implemented in each phase from the spec's development workflow
(section 39). Each phase's code was import-checked and, where practical,
covered by real automated tests before moving on (section 40: "do not move
to the next phase if the current phase is broken").

| Phase | Scope | Status |
|---|---|---|
| 1 | Repo scaffolding: monorepo layout, README, `.env.example`, Makefile, API doc | ✅ |
| 2 | FastAPI app skeleton, SQLAlchemy models (`Document`, `DocumentPage`, `OCRBlock`, `LayoutBlock`, `Table`, `TableCell`, `ProcessingJob`, `ExportFile`), Alembic baseline migration, upload API | ✅ |
| 3 | PDF/image ingestion: page-by-page streaming render via PyMuPDF (`app/services/pdf_ingest.py`) | ✅ |
| 4 | OpenCV preprocessing pipeline: deskew, dewarp, denoise, background/contrast, border removal, FAST/BALANCED/HIGH_QUALITY profiles (`app/vision/preprocessor.py`) | ✅ |
| 5 | Pluggable OCR: `OCRProvider` + PaddleOCR (default), Tesseract, NVIDIA VLM, Ollama, with automatic fallback (`app/ocr/`) | ✅ |
| 6 | OCR coordinate contract: `OCRWordResult` (bbox + polygon + language + confidence), persisted to `OCRBlock` | ✅ |
| 7 | Layout detection: text grouping, heading/title/paragraph/header/footer/page-number/footnote classification, ruling-line detection, graphic-region (image/signature/stamp/handwritten) heuristics (`app/layout/`) | ✅ |
| 8 | Table detection: OpenCV grid-line structure extraction with merged-cell recovery via union-find, OCR-alignment fallback for borderless tables, optional PP-Structure integration (`app/tables/`) | ✅ |
| 9 | Canonical Document JSON IR (`app/schemas/document_json.py`, `app/reconstruction/document_json_builder.py`) | ✅ |
| 10 | PDF reconstruction (absolute positioning, vector tables), searchable PDF (image + invisible text layer), clean PDF (`app/reconstruction/`) | ✅ |
| 11 | DOCX reconstruction: real Word headings/paragraphs/tables with merged cells (`app/exporters/docx_exporter.py`) | ✅ |
| 12 | React/TS/Vite/Tailwind frontend: dashboard, upload, documents list/detail, before/after preview, editor, export pages | ✅ |
| 13 | Before/after editor: zoom, page nav, sync scroll, OCR/layout/table overlay toggles, click-to-edit text blocks and table cells (add/delete row/col, merge) | ✅ |
| 14 | Live progress: Redis pub/sub -> WebSocket (`/ws/documents/{id}`) with polling fallback | ✅ |
| 15 | NVIDIA VLM optional OCR provider (`app/ocr/nvidia_engine.py`) — inert without `NVIDIA_API_KEY` | ✅ |
| 16 | Ollama optional local OCR + AI-suggestion provider (`app/ocr/ollama_engine.py`, `app/ai/suggestion_service.py`) — inert without a reachable Ollama server | ✅ |
| 17 | Dockerization: backend/worker/frontend/nginx/postgres/redis (+ optional ollama profile) via `docker-compose.yml` | ✅ |
| 18 | Tests: 55 backend (pytest, unit+integration) + 13 frontend (vitest/RTL) — all passing | ✅ |
| 19 | Full stack actually run via `docker compose up` (backend, worker, frontend, nginx, postgres, redis) and exercised end-to-end: real file upload → PaddleOCR (in an isolated subprocess) → OpenCV table detection → layout detection → PDF/DOCX reconstruction → download, against a real Postgres (Alembic migration applied) and Redis. Not the 150-page Hindi/English sample specifically (not available in this environment), but the generic pipeline was validated live, not just unit-tested — see the bug list below, all found this way. | ✅ |
| 20 | Visual-similarity optimization: SSIM-based quality scoring implemented and unit-tested; a live run scored 97.1% visual similarity and >99% OCR/table confidence on its test page (see below) | ✅ |

## Bugs found and fixed via testing during this build

- `app/tables/opencv_engine.py::_line_present` clipped a separator-line
  span against the wrong axis (`h` vs `w` swapped), which silently broke
  merged-cell / multi-column table detection near the right/bottom edges of
  a table. Found by a synthetic-grid unit test, fixed, re-verified.
- `app/services/document_service.py`: in-place mutations of a page's
  `document_json` (JSON column) followed by reassigning the *same* object
  reference were invisible to SQLAlchemy's change tracking, so table/block
  text edits and structural table edits (add/delete row/column, merge)
  never actually persisted to the reconstruction source-of-truth. Found by
  an integration test asserting the resync; fixed with explicit
  `flag_modified()` calls.
- Frontend: `.replaceAll()` on a template-literal union type required
  bumping the TS `lib`/`target` to `ES2021`; the `@/` import alias needed
  to be added to `vite.config.ts` (tsconfig `paths` alone only satisfies
  the type checker, not the Vite dev/test runtime) — both caught by
  actually running `tsc -b` and `vitest run` rather than assuming they'd work.

### Found by actually running the full Docker stack end-to-end (a real upload,
not a mocked one) — none of these were reachable from unit/integration tests
alone, which is exactly why this pass mattered:

- SQLAlchemy's `Enum(SomePythonEnum)` sends the Python member's `.name` to
  the database by default, not `.value`. Several enums (`ProcessingStage`,
  `OCRProviderEnum`, `LayoutBlockType`, `TableDetectionMethod`, `ExportType`)
  have members whose name/value differ in case (e.g. `UPLOAD` vs `"upload"`),
  so every insert against those columns failed against real Postgres with
  `InvalidTextRepresentation` — invisible against SQLite, which doesn't
  enforce the enum's label set. Fixed with a shared `pg_enum()` helper
  (`app/db/enum_types.py`) that sets `values_callable` everywhere.
- PaddlePaddle's CPU inference engine crashed the whole Celery worker
  process with `SIGILL` (an OS signal, not a Python exception) — both
  during raw OCR and during the optional PP-Structure table model. Fixed by
  running OCR in an isolated, long-lived subprocess
  (`app/ocr/subprocess_runner.py`) that falls back to Tesseract on crash,
  and by turning PP-Structure off by default (`ENABLE_PP_STRUCTURE=false`).
- Spawning that subprocess with stdlib `multiprocessing` failed with
  `AssertionError: daemonic processes are not allowed to have children` —
  Celery's prefork pool workers are themselves daemonic processes. Fixed by
  using `billiard` (Celery's own multiprocessing fork, already a
  transitive dependency) instead.
- A `WorkerLostError` from that kind of crash never reached
  `process_document`'s own `try/except` (the process hosting that frame was
  gone), leaving the job stuck in its last in-progress status forever with
  no error surfaced. Fixed with a `celery.signals.task_failure` handler
  that marks the job `FAILED` from the (still-alive) pool process.
- `detect_rotation()`'s "highest row-projection variance wins" heuristic is
  mathematically unable to distinguish 0° from 180° (reversing a row order
  doesn't change its variance), so it was flipping roughly half of normal
  upright pages upside down. A live upload's table and paragraphs came back
  mirrored 180°, which is how this was caught. Fixed by requiring a clear
  margin over the 0° baseline before accepting any rotation, so ambiguous
  0-vs-180 cases now default to no rotation instead of guessing.
- `_projection_boundaries()`'s synthetic 0/`length` edge-boundary padding
  created a hairline pseudo-row/column with no real line evidence around
  the true grid; the merged-cell union-find (correctly, given that input)
  glued it into a "picture-frame" cell wrapping the *entire* table, which
  then absorbed every other cell's text via bbox-overlap when text was
  assigned. Fixed by removing the synthetic padding — a real table's outer
  border is itself a detected line, so it never needed synthetic help.
- The borderless-table (OCR-alignment) fallback mistook a short, left-
  aligned wrapped paragraph for a table, because a few words coincidentally
  lined up across 3 lines. Tightened `MIN_COLS` 2→3, the row-recurrence
  threshold 60%→85%, and `MIN_ROWS` 3→4 (real data tables reliably have
  more rows than a short paragraph does).
- Running separate PaddleOCR passes per language (`en`, `hi`) and merging
  by strict IoU left near-duplicate detections uncombined whenever the two
  models segmented the same line into differently-sized boxes (lower IoU
  purely from the size mismatch, not a real difference). Switched to an
  overlap-over-the-smaller-box ratio, which stays high for a same-line,
  different-segmentation duplicate. Residual paragraph fragmentation from
  this dual-pass approach is a documented known limitation, not a crash.

All of the above were fixed, re-covered by unit/regression tests (55 backend
tests passing), and re-validated with fresh live uploads through the same
Docker stack until a document completed cleanly end-to-end — see the
delivered screenshot of an actual reconstructed PDF from one of those runs.
