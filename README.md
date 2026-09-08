# Document Clean & Reconstruct AI

Turn scanned PDFs/images (old government reports, mixed Hindi/English, tables,
skewed/noisy scans) into **clean, searchable, editable** documents while
preserving the **original page layout** — not a plain "OCR to text" dump.

```
SCANNED DOCUMENT → IMAGE CLEANING → OCR → LAYOUT DETECTION → TABLE DETECTION
  → TEXT/TABLE/IMAGE POSITION EXTRACTION → DOCUMENT STRUCTURE JSON
  → LAYOUT RECONSTRUCTION → CLEAN PDF → EDITABLE DOCX
```

## 1. Architecture

```
                         ┌────────────────────┐
                         │   React Frontend    │  Upload / Before-After / Editor
                         │  (Vite + TS + TW)   │  progress via WebSocket
                         └──────────┬──────────┘
                                    │ REST + WS
                         ┌──────────▼──────────┐
                         │      FastAPI          │  Auth-ready API, job orchestration
                         │  (Pydantic, SQLAlchemy)│
                         └──────────┬──────────┘
                     enqueue job    │        read status/results
                         ┌──────────▼──────────┐        ┌──────────────┐
                         │        Redis          │◄──────►  PostgreSQL   │
                         │  (broker + pub/sub)   │        │  (metadata)   │
                         └──────────┬──────────┘        └──────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Celery Workers      │  page-parallel pipeline
                         └──────────┬──────────┘
                                    │
      ┌─────────────────────────────────────────────────────────────────┐
      │ Document Processing Pipeline (per page, streamed, not all-in-RAM) │
      │                                                                     │
      │ 1 Ingest → 2 Render (PyMuPDF) → 3 Preprocess (OpenCV: deskew,       │
      │ dewarp, denoise, contrast, border removal) → 4 OCR (pluggable      │
      │ OCRProvider: PaddleOCR default / Tesseract / NVIDIA VLM / Ollama)  │
      │ → 5 Layout detection (heading/paragraph/table/image/header/footer/ │
      │ page-number classification from block geometry + text features)   │
      │ → 6 Table detection & structure (OpenCV line/grid detection +     │
      │ PP-Structure when available, cell/row/col/span extraction)        │
      │ → 7 Document JSON (canonical IR, normalized 0..1 coordinates)     │
      │ → 8 Reconstruction: PDF renderer (PyMuPDF, absolute positioning,  │
      │ vector tables) + searchable-PDF (image + invisible text layer) +  │
      │ DOCX exporter (python-docx, real tables)                          │
      │ → 9 Quality validation (SSIM / pixel-diff original vs recon)      │
      └─────────────────────────────────────────────────────────────────┘
```

Storage is a pluggable `StorageBackend` (local filesystem in dev, S3-compatible
in production) so nothing in the pipeline talks to the filesystem directly.

Everything downstream of OCR works off the **Document JSON** (canonical
intermediate representation) — never off raw OCR text — so reconstruction,
editing, PDF export and DOCX export all read/write the same structure.

## 2. Folder structure

```
document-clean-reconstruct/
├── frontend/                  React + TS + Vite + Tailwind SPA
│   └── src/{components,pages,features,hooks,store,services,types,utils}
├── backend/
│   ├── app/
│   │   ├── api/v1/            versioned FastAPI routers
│   │   ├── core/              config, logging, security, celery app
│   │   ├── db/                SQLAlchemy session/base
│   │   ├── models/            ORM models (Document, Page, OCRBlock, ...)
│   │   ├── schemas/           Pydantic schemas incl. Document JSON IR
│   │   ├── services/          storage backend, document service, ws hub
│   │   ├── workers/           Celery tasks = pipeline stages
│   │   ├── ocr/               OCRProvider + Paddle/Tesseract/NVIDIA/Ollama
│   │   ├── vision/            OpenCV preprocessing pipeline
│   │   ├── layout/             layout block detector
│   │   ├── tables/             TableStructureEngine (opencv + paddle)
│   │   ├── reconstruction/    PDF renderer, searchable PDF, coordinate mapper
│   │   ├── exporters/         DOCX exporter
│   │   └── utils/             validation, file safety, id generation
│   ├── tests/{unit,integration}
│   └── main.py
├── storage/{original,pages,processed,ocr,output,tmp}
├── docker/{backend,frontend,nginx}
├── docker-compose.yml
├── .env.example
└── Makefile
```

## 3. Installation

### Docker (recommended — everything: db, redis, backend, worker, frontend, nginx)

```bash
cp .env.example .env
bash backend/scripts/fetch_fonts.sh   # once — see script header for why this is separate from the Dockerfile
docker compose up -d --build
# frontend:  http://localhost:5173  (or http://localhost via nginx)
# API docs:  http://localhost:8000/docs
```

### Local development (no Docker)

```bash
# Backend
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env
alembic upgrade head
uvicorn main:app --reload

# Worker (separate shell)
celery -A app.core.celery_app worker -l info -P solo   # -P solo on Windows

# Frontend (separate shell)
cd frontend
npm install
npm run dev
```

## 4. Environment variables

See [.env.example](.env.example). Key ones:

| Variable | Purpose | Required |
|---|---|---|
| `DATABASE_URL` | PostgreSQL DSN | yes |
| `REDIS_URL` | Celery broker/backend + pub-sub for progress | yes |
| `STORAGE_BACKEND` | `local` or `s3` | yes (defaults `local`) |
| `OCR_PROVIDER` | `paddleocr` \| `tesseract` \| `nvidia` \| `ollama` | yes (defaults `paddleocr`) |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL` | optional VLM OCR/validation | no |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | optional local LLM validation | no |
| `TESSERACT_PATH` | optional fallback OCR binary | no |
| `DEFAULT_DPI` | page render DPI (150/200/300/400/600) | no (300) |
| `PREPROCESS_PROFILE` | `FAST`\|`BALANCED`\|`HIGH_QUALITY` | no (BALANCED) |

## 5. Docker commands

```bash
docker compose up -d --build     # start everything
docker compose logs -f worker    # tail pipeline logs
docker compose exec backend alembic upgrade head
docker compose down -v           # stop + wipe volumes
```

## 6. OCR setup

- **PaddleOCR (default, local, no API key)** — installed via `requirements.txt`;
  first run downloads detection/recognition/classification models per
  language in `PADDLE_OCR_LANGS` (default `en,hi`) into a persisted volume
  (`paddleocr_models` in `docker-compose.yml`), so it's a one-time cost.
  Each language runs its own detect+recognize pass over the full page; the
  results are merged by bounding-box overlap, keeping the higher-confidence
  reading where two passes detected the same region. Inference runs in an
  isolated subprocess per document job (`app/ocr/subprocess_runner.py`) so a
  native engine crash falls back to Tesseract instead of taking the whole
  worker down — see Known limitations below.
- **Tesseract (fallback)** — install `tesseract-ocr` + `tesseract-ocr-hin` on the
  image (already in `docker/backend/Dockerfile`); set `TESSERACT_PATH` if not on `PATH`.
- **NVIDIA VLM (optional)** — set `NVIDIA_API_KEY` + `NVIDIA_MODEL`; used only if
  `OCR_PROVIDER=nvidia` or explicitly selected per-job. Absence never breaks the app.
- **Ollama (optional)** — set `OLLAMA_BASE_URL`/`OLLAMA_MODEL`; used for block-level
  layout/structure *suggestions* only, never bulk rewriting.

## 7. API documentation

Interactive OpenAPI docs at `/docs` once the backend is running. Endpoint list
in [docs/API.md](docs/API.md).

## 8. Testing

```bash
cd backend && pytest -q
cd frontend && npm run test
```

## 9. Known limitations

Validated end-to-end against the live Docker stack (real upload → PaddleOCR →
layout → table detection → reconstruction → PDF/DOCX download), which is how
several of these were actually found:

- **PaddlePaddle CPU inference can hard-crash (SIGILL) on some hosts.** On
  certain CPU/virtualization combinations (observed under Docker Desktop's
  WSL2 backend) the prebuilt `paddlepaddle` wheel executes an instruction the
  host doesn't actually support, terminating the process with a signal
  `try/except` cannot catch. Mitigations already in place: raw OCR inference
  runs in an isolated, crash-detectable subprocess (`app/ocr/subprocess_runner.py`)
  that falls back to Tesseract in-process if the primary engine dies; the
  optional PP-Structure table model — which showed the same crash — is **off
  by default** (`ENABLE_PP_STRUCTURE=false`) for this reason. If PaddleOCR
  itself is unstable in your environment, set `OCR_PROVIDER=tesseract`.
- **180°-upside-down scans are not auto-corrected.** Coarse rotation
  detection reliably catches 90°/270° (sideways) pages, but a variance-based
  projection profile is mathematically unable to distinguish 0° from 180°
  (reversing a row order doesn't change its variance) — guessing on that
  toss-up used to flip ~50% of normal upright pages, which was worse than
  not guessing, so it now only rotates when there's a clear (90/270) signal.
  A reliable 180° check needs an OCR-confidence comparison, not a pixel
  heuristic; not yet implemented.
- **Mixed Hindi+English OCR can fragment a paragraph into more, smaller
  blocks than the source layout.** Running separate PaddleOCR passes per
  language and merging by bounding-box overlap sometimes leaves a few
  near-duplicate or split detections for the same line (the two models
  segment words slightly differently) — text is never lost or duplicated in
  content, but layout grouping can be choppier than a single-pass OCR would
  produce. Reviewable/editable via the OCR confidence view and block editor.
- Table structure recovery uses OpenCV grid-line detection + an OCR-alignment
  fallback for borderless tables; the fallback requires ≥4 rows and ≥3
  recurring column positions before it will call something a table
  (tightened after it initially mistook a short wrapped paragraph for one),
  so a genuinely borderless table with fewer than 4 rows may need manual
  correction in the editor instead of auto-detection.
- DOCX export approximates absolute positions using tables/frames — it is not
  pixel-identical to the PDF reconstruction (Word has no free-form canvas).
- Dewarping (curved-page correction) uses a page-boundary + perspective model;
  it is not a full non-rigid document-unwarping network.
- NVIDIA/Ollama are optional enhancers; when unset the app runs fully offline
  on PaddleOCR + OpenCV heuristics.

## 10. Future improvements

- Swap heuristic layout/table detection for a trained layout model (e.g.
  LayoutLMv3 / PP-StructureV2 checkpoints) behind the existing pluggable
  interfaces (`LayoutDetector`, `TableStructureEngine`) with no API changes.
- Multi-tenant auth (the API is auth-ready: routers accept a
  `get_current_user` dependency stub today).
- S3 lifecycle policies + signed URL expiry tuning for production storage.

See [docs/PHASES.md](docs/PHASES.md) for the phase-by-phase build log.
