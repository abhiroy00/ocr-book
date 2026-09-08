FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps:
#  - OpenCV runtime libs (libgl1, libglib2.0-0, ...)
#  - Tesseract + Hindi/English trained data (fallback OCR engine)
#  - curl (font fetching), fonts-noto* as an OS-level fallback too
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bundle Unicode fonts (Hindi Devanagari + Latin) for PDF/DOCX reconstruction
# (app/reconstruction/fonts.py). Network access is available at build time;
# if it's unreachable in an offline build, the app degrades gracefully to
# the base-14 Latin-only PDF font (see fonts.py) rather than failing.
RUN mkdir -p app/reconstruction/fonts && \
    ( curl -fsSL -o app/reconstruction/fonts/NotoSans-Regular.ttf \
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf" || true ) && \
    ( curl -fsSL -o app/reconstruction/fonts/NotoSansDevanagari-Regular.ttf \
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf" || true )

COPY backend/ .

RUN mkdir -p /app/storage/original /app/storage/pages /app/storage/processed /app/storage/ocr /app/storage/output /app/storage/tmp

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
