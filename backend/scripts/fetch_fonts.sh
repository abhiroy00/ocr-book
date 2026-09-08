#!/usr/bin/env bash
# Fetches the Unicode fonts (Hindi Devanagari + Latin) used by PDF/DOCX
# reconstruction (see app/reconstruction/fonts.py).
#
# Why this exists as a separate script instead of relying solely on the
# Dockerfile's own `curl` step: docker-compose.yml bind-mounts
# `./backend:/app/backend` into the backend/worker containers for
# live-reload development, which shadows anything the image baked in at
# that path — including fonts fetched during `docker build`. Run this once
# on the HOST so the fonts live in the bind-mounted directory itself.
#
# Usage: bash backend/scripts/fetch_fonts.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FONTS_DIR="$SCRIPT_DIR/../app/reconstruction/fonts"
mkdir -p "$FONTS_DIR"

echo "Fetching NotoSans-Regular.ttf..."
curl -fsSL -o "$FONTS_DIR/NotoSans-Regular.ttf" \
  "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"

echo "Fetching NotoSansDevanagari-Regular.ttf..."
curl -fsSL -o "$FONTS_DIR/NotoSansDevanagari-Regular.ttf" \
  "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"

echo "Done. Fonts are in $FONTS_DIR"
echo "Restart the backend/worker containers (or they'll pick it up on next request)."
