#!/bin/bash
# entrypoint.sh — Check for actual index files (not just directories)

set -e

echo "============================================================"
echo "  ODOT BDM + AASHTO LRFD Specification Assistant"
echo "============================================================"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

# Check for the actual FAISS index files, not just the directory.
# A directory can exist from a partial/failed previous run without the index inside.
BDM_INDEX="/app/vector_store/bdm/index.faiss"
AASHTO_INDEX="/app/vector_store/aashto/index.faiss"

if [ ! -f "$BDM_INDEX" ] || [ ! -f "$AASHTO_INDEX" ]; then
    echo ""
    echo "[+] Vector stores incomplete — running ingestion from /app/doc/ ..."
    echo "    This may take several minutes on first run."
    echo ""
    python ingest.py
    echo ""
    echo "[✓] Ingestion complete."
else
    echo "[✓] Vector stores ready — skipping ingestion."
fi

echo ""
echo "[+] Starting API server on http://0.0.0.0:8000 ..."
echo ""

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
