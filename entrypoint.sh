#!/bin/bash
# entrypoint.sh — Container startup script
# Runs ingestion automatically if vector stores are missing, then starts the API server.

set -e

echo "============================================================"
echo "  ODOT BDM + AASHTO LRFD Specification Assistant"
echo "============================================================"

# Validate required environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Set it in your .env file or pass it with -e OPENAI_API_KEY=sk-..."
    echo ""
    exit 1
fi

# Build vector stores if they don't already exist
if [ ! -d "/app/vector_store/bdm" ] || [ ! -d "/app/vector_store/aashto" ]; then
    echo ""
    echo "[+] Vector stores not found — running ingestion from /app/doc/ ..."
    echo "    This may take several minutes on first run."
    echo ""
    python ingest.py
    echo ""
    echo "[✓] Ingestion complete."
else
    echo "[✓] Vector stores found — skipping ingestion."
fi

echo ""
echo "[+] Starting API server on http://0.0.0.0:8000 ..."
echo ""

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
