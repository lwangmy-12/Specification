# ── ODOT BDM + AASHTO LRFD Specification Assistant ───────────────────────────
# Python 3.11 slim — smaller image, still has build tools for faiss-cpu

FROM python:3.11-slim

# System deps needed by faiss-cpu and pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY ingest.py rag.py main.py ./
COPY static/ ./static/

# The doc/ folder and vector_store/ are mounted at runtime via docker-compose
# so they are NOT copied into the image.

EXPOSE 8000

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
