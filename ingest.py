"""
ingest.py — Load PDFs and build FAISS vector stores

Rate limiting: OpenAI free/tier-1 accounts have 40,000 TPM limit.
AASHTO has ~7000 chunks; ingestion uses batches of 50 with 22s sleep
between batches to stay under the limit. Already-built stores are skipped.
"""

import os
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DOC_DIR   = ROOT / "doc"
STORE_DIR = ROOT / "vector_store"

BDM_PDF    = DOC_DIR / "BDM (July 2023).pdf"
AASHTO_PDF = DOC_DIR / "AASHTO LRFD 9th Edition 3.pdf"
BDM_STORE    = STORE_DIR / "bdm"
AASHTO_STORE = STORE_DIR / "aashto"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 250

# ── Rate-limit parameters ─────────────────────────────────────────────────────
# OpenAI tier-1: 40,000 TPM limit for text-embedding-3-small
# Each batch of 50 chunks ≈ 50 × 290 tokens = 14,500 tokens
# Sleep 22s between batches → ~14,500 / 22s ≈ 39,500 tokens/min (safe)
EMBED_BATCH   = 50   # chunks per embedding API call
BATCH_SLEEP   = 22   # seconds between batches

_SECTION_1000_RE = re.compile(r'\b1[0-9]{3}\b')


def _classify_bdm_chunk(text: str) -> dict:
    has_supplement = bool(_SECTION_1000_RE.search(text)) or any(
        kw in text for kw in ["ODOT Supplement", "in lieu of", "modify",
                               "supersede", "Section 1000", "1000."]
    )
    is_commentary = bool(re.search(r'\bC\d{3}', text))
    return {"is_supplement": has_supplement, "is_commentary": is_commentary}


def load_and_split(pdf_path: Path, source_tag: str) -> list:
    print(f"[+] Loading {pdf_path.name} ...")
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()
    for doc in pages:
        doc.metadata["source_tag"] = source_tag

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    if source_tag == "BDM":
        for chunk in chunks:
            chunk.metadata.update(_classify_bdm_chunk(chunk.page_content))

    print(f"    → {len(pages)} pages → {len(chunks)} chunks")
    return chunks


def build_store(chunks: list, store_path: Path, embeddings) -> None:
    """
    Build FAISS vector store in small batches with sleep between calls
    to respect OpenAI TPM rate limits. Skips if already built.
    """
    index_file = store_path / "index.faiss"
    if index_file.exists():
        print(f"[✓] {store_path.name} already built — skipping.")
        return

    store_path.mkdir(parents=True, exist_ok=True)
    total      = len(chunks)
    n_batches  = (total + EMBED_BATCH - 1) // EMBED_BATCH
    est_min    = round(n_batches * BATCH_SLEEP / 60)
    print(f"[+] Building {store_path.name}: {total} chunks "
          f"in {n_batches} batches (~{est_min} min) ...")

    db = None
    for i in range(0, total, EMBED_BATCH):
        batch     = chunks[i : i + EMBED_BATCH]
        batch_num = i // EMBED_BATCH + 1

        while True:
            try:
                batch_db = FAISS.from_documents(batch, embeddings)
                if db is None:
                    db = batch_db
                else:
                    db.merge_from(batch_db)
                print(f"    [{batch_num}/{n_batches}] "
                      f"{min(i + EMBED_BATCH, total)}/{total} chunks embedded")
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    print("    Rate limit hit — waiting 60s and retrying ...")
                    time.sleep(60)
                else:
                    raise

        # Sleep between batches (skip after the last one)
        if i + EMBED_BATCH < total:
            time.sleep(BATCH_SLEEP)

    db.save_local(str(store_path))
    print(f"    → Done. {db.index.ntotal} vectors stored.\n")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY first.")

    # chunk_size=50 limits each individual OpenAI API call to ~50 texts
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=50)

    # BDM first — skipped automatically if index.faiss already exists
    bdm_chunks = load_and_split(BDM_PDF, "BDM")
    build_store(bdm_chunks, BDM_STORE, embeddings)

    # AASHTO — rate-limited batching (~50 min on 40K TPM accounts)
    aashto_chunks = load_and_split(AASHTO_PDF, "AASHTO")
    build_store(aashto_chunks, AASHTO_STORE, embeddings)

    print("[✓] All vector stores built successfully.")


if __name__ == "__main__":
    main()
