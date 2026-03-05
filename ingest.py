"""
ingest.py — Load PDFs and build FAISS vector stores

Documents:
    doc/BDM (July 2023).pdf        — ODOT Bridge Design Manual (2020 Ed., July 2023)
    doc/AASHTO LRFD 9th Edition 3.pdf  — AASHTO LRFD Bridge Design Specifications, 9th Ed.

BDM format note:
    The BDM uses a two-column layout.
    - Left column  : binding ODOT structural design specifications (imperative mood)
    - Right column : commentary only, prefixed "C" (e.g., C101.1) — NOT binding
    BDM Section 1000 is the "ODOT Supplement to the LRFD Bridge Design Specifications"
    and is where Ohio modifies or supersedes AASHTO LRFD requirements.

Output:
    vector_store/bdm/      — BDM vector store
    vector_store/aashto/   — AASHTO LRFD vector store

Usage:
    python ingest.py
"""

import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Path configuration ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DOC_DIR = ROOT / "doc"
STORE_DIR = ROOT / "vector_store"

BDM_PDF    = DOC_DIR / "BDM (July 2023).pdf"
AASHTO_PDF = DOC_DIR / "AASHTO LRFD 9th Edition 3.pdf"

BDM_STORE    = STORE_DIR / "bdm"
AASHTO_STORE = STORE_DIR / "aashto"

# ── Chunking parameters ───────────────────────────────────────────────────────
# Larger chunks preserve section context (section headers + full requirement text)
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 250

# ── BDM Section 1000 detection ────────────────────────────────────────────────
# Chunks from BDM Section 1000 ("ODOT Supplement to LRFD") are tagged specially
# so the retriever can flag them as Ohio-specific overrides of AASHTO.
_SECTION_1000_RE = re.compile(r'\b1[0-9]{3}\b')   # matches 1000–1999


def _classify_bdm_chunk(text: str) -> dict:
    """
    Return extra metadata for a BDM chunk:
      - is_supplement : True if the chunk appears to be from BDM Section 1000
                        (ODOT Supplement to LRFD), meaning it directly modifies
                        or overrides an AASHTO article.
      - is_commentary : True if the chunk appears to be from a C-numbered section
                        (right-column commentary), which is informational only.
    """
    has_supplement = bool(_SECTION_1000_RE.search(text)) or any(
        kw in text for kw in ["ODOT Supplement", "in lieu of", "modify", "supersede",
                               "Section 1000", "1000."]
    )
    # Commentary sections start with "C" followed by a section number, e.g. "C101.1"
    is_commentary = bool(re.search(r'\bC\d{3}', text))
    return {"is_supplement": has_supplement, "is_commentary": is_commentary}


def load_and_split(pdf_path: Path, source_tag: str) -> list:
    """Load a PDF, split into chunks, and inject source metadata."""
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

    # Enrich BDM chunks with structural metadata
    if source_tag == "BDM":
        for chunk in chunks:
            chunk.metadata.update(_classify_bdm_chunk(chunk.page_content))

    print(f"    → {len(pages)} pages → {len(chunks)} chunks")
    return chunks


def build_store(chunks: list, store_path: Path, embeddings) -> None:
    """Build a FAISS vector store from chunks and persist to disk."""
    print(f"[+] Building vector store → {store_path} ...")
    store_path.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(store_path))
    print(f"    → Done. {db.index.ntotal} vectors stored.")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable first.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    bdm_chunks    = load_and_split(BDM_PDF,    "BDM")
    build_store(bdm_chunks, BDM_STORE, embeddings)

    aashto_chunks = load_and_split(AASHTO_PDF, "AASHTO")
    build_store(aashto_chunks, AASHTO_STORE, embeddings)

    print("\n[✓] All vector stores built successfully.")


if __name__ == "__main__":
    main()
