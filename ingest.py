"""
ingest.py — Load PDFs and build FAISS vector stores

Article-boundary chunking: splits at article headings (e.g. "303.1.1 VEHICULAR LIVE LOAD")
so each article is its own chunk with no cross-article contamination.

Rate limiting: OpenAI free/tier-1 accounts have 40,000 TPM limit.
Already-built stores are skipped.
"""

import bisect
import os
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDoc
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
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 300
MIN_CONTENT   = 30   # skip articles with fewer than this many body chars

# Article heading patterns
_BDM_ART_HDR_RE = re.compile(
    r'^(\d{3,4}\.\d+(?:\.\d+)*)\s{1,4}([A-Z][A-Z\s\-/]{2,60}?)$',
    re.MULTILINE,
)
_AASHTO_ART_HDR_RE = re.compile(
    r'^(\d\.\d+(?:\.\d+)+)\s{1,4}([A-Z][A-Za-z\s\-/:,\(\)]{2,60}?)$',
    re.MULTILINE,
)

# TOC sentinel: heading lines in the table of contents have long dot sequences
_TOC_DOTS_RE = re.compile(r'\.{8,}')

# ── Rate-limit parameters ─────────────────────────────────────────────────────
EMBED_BATCH = 50   # chunks per embedding API call
BATCH_SLEEP = 22   # seconds between batches

_SECTION_1000_RE = re.compile(r'\b1[0-9]{3}\b')


def _classify_bdm_chunk(text: str) -> dict:
    has_supplement = bool(_SECTION_1000_RE.search(text)) or any(
        kw in text for kw in ["ODOT Supplement", "in lieu of", "modify",
                               "supersede", "Section 1000", "1000."]
    )
    is_commentary = bool(re.search(r'\bC\d{3}', text))
    return {"is_supplement": has_supplement, "is_commentary": is_commentary}


def _find_article_boundaries(combined: str, pattern) -> list:
    """
    Return list of (char_offset, art_num, art_title) for each article heading,
    excluding TOC entries (detected by long dot sequences following the heading).
    """
    boundaries = []
    for m in pattern.finditer(combined):
        # Check the 120 chars after the match for TOC dot leaders
        following = combined[m.end(): m.end() + 120]
        if _TOC_DOTS_RE.search(following):
            continue   # skip TOC entry
        boundaries.append((m.start(), m.group(1), m.group(2).strip()))
    return boundaries


def load_and_split(pdf_path: Path, source_tag: str) -> list:
    """
    Load a PDF and split into one chunk per article heading.

    Strategy:
    1. Load all pages and concatenate into one combined string.
    2. Build a char-offset → page-number lookup (via bisect).
    3. Find all article heading positions (excluding TOC entries).
    4. For each article, extract text from its heading to the next heading.
    5. If the article text is short enough, store as one chunk.
       If it is long, use RecursiveCharacterTextSplitter into sub-chunks,
       each prefixed with the same article context header.
    """
    print(f"[+] Loading {pdf_path.name} ...")
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()

    # Build combined text and char-offset → page-number mapping
    page_offsets = []
    page_numbers = []
    parts        = []
    pos          = 0
    for doc in pages:
        pg = doc.metadata.get("page", 0) + 1   # 1-indexed
        page_offsets.append(pos)
        page_numbers.append(pg)
        parts.append(doc.page_content)
        pos += len(doc.page_content) + 1        # +1 for the joining "\n"
    combined = "\n".join(parts)

    def get_page(offset: int) -> int:
        i = bisect.bisect_right(page_offsets, offset) - 1
        return page_numbers[max(0, i)]

    pattern    = _BDM_ART_HDR_RE if source_tag == "BDM" else _AASHTO_ART_HDR_RE
    boundaries = _find_article_boundaries(combined, pattern)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for idx, (start, art_num, art_title) in enumerate(boundaries):
        end      = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(combined)
        art_text = combined[start:end].strip()
        page     = get_page(start)

        # Skip articles whose body is too short (e.g. pure section-title lines)
        first_nl = art_text.find("\n")
        body     = art_text[first_nl + 1:].strip() if first_nl >= 0 else ""
        if len(body) < MIN_CONTENT:
            continue

        ctx_hdr = f"[{source_tag} Article {art_num} — {art_title[:50]} | p.{page}]"

        if len(art_text) <= CHUNK_SIZE:
            final = ctx_hdr + "\n" + art_text
            meta  = {"source_tag": source_tag, "page": page}
            if source_tag == "BDM":
                meta.update(_classify_bdm_chunk(final))
            chunks.append(LCDoc(page_content=final, metadata=meta))
        else:
            for sub in splitter.split_text(art_text):
                final = ctx_hdr + "\n" + sub
                meta  = {"source_tag": source_tag, "page": page}
                if source_tag == "BDM":
                    meta.update(_classify_bdm_chunk(final))
                chunks.append(LCDoc(page_content=final, metadata=meta))

    print(f"    -> {len(pages)} pages, {len(boundaries)} article headings -> {len(chunks)} chunks")
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
    total     = len(chunks)
    n_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH
    est_min   = round(n_batches * BATCH_SLEEP / 60)
    print(f"[+] Building {store_path.name}: {total} chunks "
          f"in {n_batches} batches (~{est_min} min) ...")

    db = None
    for i in range(0, total, EMBED_BATCH):
        batch     = chunks[i: i + EMBED_BATCH]
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
