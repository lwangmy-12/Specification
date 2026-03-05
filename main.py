"""
main.py — FastAPI backend for the ODOT BDM + AASHTO LRFD Specification Assistant

Endpoints:
    GET  /                  → Frontend HTML
    GET  /pdf/bdm           → Serve BDM PDF (supports #page=N in browser)
    GET  /pdf/aashto        → Serve AASHTO LRFD PDF (supports #page=N)
    POST /api/query         → Dual-store RAG with multi-turn history
    GET  /api/health        → Engine + store status
    POST /api/ingest        → Trigger background PDF ingestion
"""

import re
import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import DualStoreRAG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT         = Path(__file__).parent
STATIC_DIR   = ROOT / "static"
DOC_DIR      = ROOT / "doc"
BDM_STORE    = ROOT / "vector_store" / "bdm"
AASHTO_STORE = ROOT / "vector_store" / "aashto"
BDM_PDF      = DOC_DIR / "BDM (July 2023).pdf"
AASHTO_PDF   = DOC_DIR / "AASHTO LRFD 9th Edition 3.pdf"

app = FastAPI(
    title="ODOT BDM + AASHTO LRFD Specification Assistant",
    version="1.0.0",
)

_rag: DualStoreRAG | None = None
_ingesting: bool = False


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global _rag
    if BDM_STORE.exists() and AASHTO_STORE.exists():
        log.info("Loading vector stores ...")
        _rag = DualStoreRAG(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        log.info("RAG engine ready.")
    else:
        log.warning("Vector stores not found. POST /api/ingest to build them.")


# ── Models ────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str     # "user" | "assistant"
    content: str  # plain text (for user), or raw LLM answer (for assistant)


class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


class Citation(BaseModel):
    source:  str   # "BDM" | "AASHTO"
    page:    int   # 1-indexed PDF page number
    excerpt: str   # ~180 char preview
    binding: bool  # False if commentary (C-section)


class QueryResponse(BaseModel):
    has_override:     bool
    reference:        str
    summary:          str
    guidance:         str
    raw:              str
    standalone_q:     str
    bdm_citations:    list[Citation]
    aashto_citations: list[Citation]


class HealthResponse(BaseModel):
    status:       str
    engine_ready: bool
    bdm_store:    bool
    aashto_store: bool
    ingesting:    bool


# ── Response section parser ───────────────────────────────────────────────────

_SECTION_PATTERNS = [
    ("reference", r"\[Article/Section Reference\](.*?)(?=\[Specification Summary\]|\Z)"),
    ("summary",   r"\[Specification Summary\](.*?)(?=\[Engineering Practice Guidance\]|\Z)"),
    ("guidance",  r"\[Engineering Practice Guidance\](.*?)(?=\Z)"),
]

def _parse_sections(text: str) -> dict:
    sections = {}
    for key, pattern in _SECTION_PATTERNS:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sections[key] = m.group(1).strip() if m else ""
    return sections


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/pdf/bdm", include_in_schema=False)
async def serve_bdm_pdf():
    """Serve the BDM PDF. Browser PDF viewer supports #page=N URL fragment."""
    if not BDM_PDF.exists():
        raise HTTPException(status_code=404, detail="BDM PDF not found in doc/ folder.")
    return FileResponse(
        str(BDM_PDF),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=\"BDM-July2023.pdf\""},
    )


@app.get("/pdf/aashto", include_in_schema=False)
async def serve_aashto_pdf():
    """Serve the AASHTO LRFD PDF. Browser PDF viewer supports #page=N URL fragment."""
    if not AASHTO_PDF.exists():
        raise HTTPException(status_code=404, detail="AASHTO PDF not found in doc/ folder.")
    return FileResponse(
        str(AASHTO_PDF),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=\"AASHTO-LRFD-9th.pdf\""},
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        engine_ready=_rag is not None,
        bdm_store=BDM_STORE.exists(),
        aashto_store=AASHTO_STORE.exists(),
        ingesting=_ingesting,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not ready. POST /api/ingest to build vector stores.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    log.info(f"Query: {req.question[:120]}  [history turns: {len(req.history)}]")

    history = [{"role": m.role, "content": m.content} for m in req.history]
    result  = _rag.ask(req.question, history if history else None)

    sections = _parse_sections(result["answer"])
    has_override = result["has_override"] or "ODOT OVERRIDE" in result["answer"]

    return QueryResponse(
        has_override     = has_override,
        reference        = sections.get("reference", ""),
        summary          = sections.get("summary", ""),
        guidance         = sections.get("guidance", ""),
        raw              = result["answer"],
        standalone_q     = result.get("standalone_q", req.question),
        bdm_citations    = [Citation(**c) for c in result.get("bdm_citations",    [])],
        aashto_citations = [Citation(**c) for c in result.get("aashto_citations", [])],
    )


@app.post("/api/ingest", status_code=202)
async def ingest(background_tasks: BackgroundTasks):
    global _ingesting
    if _ingesting:
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    def _run_ingest():
        global _rag, _ingesting
        _ingesting = True
        try:
            log.info("Starting PDF ingestion ...")
            import ingest as ingest_module
            ingest_module.main()
            log.info("Ingestion complete. Reloading RAG engine ...")
            _rag = DualStoreRAG(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
            log.info("RAG engine ready.")
        except Exception as e:
            log.error(f"Ingestion failed: {e}")
        finally:
            _ingesting = False

    background_tasks.add_task(_run_ingest)
    return {"message": "Ingestion started. Poll GET /api/health for status."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
