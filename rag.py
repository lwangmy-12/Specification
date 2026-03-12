"""
rag.py — Dual vector store RAG with HyDE + multi-query retrieval

Retrieval flow:
    1. If follow-up, reformulate into standalone question.
    2. LLM generates: HyDE excerpt + 3 search queries + design context.
       HyDE (Hypothetical Document Embedding): generate a fake spec excerpt
       that *would* answer the question, embed it, search FAISS with that
       embedding — the hypothetical text shares vocabulary with real spec text
       and retrieves far better than the raw question embedding alone.
    3. Run all queries + HyDE against FAISS with NO prefix filter (k=30 each,
       union + deduplicate).  FAISS uses its full semantic strength.
    4. Apply design-context hard exclusions (rehab/rating chapters for new design).
    5. LLM re-rank the pool (score 1-10, keep >=6).  This is the ONLY relevance
       gate — no article-number prefix filter silently drops correct content.
    6. Augment AASHTO with articles explicitly cross-referenced in BDM.
    7. Build verbatim-quote prompt, call LLM, parse structured response.

Page-number accuracy:
    _extract_article() only searches the context header (first line), never the
    body text.  This prevents equation numbers like "Eq. 6.7.4.2-1" in a body
    from being mistaken for an article number and causing wrong PDF page jumps.
    _citations_from_answer() is augmented with the full TOC dicts so that any
    article cited in the answer (even if not retrieved directly) gets the
    correct page number.

BDM document authority (per BDM 101.4):
    BDM > AASHTO LRFD for Ohio DOT projects.
    BDM Section 1000 = ODOT Supplement directly modifies/supersedes AASHTO.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

ROOT         = Path(__file__).parent
BDM_STORE    = ROOT / "vector_store" / "bdm"
AASHTO_STORE = ROOT / "vector_store" / "aashto"

BDM_TOP_K     = 8    # max chunks kept after reranking
AASHTO_TOP_K  = 8
RERANK_POOL_K = 30   # candidates fetched per query before reranking
RERANK_MIN    = 6    # min reranker score (1-10) to keep a chunk

_OVERRIDE_PHRASES = [
    "in lieu of", "modify", "supersede", "supplement", "exception",
    "shall not", "not applicable", "ODOT Supplement", "Section 1000",
    "1000.", "replace", "instead of",
]
_SECTION_1000_RE = re.compile(r'\b1[0-9]{3}\.[0-9]')
_BDM_ART_RE      = re.compile(r'([1-9]\d{2,3}\.\d+(?:\.\d+)*)')
_AASHTO_ART_RE   = re.compile(r'([1-9]\.\d+(?:\.\d+)+)')
_LRFD_XREF_RE    = re.compile(r'LRFD\s+(?:Article\s+)?(\d+\.\d+(?:\.\d+)*)', re.I)
_ANS_BDM_RE      = re.compile(r'BDM\s+(\d{3,4}\.\d+(?:\.\d+)*)', re.I)
_ANS_AASHTO_RE   = re.compile(r'AASHTO\s+(?:LRFD\s+)?(?:Article\s+)?(\d+\.\d+(?:\.\d+)*)', re.I)
# Matches context header written by ingest.py: "[BDM Article 303.1 — TITLE | p.45]"
_TOC_HEADER_RE   = re.compile(
    r'^\[(BDM|AASHTO) Article ([\d.]+) [\u2014\-] (.+?) \| p\.(\d+)\]'
)
_AASHTO_TITLE_RE = re.compile(
    r'^(\d\.\d+(?:\.\d+)+)[—–\-]\s*([A-Z][A-Za-z\-\/][A-Za-z\s:,\(\)\.\/\-]{2,70}?)$',
    re.MULTILINE,
)
_BDM_TITLE_RE = re.compile(
    r'^(\d{3,4}\.\d+(?:\.\d+)*)\s{1,6}([A-Z][A-Z\s\-/]{2,60}?)$',
    re.MULTILINE,
)
_TOC_DOTS_RE = re.compile(r'\.{5,}')   # detects TOC dot leaders in AASHTO


# ── Basic helpers ──────────────────────────────────────────────────────────────

def _has_odot_override(bdm_docs):
    for doc in bdm_docs:
        text = doc.page_content
        if doc.metadata.get("is_supplement"):
            return True
        if _SECTION_1000_RE.search(text):
            return True
        if any(p.lower() in text.lower() for p in _OVERRIDE_PHRASES):
            return True
    return False


def _is_commentary(doc):
    if doc.metadata.get("is_commentary"):
        return True
    return bool(re.search(r'\bC\d{3}', doc.page_content[:200]))


def _extract_article(text, source_tag):
    """
    Extract article number from the context header (first line) ONLY.

    Never search the body text: equation references like "Eq. 6.7.4.2-1"
    would match the AASHTO regex and produce a wrong article-to-page mapping,
    causing PDF page links to jump to the wrong section.
    """
    first_line = text[:text.index("\n")] if "\n" in text else text[:200]
    pat = _BDM_ART_RE if source_tag == "BDM" else _AASHTO_ART_RE
    m = pat.search(first_line)
    return m.group(1) if m else ""


def _is_notes_query(question):
    keywords = [
        "note", "plan note", "drawing note", "contract drawing", "plan sheet",
        "what to write", "what should be shown", "what goes on the plans",
        "general notes", "bridge notes", "plan requirement",
    ]
    return any(k in question.lower() for k in keywords)


# ── TOC extraction ────────────────────────────────────────────────────────────

def _extract_toc(db, source_tag):
    """
    Build article_number -> {title, page} from all docs in a FAISS store.

    For BDM (article-boundary ingest): reads the context header prepended by
    ingest.py — each chunk has exactly one article heading.
    For AASHTO (old page-joining ingest): also text-mines mixed-case headings
    like "6.7.4.2—Cross-Frames and Diaphragms" from chunk bodies.

    This TOC is used for accurate PDF page-number lookups in citations.
    """
    articles = {}
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        if doc is None:
            continue
        text = doc.page_content
        page = int(doc.metadata.get('page', 0)) + 1

        # Primary: context header set by ingest.py (most accurate page number)
        m = _TOC_HEADER_RE.match(text)
        if m and m.group(1) == source_tag:
            art   = m.group(2)
            title = m.group(3).strip()
            pg    = int(m.group(4))
            if art not in articles:
                articles[art] = {"title": title, "page": pg}
            if source_tag == "AASHTO":
                continue   # context header is sufficient for AASHTO

        # Fallback: text-mine article headings from chunk body
        if source_tag == "AASHTO":
            # AASHTO uses mixed-case: "6.7.4.2—Cross-Frames and Diaphragms"
            for m2 in _AASHTO_TITLE_RE.finditer(text[:800]):
                # Skip TOC entries (dot leaders follow the title)
                line_end = text.find(chr(10), m2.end())
                if line_end < 0:
                    line_end = len(text)
                if _TOC_DOTS_RE.search(text[m2.start():line_end]):
                    continue
                art   = m2.group(1)
                title = m2.group(2).strip()
                if art not in articles:
                    articles[art] = {"title": title, "page": page}
        elif source_tag == "BDM":
            # BDM uses ALL-CAPS: "303.1.1  VEHICULAR LIVE LOAD"
            for m2 in _BDM_TITLE_RE.finditer(text):
                art   = m2.group(1)
                title = m2.group(2).strip()
                if art not in articles:
                    articles[art] = {"title": title, "page": page}

    return articles


# ── Query generation with HyDE ────────────────────────────────────────────────

def _generate_queries(question: str, llm) -> dict:
    """
    Generate HyDE text, multiple search queries, and design context.

    HyDE (Hypothetical Document Embedding): the LLM writes a 2-3 sentence
    excerpt as if quoting from the actual specification.  Embedding this
    hypothetical text retrieves far better matches than embedding the raw
    question, because the hypothetical text shares vocabulary and phrasing
    with real specification chunks.

    Returns: {hyde, queries, design_context}
    """
    nl = chr(10)
    prompt = (
        f"You are a bridge engineering expert. A designer asks:{nl}"
        f"Question: {question}{nl}{nl}"
        f"Task 1 — HyDE (Hypothetical Document Embedding):{nl}"
        f"Write 2-3 sentences as if quoting VERBATIM from either the ODOT BDM or{nl}"
        f"AASHTO LRFD 9th Edition that would DIRECTLY answer this question.{nl}"
        f"Use precise technical language, specific numeric values if known, and{nl}"
        f"article numbers if you know them (e.g. 'BDM 303.1.1', 'LRFD 3.6.1.2')."
        f" This will be embedded and used to search a vector database.{nl}{nl}"
        f"Task 2 — Search queries:{nl}"
        f"Write exactly 3 diverse search queries (8-15 words each):{nl}"
        f"  [0] Most precise technical terminology for this topic{nl}"
        f"  [1] Include article/section number anchors if known (BDM/AASHTO numbers){nl}"
        f"  [2] Alternative phrasing, synonyms, or related design concepts{nl}{nl}"
        f"Task 3 — Design context:{nl}"
        f"Return 'new_design', 'rehab', or 'rating' based on the question intent.{nl}{nl}"
        f"Return ONLY valid JSON (no markdown fences):{nl}"
        f'{{"hyde": "...", "queries": ["...", "...", "..."], '
        f'"design_context": "new_design"}}'
    )
    defaults = {
        "hyde":           "",
        "queries":        [question],
        "design_context": "new_design",
    }
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        data = json.loads(text)
        if data.get("hyde"):
            defaults["hyde"] = str(data["hyde"])
        if isinstance(data.get("queries"), list) and data["queries"]:
            defaults["queries"] = [str(q) for q in data["queries"][:4]]
        if data.get("design_context") in ("new_design", "rehab", "rating"):
            defaults["design_context"] = data["design_context"]
    except Exception as e:
        print(f"     Query generation error: {e} — using fallback")
    return defaults


# ── Broad semantic search ─────────────────────────────────────────────────────

def _broad_search(db, queries: list, hyde_text: str,
                  source_tag: str, k: int = RERANK_POOL_K) -> list:
    """
    Search FAISS with multiple queries + HyDE. Return deduplicated union.

    No prefix filtering: FAISS uses full semantic strength. The LLM reranker
    (called next) is the sole relevance gate.
    """
    all_queries = list(queries)
    if hyde_text:
        all_queries.append(hyde_text)

    k_per = max(12, k // len(all_queries) + 4)

    seen    = set()
    results = []
    for q in all_queries:
        raw = db.similarity_search(q, k=k_per)
        for doc in raw:
            page = doc.metadata.get("page", -1)
            key  = (page, doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                results.append(doc)

    print(f"     [{source_tag}] {len(all_queries)} queries → {len(results)} unique candidates")
    return results


# ── Design-context hard filter ────────────────────────────────────────────────

_BDM_NEW_DESIGN_EXCLUDE  = set(range(400, 500)) | set(range(900, 1000))
_BDM_NOTES_ONLY_CHAPTERS = set(range(600, 800))
_BDM_NOTES_HEADER_RE     = re.compile(
    r'(SECTION\s+[67]\d{2}|TYPICAL\s+GENERAL\s+NOTES|TYPICAL\s+SPECIAL\s+NOTES'
    r'|6\d{2}\.\d|7\d{2}\.\d|\[6\d{2}\.\d|\[7\d{2}\.\d)',
    re.I
)
_RATING_CONTENT_RE = re.compile(
    r'(load rating|rating factor|inventory rating|operating rating|load posting'
    r'|rating vehicle|legal load|permit load|MBE|Manual for Bridge Evaluation)',
    re.I
)


def _context_filter(docs, source_tag, design_context, notes_query=False):
    if source_tag != "BDM":
        return docs

    # Stage 1: hard exclusion of plan-notes chapters 600-799
    if not notes_query:
        hard_keep = []
        for doc in docs:
            art = _extract_article(doc.page_content, "BDM")
            if art:
                try:
                    if int(art.split(".")[0]) in _BDM_NOTES_ONLY_CHAPTERS:
                        continue
                except ValueError:
                    pass
            if _BDM_NOTES_HEADER_RE.search(doc.page_content[:400]):
                continue
            hard_keep.append(doc)
        docs = hard_keep

    # Stage 2: design-context exclusion
    if design_context != "new_design":
        return docs

    filtered = []
    for doc in docs:
        art  = _extract_article(doc.page_content, "BDM")
        skip = False
        if art:
            try:
                if int(art.split(".")[0]) in _BDM_NEW_DESIGN_EXCLUDE:
                    skip = True
            except ValueError:
                pass
        if not skip and not art:
            if _RATING_CONTENT_RE.search(doc.page_content[:500]):
                skip = True
        if not skip:
            filtered.append(doc)

    return filtered if filtered else []


# ── LLM re-ranking ────────────────────────────────────────────────────────────

def _rerank_docs(docs, query, llm, source_tag, design_context="new_design",
                 top_k=8):
    """
    Score each doc 1-10; keep those scoring >= RERANK_MIN, up to top_k.
    This is the primary relevance gate — no prefix filter precedes it.
    """
    if not docs:
        return docs

    nl = chr(10)
    passages = nl.join(
        f"[{i+1}] {doc.page_content[:300].strip().replace(nl, ' ')}"
        for i, doc in enumerate(docs)
    )

    if design_context == "new_design":
        ctx_rule = (
            f"- CRITICAL: NEW BRIDGE DESIGN question. Rehab or load-rating passages -> <=2.{nl}"
            f"- BDM 401.3 is rehab only -> <=2 for new design.{nl}"
        )
    elif design_context == "rating":
        ctx_rule = f"- CRITICAL: LOAD RATING question. New-construction-only passages -> <=3.{nl}"
    else:
        ctx_rule = f"- CRITICAL: REHABILITATION question. New-construction / rating only -> <=3.{nl}"

    domain_rules = (
        f"Bridge engineering disambiguation rules:{nl}"
        f"{ctx_rule}"
        f"- BDM Chapters 600-799: PLAN NOTES ONLY -> <=2 unless question is about plan notes.{nl}"
        f"- TOC/index pages or chapter headings with no spec text -> <=3.{nl}"
        f"- Commentary (C-prefix articles): 2 pts lower than binding spec.{nl}"
        f"- Passage is about a completely different topic from the question -> 1.{nl}"
    )
    prompt = (
        f"You are a bridge engineering specification relevance filter.{nl}"
        f"Question: {query}{nl}{nl}"
        f"{domain_rules}{nl}"
        f"Rate each passage 1-10 (1=off-topic, 10=directly answers the question).{nl}"
        f"Passages:{nl}{passages}{nl}{nl}"
        f"Reply ONLY with comma-separated integers, one per passage (e.g. 8,2,7,5):"
    )
    try:
        resp   = llm.invoke(prompt)
        raw    = resp.content.strip().split(nl)[0]
        scores = [int(s.strip()) for s in raw.split(",")]
        if len(scores) != len(docs):
            return docs[:top_k]
        ranked = sorted(
            [(doc, score) for doc, score in zip(docs, scores)],
            key=lambda x: -x[1]
        )
        kept   = [doc for doc, score in ranked if score >= RERANK_MIN]
        result = kept[:top_k] if kept else [doc for doc, _ in ranked[:3]]
        print(f"     [{source_tag}] rerank scores: {scores} -> kept {len(result)}")
        return result
    except Exception:
        return docs[:top_k]


# ── AASHTO cross-reference augmentation ───────────────────────────────────────

def _extract_lrfd_refs(bdm_docs):
    seen, refs = set(), []
    for doc in bdm_docs:
        for m in _LRFD_XREF_RE.finditer(doc.page_content):
            art = m.group(1)
            if art not in seen:
                seen.add(art)
                refs.append(art)
    return refs


def _fetch_aashto_by_xrefs(refs, aashto_db, k_per_ref=2):
    fetched, seen_pages = [], set()
    for ref in refs[:8]:
        results = aashto_db.similarity_search(f"AASHTO LRFD Article {ref}", k=k_per_ref)
        for doc in results:
            page = doc.metadata.get("page", -1)
            if page not in seen_pages:
                seen_pages.add(page)
                fetched.append(doc)
    return fetched


# ── Context & citation helpers ────────────────────────────────────────────────

def _format_docs(docs, source):
    if not docs:
        return "(No relevant content retrieved)"
    parts = []
    for i, doc in enumerate(docs, 1):
        page  = doc.metadata.get("page", "?")
        label = ""
        if source == "BDM":
            label = " | COMMENTARY — not binding" if _is_commentary(doc) else " | BINDING SPEC"
        parts.append(f"[Excerpt {i} | Page {page}{label}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _build_page_map(docs, source_tag):
    """
    Build article -> page mapping from retrieved chunks.

    Uses the context header page number (from ingest.py) as the primary source.
    Only falls back to metadata page if no context header is present.
    Never text-mines the body for article numbers (avoids equation-reference
    false positives like 'Eq. 6.7.4.2-1' -> article 6.7.4.2 -> wrong page).
    """
    page_map = {}
    for doc in docs:
        text = doc.page_content

        # Primary: context header "[BDM Article 303.1.1 — TITLE | p.62]"
        m = _TOC_HEADER_RE.match(text)
        if m and m.group(1) == source_tag:
            art = m.group(2)
            pg  = int(m.group(4))
        else:
            # Fallback header "[BDM | p.62]" — use metadata page
            art = _extract_article(text, source_tag)  # only looks at first line
            pg  = int(doc.metadata.get("page", 0)) + 1

        if not art:
            continue
        if art not in page_map:
            page_map[art] = pg
        # Also map parent prefixes for partial-match lookups
        parts = art.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:depth])
            if parent not in page_map:
                page_map[parent] = pg

    return page_map


def _citations_from_answer(answer, bdm_docs, aashto_docs,
                            bdm_toc=None, aashto_toc=None):
    """
    Extract article citations from the LLM answer and resolve PDF page numbers.

    Page lookup priority:
      1. page_map built from the actual retrieved chunks (most specific)
      2. Full TOC dicts passed from DualStoreRAG (covers articles cited in the
         answer but not directly retrieved — e.g. cross-references)
      3. Returns page=0 if not found in either source
    """
    bdm_page_map    = _build_page_map(bdm_docs,    "BDM")
    aashto_page_map = _build_page_map(aashto_docs, "AASHTO")

    # Augment with full TOC (handles articles cited but not retrieved)
    if bdm_toc:
        for art, info in bdm_toc.items():
            if art not in bdm_page_map:
                bdm_page_map[art] = info["page"]
    if aashto_toc:
        for art, info in aashto_toc.items():
            if art not in aashto_page_map:
                aashto_page_map[art] = info["page"]

    ref_section = re.search(
        r'\[Article/Section Reference\](.*?)(?=\[Engineering Practice Guidance\]|\Z)',
        answer, re.DOTALL | re.I,
    )
    search_text = ref_section.group(1) if ref_section else answer

    def lookup_page(article, page_map):
        if article in page_map:
            return page_map[article]
        parts = article.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:depth])
            if prefix in page_map:
                return page_map[prefix]
        return 0

    bdm_citations, seen_bdm = [], set()
    for m in _ANS_BDM_RE.finditer(search_text):
        art = m.group(1)
        if art in seen_bdm:
            continue
        seen_bdm.add(art)
        page    = lookup_page(art, bdm_page_map)
        excerpt = ""
        for doc in bdm_docs:
            if art in doc.page_content[:500]:
                excerpt = doc.page_content[:180].strip().replace(chr(10), " ")
                break
        bdm_citations.append({
            "source":  "BDM",
            "page":    page,
            "article": art,
            "excerpt": excerpt,
            "binding": not bool(re.match(r'C\d', art)),
        })

    aashto_citations, seen_aashto = [], set()
    for m in _ANS_AASHTO_RE.finditer(search_text):
        art = m.group(1)
        if "." not in art or art in seen_aashto:
            continue
        seen_aashto.add(art)
        page    = lookup_page(art, aashto_page_map)
        excerpt = ""
        for doc in aashto_docs:
            if art in doc.page_content[:500]:
                excerpt = doc.page_content[:180].strip().replace(chr(10), " ")
                break
        aashto_citations.append({
            "source":  "AASHTO",
            "page":    page,
            "article": art,
            "excerpt": excerpt,
            "binding": True,
        })

    return bdm_citations, aashto_citations


def _docs_to_citations(docs, source_tag):
    seen, citations = set(), []
    for doc in docs:
        article = _extract_article(doc.page_content, source_tag)
        if not article:
            continue
        key = (source_tag, article)
        if key in seen:
            continue
        seen.add(key)
        # Use context header page number when available
        m = _TOC_HEADER_RE.match(doc.page_content)
        if m and m.group(1) == source_tag:
            page_1indexed = int(m.group(4))
        else:
            page_1indexed = int(doc.metadata.get("page", 0)) + 1
        is_comment = _is_commentary(doc)
        excerpt    = doc.page_content[:180].strip().replace(chr(10), ' ')
        citations.append({
            "source":  source_tag,
            "page":    page_1indexed,
            "article": article,
            "excerpt": excerpt,
            "binding": not is_comment,
        })
    return citations


# ── Prompt builder ────────────────────────────────────────────────────────────

# Brief spec structure primer injected into every prompt so the LLM knows the
# document conventions without needing to infer them from retrieved chunks.
_SPEC_STRUCTURE_PRIMER = """\
SPECIFICATION STRUCTURE (read before answering):
BDM (Ohio Bridge Design Manual):
  - Two-column layout: LEFT = binding imperative spec (shall/must);
    RIGHT = Commentary (C-prefix articles, e.g. C303.1) — informational only.
  - Numbering: 3-4 digit chapter (e.g. 303.1.1 = Chapter 303, Article 1.1).
  - BDM Section 1000 = ODOT Supplement that directly modifies AASHTO LRFD.
  - BDM 101.4: BDM takes precedence over AASHTO LRFD for Ohio DOT projects.
AASHTO LRFD 9th Edition:
  - Single-digit section (e.g. 3.6.1.2.1 = Section 3, Article 6.1.2.1).
  - BDM cross-references written as "LRFD X.X.X" (italic in source).
"""


def _build_prompt(bdm_context, aashto_context, has_override, question,
                  design_context="new_design"):
    note = (
        " Note: BDM Section 1000 (ODOT Supplement) material retrieved."
        " Present it in [BDM Specification Text]."
        if has_override else ""
    )
    safe_bdm    = bdm_context.replace("{", "{{").replace("}", "}}")
    safe_aashto = aashto_context.replace("{", "{{").replace("}", "}}")
    safe_q      = question.replace("{", "{{").replace("}", "}}")
    sep = "=" * 67

    if design_context == "rating":
        ctx_note = "Design context: LOAD RATING of existing bridges (BDM Chapter 9, AASHTO MBE)."
        ctx_rule = "RULE 4: Load rating question. Exclude new-construction-only content."
    elif design_context == "rehab":
        ctx_note = "Design context: BRIDGE REHABILITATION / REPAIR."
        ctx_rule = "RULE 4: Rehabilitation question. Exclude new-construction / rating content."
    else:
        ctx_note = "Design context: NEW BRIDGE DESIGN AND CONSTRUCTION."
        ctx_rule = (
            "RULE 4 (overrides RULE 2): NEW BRIDGE DESIGN question. "
            "If any excerpt applies ONLY to rehabilitation or load rating, "
            "do NOT quote it. Write: [Context Note: BDM X.X.X applies to rehab/rating only.]"
        )

    lines = [
        "You are a professional bridge design specification consultant for Ohio DOT projects.",
        f"Audience: Designer of Record. {ctx_note}{note}",
        "",
        _SPEC_STRUCTURE_PRIMER,
        "RULE 1 — BOTH SPECS REQUIRED: BDM and AASHTO LRFD are independently required.",
        "Both must be satisfied. Do NOT state one supersedes the other.",
        "",
        "RULE 2 — VERBATIM: Quote WORD-FOR-WORD from the excerpts below.",
        "Do not paraphrase. Use exact article numbers from the source.",
        "",
        "RULE 3 — RELEVANCE: Only include text DIRECTLY relevant to the question.",
        "If an excerpt covers a different topic, exclude it and say so.",
        "",
        ctx_rule,
        "",
        sep,
        "OHIO BDM — Retrieved Excerpts (quote verbatim)",
        sep,
        safe_bdm,
        "",
        sep,
        "AASHTO LRFD 9th Ed. — Retrieved Excerpts (quote verbatim)",
        sep,
        safe_aashto,
        "",
        sep,
        f"Designer Question: {safe_q}",
        sep,
        "",
        "Respond ONLY in this exact format:",
        "",
        "[Article/Section Reference]",
        "List ALL articles from BOTH documents directly relevant to the question:",
        "  - BDM X.X.X - short description",
        "  - AASHTO LRFD Article X.X.X.X - short description",
        "",
        "[BDM Specification Text]",
        'Quote relevant BDM text VERBATIM from the excerpts.',
        '  Format: BDM X.X.X: "exact quoted text"',
        '  Commentary: BDM C302.3 (Commentary — informational only): "exact text"',
        "  If nothing relevant: The retrieved BDM excerpts do not contain directly relevant requirements.",
        "",
        "[AASHTO LRFD Specification Text]",
        "Quote relevant AASHTO text VERBATIM from the excerpts.",
        '  Format: AASHTO LRFD X.X.X.X: "exact quoted text"',
        "  If nothing relevant: The retrieved AASHTO excerpts do not contain directly relevant requirements.",
        "",
        "[Engineering Practice Guidance]",
        "Write as a practicing Ohio DOT bridge engineer advising the Designer of Record.",
        "Structure as:",
        "  1. DESIGN PROCEDURE: Step-by-step applying BOTH BDM and AASHTO requirements.",
        "  2. OHIO-SPECIFIC NOTES: BDM requirements that differ from AASHTO defaults.",
        "  3. CALCULATIONS / KEY VALUES: Parameters, equations, limit values.",
        "  4. COMMON DESIGN ERRORS: Pitfalls on Ohio DOT projects (supported by excerpts only).",
        "",
        "Rules:",
        "- Do not fabricate requirements not in the excerpts.",
        "- Do NOT state BDM overrides AASHTO; both are independently required.",
        "- If retrieved content is insufficient, state which BDM/AASHTO section to consult directly.",
    ]
    return chr(10).join(lines)


# ── DualStoreRAG ──────────────────────────────────────────────────────────────

class DualStoreRAG:
    """
    Dual vector store RAG with HyDE + multi-query retrieval.

    ask(question, history) -> {answer, has_override, bdm_citations,
                                aashto_citations, standalone_q}
    """

    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm        = ChatOpenAI(model=model, temperature=temperature)

        print("[+] Loading vector stores ...")
        for path, name in [(BDM_STORE, "BDM"), (AASHTO_STORE, "AASHTO LRFD")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} vector store not found: {path}\n"
                    "Run `python ingest.py` first."
                )
        self.bdm_db    = FAISS.load_local(str(BDM_STORE),    self.embeddings,
                                           allow_dangerous_deserialization=True)
        self.aashto_db = FAISS.load_local(str(AASHTO_STORE), self.embeddings,
                                           allow_dangerous_deserialization=True)

        print("[+] Extracting TOC from vector stores ...")
        self.bdm_toc    = _extract_toc(self.bdm_db,    "BDM")
        self.aashto_toc = _extract_toc(self.aashto_db, "AASHTO")
        print(f"    BDM TOC: {len(self.bdm_toc)} articles  |  "
              f"AASHTO TOC: {len(self.aashto_toc)} articles")

        print("[OK] RAG engine ready.\n")

    def _reformulate(self, question, history):
        recent       = history[-6:]
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        )
        prompt = (
            "Given this conversation history:\n"
            f"{history_text}\n\n"
            "Reformulate the following follow-up question as a concise, standalone "
            "engineering question that can be understood without conversation history. "
            "Preserve all technical terms and article numbers. "
            "If already standalone, return it unchanged.\n\n"
            f"Follow-up: {question}\n"
            "Standalone:"
        )
        response     = self.llm.invoke(prompt)
        reformulated = response.content.strip()
        return reformulated if reformulated else question

    def ask(self, question, history=None):
        """
        Run dual-store retrieval with HyDE + multi-query expansion.
        Returns: {answer, has_override, bdm_citations, aashto_citations, standalone_q}
        """
        # Step 0: reformulate follow-up questions
        if history:
            print("  -> Reformulating follow-up question ...")
            standalone_q = self._reformulate(question, history)
            if standalone_q != question:
                print(f"     Original    : {question}")
                print(f"     Reformulated: {standalone_q}")
        else:
            standalone_q = question

        # Step 1: Generate HyDE + multi-query + design context
        print("  -> [1] Generating HyDE + search queries ...")
        qgen       = _generate_queries(standalone_q, self.llm)
        design_ctx = qgen["design_context"]
        notes_q    = _is_notes_query(standalone_q)
        # Append original question to each generated query for robustness
        queries    = [f"{q} {standalone_q}" for q in qgen["queries"]]
        hyde       = qgen["hyde"]
        print(f"     Design context: {design_ctx}")
        if hyde:
            print(f"     HyDE excerpt  : {hyde[:100]}...")
        print(f"     Queries       : {qgen['queries']}")

        # Step 2: Broad FAISS search — no prefix filter, full semantic strength
        print("  -> [2] Broad semantic search (BDM + AASHTO in parallel) ...")

        def _fetch_bdm():
            docs = _broad_search(self.bdm_db, queries, hyde, "BDM", k=RERANK_POOL_K)
            docs = _context_filter(docs, "BDM", design_ctx, notes_query=notes_q)
            docs = _rerank_docs(docs, standalone_q, self.llm, "BDM", design_ctx,
                                top_k=BDM_TOP_K)
            return docs

        def _fetch_aashto():
            docs = _broad_search(self.aashto_db, queries, hyde, "AASHTO", k=RERANK_POOL_K)
            docs = _context_filter(docs, "AASHTO", design_ctx, notes_query=notes_q)
            docs = _rerank_docs(docs, standalone_q, self.llm, "AASHTO", design_ctx,
                                top_k=AASHTO_TOP_K)
            return docs

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bdm    = pool.submit(_fetch_bdm)
            fut_aashto = pool.submit(_fetch_aashto)
            bdm_docs    = fut_bdm.result()
            aashto_docs = fut_aashto.result()

        # Step 2b: Augment AASHTO with articles cross-referenced in BDM
        lrfd_refs = _extract_lrfd_refs(bdm_docs)
        if lrfd_refs:
            print(f"  -> BDM LRFD cross-references: {lrfd_refs}")
            xref_docs    = _fetch_aashto_by_xrefs(lrfd_refs, self.aashto_db)
            existing_pgs = {d.metadata.get("page") for d in aashto_docs}
            for doc in xref_docs:
                if doc.metadata.get("page") not in existing_pgs:
                    aashto_docs.insert(0, doc)
                    existing_pgs.add(doc.metadata.get("page"))

        # Step 3: Override detection
        has_override = _has_odot_override(bdm_docs)
        if has_override:
            print("  [!] BDM Section 1000 / ODOT supplement language detected.")

        print(f"     Final: {len(bdm_docs)} BDM chunks, {len(aashto_docs)} AASHTO chunks")

        # Step 4: Build prompt and call LLM
        bdm_context    = _format_docs(bdm_docs,    "BDM")
        aashto_context = _format_docs(aashto_docs, "AASHTO")
        prompt_text    = _build_prompt(bdm_context, aashto_context,
                                       has_override, standalone_q, design_ctx)

        print("  -> [3] Generating structured response ...")
        response = self.llm.invoke(prompt_text)

        # Step 5: Build citations — pass full TOC for accurate page numbers
        bdm_cits, aashto_cits = _citations_from_answer(
            response.content, bdm_docs, aashto_docs,
            bdm_toc=self.bdm_toc, aashto_toc=self.aashto_toc,
        )
        if not bdm_cits:
            bdm_cits    = _docs_to_citations(bdm_docs,    "BDM")
        if not aashto_cits:
            aashto_cits = _docs_to_citations(aashto_docs, "AASHTO")

        return {
            "answer":           response.content,
            "has_override":     has_override,
            "bdm_citations":    bdm_cits,
            "aashto_citations": aashto_cits,
            "standalone_q":     standalone_q,
        }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    rag     = DualStoreRAG(model="gpt-4o-mini")
    history = []

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result   = rag.ask(question)
        print(f"\n{'=' * 70}\n{result['answer']}\n{'=' * 70}")
        return

    print("=" * 70)
    print("  ODOT BDM + AASHTO LRFD Specification Assistant (multi-turn)")
    print("  Type 'quit' to exit, 'clear' to start a new conversation.")
    print("=" * 70)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "clear":
            history = []
            print("Conversation cleared.\n")
            continue

        result = rag.ask(question, history)
        answer = result["answer"]
        print(f"\n{'--' * 35}\n{answer}")
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
        if result["bdm_citations"]:
            print("\nBDM Sources:", ", ".join(
                f"{c['article']} p.{c['page']}" for c in result["bdm_citations"]
            ))
        if result["aashto_citations"]:
            print("AASHTO Sources:", ", ".join(
                f"{c['article']} p.{c['page']}" for c in result["aashto_citations"]
            ))
        print("--" * 35)


if __name__ == "__main__":
    main()
