"""
rag.py — Dual vector store RAG with multi-turn conversation support

Retrieval flow:
    1. If conversation history exists, reformulate the follow-up question
       into a standalone question (resolves pronouns / back-references).
    2. Query BDM vector store first  (Ohio state authority).
    3. Detect BDM Section 1000 / ODOT override language.
    4. Inject BDM context into a dynamic PromptTemplate.
    5. Run RetrievalQA against AASHTO vector store.
    6. Return structured answer + source citations (page numbers) for both stores.

BDM document authority (per BDM 101.4):
    BDM > AASHTO LRFD for Ohio DOT projects.
    BDM two-column layout: left = binding spec, right = commentary (C-prefix, not binding).
    BDM Section 1000 = ODOT Supplement — directly modifies/supersedes AASHTO articles.
"""

import os
import re
import sys
from pathlib import Path
from textwrap import dedent

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
BDM_STORE    = ROOT / "vector_store" / "bdm"
AASHTO_STORE = ROOT / "vector_store" / "aashto"

BDM_TOP_K    = 10
AASHTO_TOP_K = 10

# L2 distance threshold (0=identical, 2=opposite); excludes off-topic chunks.
SCORE_THRESHOLD = 0.95

# ── Override detection ────────────────────────────────────────────────────────
_OVERRIDE_PHRASES = [
    "in lieu of", "modify", "supersede", "supplement", "exception",
    "shall not", "not applicable", "ODOT Supplement", "Section 1000",
    "1000.", "replace", "instead of",
]
_SECTION_1000_RE = re.compile(r'\b1[0-9]{3}\.[0-9]')


def _has_odot_override(bdm_docs: list[Document]) -> bool:
    for doc in bdm_docs:
        text = doc.page_content
        if doc.metadata.get("is_supplement"):
            return True
        if _SECTION_1000_RE.search(text):
            return True
        if any(p.lower() in text.lower() for p in _OVERRIDE_PHRASES):
            return True
    return False


def _is_commentary(doc: Document) -> bool:
    if doc.metadata.get("is_commentary"):
        return True
    return bool(re.search(r'\bC\d{3}', doc.page_content[:200]))


# ── Article extraction ──────────────────────────────────────────────────────

def _extract_article(text: str, source_tag: str) -> str:
    """Return the first article/section number found in text, or empty string."""
    if source_tag == "BDM":
        m = _BDM_ART_RE.search(text)
    else:
        m = _AASHTO_ART_RE.search(text)
    return m.group(1) if m else ""


# ── LLM relevance re-ranking ─────────────────────────────────────────────────

def _rerank_docs(docs: list, query: str, llm, source_tag: str) -> list:
    """
    Ask LLM to score each chunk 1-10 for relevance to query.
    Filter out chunks scoring < 4. Falls back to all docs on parse failure.
    """
    if not docs:
        return docs
    nl = chr(10)
    passages = nl.join(
        f"[{i+1}] {doc.page_content[:250].strip().replace(nl, ' ')}"
        for i, doc in enumerate(docs)
    )
    sep = nl + nl
    prompt = (
        f"You are a bridge engineering relevance filter.{nl}"
        f"Question: {query}{sep}"
        f"Rate each passage 1-10 (1=completely off-topic, 10=directly answers question).{nl}"
        f"Consider: wrong section (e.g. load rating vs design loads) = score <= 3.{nl}"
        f"Consider: wrong document type (e.g. load rating vs live load design) = score <= 2.{sep}"
        f"{passages}{sep}"
        f"Reply with ONLY comma-separated integers, one per passage (e.g. 8,2,7):"
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip().split(chr(10))[0]  # first line only
        scores = [int(s.strip()) for s in raw.split(",")]
        if len(scores) != len(docs):
            return docs
        kept = [doc for doc, score in zip(docs, scores) if score >= 4]
        return kept if kept else docs
    except Exception:
        return docs

# ── Score-filtered retrieval ─────────────────────────────────────────────────

def _retrieve_with_filter(db, query: str, k: int, threshold: float) -> list:
    """Fetch k*3 candidates, keep those within L2 threshold, return top k."""
    results = db.similarity_search_with_score(query, k=k * 3)
    filtered = [(doc, score) for doc, score in results if score <= threshold]
    if not filtered:
        filtered = results[:k]
    return [doc for doc, _ in filtered[:k]]


# ── Context & citation helpers ────────────────────────────────────────────────

def _format_docs(docs: list[Document], source: str) -> str:
    if not docs:
        return "(No relevant content retrieved)"
    parts = []
    for i, doc in enumerate(docs, 1):
        page  = doc.metadata.get("page", "?")
        label = ""
        if source == "BDM":
            label = " | COMMENTARY — not binding" if _is_commentary(doc) else " | BINDING SPEC"
        parts.append(
            f"[Excerpt {i} | Page {page}{label}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


def _docs_to_citations(docs: list[Document], source_tag: str) -> list[dict]:
    """
    Convert retrieved Document objects into citation dicts for the API response.
    page is stored 0-indexed by PyPDFLoader; convert to 1-indexed for PDF viewer.
    """
    seen, citations = set(), []
    for doc in docs:
        raw_page = doc.metadata.get("page", 0)
        page_1indexed = int(raw_page) + 1  # PDF #page= fragment is 1-indexed
        excerpt = doc.page_content[:180].strip().replace("\n", " ")
        key = (source_tag, page_1indexed)
        if key in seen:
            continue
        seen.add(key)
        is_comment = _is_commentary(doc)
        article = _extract_article(doc.page_content, source_tag)
        citations.append({
            "source":   source_tag,
            "page":     page_1indexed,
            "article":  article,
            "excerpt":  excerpt,
            "binding":  not is_comment,
        })
    return citations


# -- Prompt builder -----------------------------------------------------------

def _build_prompt(bdm_context, aashto_context, has_override, question):
    note = (
        " Note: BDM Section 1000 (ODOT Supplement) material retrieved."
        " Present it in [BDM Specification Text]."
        if has_override else ""
    )
    safe_bdm    = bdm_context.replace("{", "{{").replace("}", "}}")
    safe_aashto = aashto_context.replace("{", "{{").replace("}", "}}")
    safe_q      = question.replace("{", "{{").replace("}", "}}")
    sep = "=" * 67

    lines = [
        "You are a professional bridge design specification consultant for Ohio DOT projects.",
        f"Audience: Designer of Record.{note}",
        "",
        "RULE 1 - BOTH SPECS REQUIRED: BDM and AASHTO LRFD are independently required.",
        "Both must be satisfied. Do NOT state one supersedes the other.",
        "",
        "RULE 2 - VERBATIM: Quote WORD-FOR-WORD from the excerpts below.",
        "Do not paraphrase. Use exact article numbers from the source.",
        "",
        "RULE 3 - RELEVANCE: Only include text DIRECTLY relevant to the question.",
        "If an excerpt covers a different topic, exclude it and say so.",
        "",
        sep,
        "OHIO BDM - Retrieved Excerpts (quote verbatim)",
        sep,
        safe_bdm,
        "",
        sep,
        "AASHTO LRFD 9th Ed. - Retrieved Excerpts (quote verbatim)",
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
        "List ALL articles from BOTH documents that are directly relevant to the question (no limit on count):",
        "  - BDM X.X.X - short description",
        "  - AASHTO LRFD Article X.X.X.X - short description",
        "",
        "[BDM Specification Text]",
        "Quote relevant BDM text VERBATIM from the excerpts.",
        "  Format: BDM X.X.X: [exact quoted text]",
        "  Commentary: BDM C302.3 (Commentary - informational only): [exact text]",
        "  If nothing relevant: The retrieved BDM excerpts do not contain directly relevant requirements.",
        "",
        "[AASHTO LRFD Specification Text]",
        "Quote relevant AASHTO text VERBATIM from the excerpts.",
        "  Format: AASHTO LRFD X.X.X.X: [exact quoted text]",
        "  If nothing relevant: The retrieved AASHTO excerpts do not contain directly relevant requirements.",
        "",
        "[Engineering Practice Guidance]",
        "Actionable design guidance for an Ohio DOT project.",
        "Apply BOTH BDM and AASHTO requirements - both are required.",
        "Reference specific article numbers.",
        "",
        "Rules:",
        "- Do not fabricate requirements not in the excerpts.",
        "- Do NOT state BDM overrides AASHTO.",
        "- If retrieved content is off-topic, say so clearly.",
    ]
    return chr(10).join(lines)


class DualStoreRAG:
    """
    Dual vector store RAG with multi-turn conversation support.

    ask(question, history) returns:
        {
            "answer":          str,   # full LLM response text
            "has_override":    bool,
            "bdm_citations":   list[{source, page, excerpt, binding}],
            "aashto_citations":list[{source, page, excerpt, binding}],
        }
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model, temperature=temperature)

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

        print("[OK] RAG engine ready.\n")

    # ── History-aware question reformulation ──────────────────────────────────

    def _reformulate(self, question: str, history: list[dict]) -> str:
        """
        Given conversation history, reformulate a follow-up question into a
        standalone question that can be answered without the prior context.
        Uses the last 6 messages (3 turns) to keep the prompt concise.
        """
        recent = history[-6:]
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        )
        prompt = (
            "Given this conversation history:\n"
            f"{history_text}\n\n"
            "Reformulate the following follow-up question as a concise, standalone "
            "engineering question that can be understood without the conversation history. "
            "Preserve all technical terms and article numbers. "
            "If the question is already standalone, return it unchanged.\n\n"
            f"Follow-up question: {question}\n"
            "Standalone question:"
        )
        response = self.llm.invoke(prompt)
        reformulated = response.content.strip()
        return reformulated if reformulated else question

    # ── Main entry point ──────────────────────────────────────────────────────

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """
        Run dual-store retrieval and return a structured response dict.

        Args:
            question: Current user question.
            history:  List of {"role": "user"|"assistant", "content": str}
                      representing prior turns. Pass [] or None for first turn.

        Returns:
            {
                "answer":           str,
                "has_override":     bool,
                "bdm_citations":    list[dict],
                "aashto_citations": list[dict],
                "standalone_q":     str,   # question actually sent to retrieval
            }
        """
        # Step 0: reformulate if this is a follow-up question
        if history:
            print("  -> Reformulating follow-up question ...")
            standalone_q = self._reformulate(question, history)
            if standalone_q != question:
                print(f"     Original    : {question}")
                print(f"     Reformulated: {standalone_q}")
        else:
            standalone_q = question

        # Step 1: Score-filtered retrieval from both stores
        print("  -> [1/2] Querying Ohio BDM vector store ...")
        bdm_docs = _retrieve_with_filter(
            self.bdm_db, standalone_q, BDM_TOP_K, SCORE_THRESHOLD
        )
        bdm_docs = _rerank_docs(bdm_docs, standalone_q, self.llm, "BDM")

        print("  -> [2/2] Querying AASHTO LRFD vector store ...")
        aashto_docs = _retrieve_with_filter(
            self.aashto_db, standalone_q, AASHTO_TOP_K, SCORE_THRESHOLD
        )
        aashto_docs = _rerank_docs(aashto_docs, standalone_q, self.llm, "AASHTO")

        # Step 2: Override detection (informational only)
        has_override = _has_odot_override(bdm_docs)
        if has_override:
            print("  [\!] BDM Section 1000 / ODOT supplement language detected.")

        # Step 3: Build prompt with both contexts and call LLM directly
        bdm_context    = _format_docs(bdm_docs,    "BDM")
        aashto_context = _format_docs(aashto_docs, "AASHTO")
        prompt_text    = _build_prompt(bdm_context, aashto_context,
                                       has_override, standalone_q)

        print("  -> Generating structured response ...")
        response = self.llm.invoke(prompt_text)

        return {
            "answer":           response.content,
            "has_override":     has_override,
            "bdm_citations":    _docs_to_citations(bdm_docs,    "BDM"),
            "aashto_citations": _docs_to_citations(aashto_docs, "AASHTO"),
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
            print("Session ended.")
            break
        if question.lower() == "clear":
            history = []
            print("Conversation cleared.\n")
            continue

        result = rag.ask(question, history)
        answer = result["answer"]

        print(f"\n{'─' * 70}\n{answer}")

        # Append to history (store raw text for assistant turn)
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})

        if result["bdm_citations"]:
            print("\nBDM Sources:", ", ".join(
                f"p.{c['page']}" for c in result["bdm_citations"]
            ))
        if result["aashto_citations"]:
            print("AASHTO Sources:", ", ".join(
                f"p.{c['page']}" for c in result["aashto_citations"]
            ))
        print("─" * 70)


if __name__ == "__main__":
    main()
