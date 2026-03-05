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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
BDM_STORE    = ROOT / "vector_store" / "bdm"
AASHTO_STORE = ROOT / "vector_store" / "aashto"

BDM_TOP_K    = 5
AASHTO_TOP_K = 5

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
        citations.append({
            "source":   source_tag,
            "page":     page_1indexed,
            "excerpt":  excerpt,
            "binding":  not is_comment,
        })
    return citations


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(bdm_context: str, has_override: bool) -> PromptTemplate:
    override_instruction = (
        "\n⚠️  IMPORTANT: BDM Section 1000 material (ODOT Supplement to LRFD) retrieved. "
        "Flag Ohio-specific modifications with: ⚠️ ODOT OVERRIDE — BDM X.X supersedes LRFD X.X.X"
        if has_override else ""
    )
    safe_bdm = bdm_context.replace("{", "{{").replace("}", "}}")

    template = dedent(f"""\
        You are a professional bridge design specification consultant for Ohio DOT projects.
        Audience: Designer of Record (American engineers).{override_instruction}

        DOCUMENT AUTHORITY (BDM 101.4): BDM > AASHTO LRFD for Ohio DOT projects.
        BDM left-column text: BINDING (imperative mood, addressed to Designer of Record).
        BDM C-numbered sections (e.g. C101.1): COMMENTARY ONLY — informational, not binding.
        BDM Section 1000 ("ODOT Supplement to LRFD"): modifies/supersedes specific AASHTO articles.
        AASHTO cross-references in BDM appear as: LRFD X.X.X

        ═══════════════════════════════════════════
        OHIO BDM — Retrieved Excerpts (State Authority)
        ═══════════════════════════════════════════
        {safe_bdm}

        ═══════════════════════════════════════════
        AASHTO LRFD 9th Ed. — Retrieved Excerpts
        ═══════════════════════════════════════════
        {{context}}

        ═══════════════════════════════════════════
        Designer's Question: {{question}}
        ═══════════════════════════════════════════

        Respond ONLY in this exact format (use these exact headers):

        [Article/Section Reference]
        List all relevant article/section numbers. Example format:
          • BDM 302.3  — Girder Spacing
          • AASHTO LRFD Article 6.10.3.2
        If BDM overrides AASHTO: ⚠️ ODOT OVERRIDE — BDM 1000.X supersedes LRFD X.X.X

        [Specification Summary]
        Summarize key requirements (150–250 words).
        Distinguish binding BDM spec text from BDM commentary (C-sections).
        Prefix Ohio-specific requirements with: ⚠️ Ohio-Specific Requirement:

        [Engineering Practice Guidance]
        Actionable design/construction guidance for an Ohio DOT project (200–350 words).
        Specify where BDM governs over AASHTO. Reference article numbers.

        Rules:
        - Do not fabricate article numbers or requirements.
        - Quote article numbers exactly as they appear in the source.
        - BDM commentary (C-sections) may be cited for context but is not binding.
        - If retrieved content is insufficient, state so clearly.
    """)
    return PromptTemplate(input_variables=["context", "question"], template=template)


# ── Main RAG engine ───────────────────────────────────────────────────────────

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

        self.bdm_retriever = self.bdm_db.as_retriever(
            search_kwargs={"k": BDM_TOP_K}
        )
        self.aashto_retriever = self.aashto_db.as_retriever(
            search_kwargs={"k": AASHTO_TOP_K}
        )
        print("[✓] RAG engine ready.\n")

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
            print("  → Reformulating follow-up question ...")
            standalone_q = self._reformulate(question, history)
            if standalone_q != question:
                print(f"     Original : {question}")
                print(f"     Reformulated: {standalone_q}")
        else:
            standalone_q = question

        # Step 1: BDM retrieval
        print("  → [1/2] Querying Ohio BDM vector store ...")
        bdm_docs    = self.bdm_retriever.get_relevant_documents(standalone_q)
        bdm_context = _format_docs(bdm_docs, "BDM")

        # Step 2: Override detection
        has_override = _has_odot_override(bdm_docs)
        if has_override:
            print("  ⚠️  BDM Section 1000 / ODOT override language detected.")

        # Step 3: Build dynamic prompt with BDM context embedded
        prompt = _build_prompt(bdm_context, has_override)

        # Step 4: RetrievalQA — AASHTO retrieval + LLM generation
        print("  → [2/2] Querying AASHTO LRFD vector store (RetrievalQA) ...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.aashto_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        result      = qa_chain.invoke({"query": standalone_q})
        aashto_docs = result.get("source_documents", [])

        return {
            "answer":           result["result"],
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
