"""
=============================================================================
WALMART COMPLIANCE MANUAL — RAG CHATBOT (PDF-NATIVE VERSION)
=============================================================================

INSTALL:
  pip install pymupdf openai chromadb rank-bm25 python-dotenv

USAGE:
  1. Place the Walmart PDF in the same folder as this script
  2. Set OPENAI_API_KEY in your .env file
  3. Run: python walmart_rag_v2.py

WHAT'S DIFFERENT FROM V1:
  ❌ V1: Hardcoded extracted text pasted directly into Python string
  ✅ V2: PyMuPDF reads the actual PDF file at runtime
  ✅ V2: Chunks by SECTIONS (Roman numeral headers) not just character count
  ✅ V2: Saves chunks to chunks.json locally so you can inspect them
  ✅ V2: Full ChromaDB + BM25 + RRF + GPT-4o-mini pipeline unchanged
=============================================================================
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import fitz                    
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1 — PDF EXTRACTION WITH PyMuPDF
# =============================================================================
"""
CONCEPT: WHY PyMuPDF (fitz)?

PyPDF2 is older and struggles with:
  - Multi-column layouts (merges columns incorrectly)
  - Tables (produces garbled text)
  - Font-encoded PDFs (misses characters)

PyMuPDF uses the MuPDF rendering engine — the same engine used by
production PDF viewers. It gives you:
  - Page-by-page text with correct reading order
  - Font size and font name per text block (useful for detecting headers)
  - Bounding box coordinates of every text block
  - Clean Unicode output

We use font size to DETECT SECTION HEADERS — Walmart's PDF uses larger,
bolder text for Roman numeral section titles (I., II., III., etc.)
"""

def extract_pdf_with_metadata(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page by page using PyMuPDF.
    Returns a list of text blocks with page number and font metadata.

    Each block looks like:
    {
        "page": 1,
        "text": "I. Introduction",
        "font_size": 14.0,
        "font_name": "Arial-Bold",
        "is_bold": True,
        "bbox": (x0, y0, x1, y1)
    }
    """
    doc = fitz.open(pdf_path)
    blocks = []

    print(f"📄 PDF loaded: {doc.page_count} pages")

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # get_text("dict") returns structured block data with font info
        # This is the key PyMuPDF advantage over PyPDF2
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            # blocks can be "image" type — we only want "text" type
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    blocks.append({
                        "page": page_num + 1,
                        "text": text,
                        "font_size": round(span["size"], 1),
                        "font_name": span["font"],
                        "is_bold": "bold" in span["font"].lower() or span["flags"] & 2**4,
                        "bbox": span["bbox"]
                    })
    page_count = doc.page_count
    doc.close()
    print(f"✅ Extracted {len(blocks)} text spans across {page_count} pages")
    return blocks


# =============================================================================
# STEP 2 — SECTION-BASED CHUNKING
# =============================================================================
"""
CONCEPT: SECTION-BASED CHUNKING vs CHARACTER-BASED CHUNKING

CHARACTER-BASED (V1 approach — what we replaced):
  Split every N characters with overlap.
  Problem: Cuts mid-sentence, mid-paragraph, ignores document structure.

  "...The CAP must be approved before submitting the corrected product for re"
  "test. The retest sample must be submitted within 3 months..."
  ↑ The sentence "approved before retest" is split across two chunks.

SECTION-BASED (V2 approach):
  Detect section headers from the PDF's font metadata.
  Split at actual section boundaries the document authors intended.
  
  Section headers detected by:
  1. Roman numeral pattern: "I.", "II.", "III.", "IV.", "V."
  2. Letter subsection pattern: "A.", "B.", "C.", "D."
  3. Large font size (> median font size of document)
  4. Bold font flag from PyMuPDF metadata

  Result: Each chunk = one complete section = semantically coherent unit.
  The retriever gets "Failure Management Program" as one clean chunk,
  not split across 3 character-boundary chunks.

WHEN SECTIONS ARE TOO LONG:
  Some sections (like Section III with all testing cycles) are very long.
  We apply a MAX_CHUNK_CHARS limit — if a section exceeds it, we split
  it into sub-chunks with overlap to preserve context at split boundaries.
"""

# Regex patterns for Walmart compliance manual header detection
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+.+", re.IGNORECASE
)
LETTER_SUBSECTION_PATTERN = re.compile(
    r"^[A-Z]\.\s+.{3,}"
)
NUMBERED_SUBSECTION_PATTERN = re.compile(
    r"^\d+\.\s+.{3,}"
)

MAX_CHUNK_CHARS = 1200   # Max characters per chunk before sub-splitting
OVERLAP_CHARS   = 150    # Overlap when sub-splitting long sections


def is_section_header(block: Dict[str, Any], median_font_size: float) -> bool:
    """
    Determine if a text block is a section header.

    Detection logic (all must pass):
    - Font size >= median + 1pt  OR  block is bold
    - Text matches Roman numeral or letter subsection pattern
    - Text is not too short (avoids catching stray large characters)
    """
    text = block["text"]
    font_size = block["font_size"]
    is_bold = block["is_bold"]

    size_is_large = font_size >= median_font_size + 1.0
    matches_pattern = (
        ROMAN_NUMERAL_PATTERN.match(text) or
        LETTER_SUBSECTION_PATTERN.match(text)
    )
    long_enough = len(text) > 5

    return bool(matches_pattern and (size_is_large or is_bold) and long_enough)


def sub_chunk(text: str, section_id: str, header: str,
              max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Split a long section into overlapping sub-chunks.
    Used when a section exceeds MAX_CHUNK_CHARS.
    """
    sub_chunks = []
    start = 0
    sub_idx = 0

    while start < len(text):
        end = start + max_chars
        content = text[start:end].strip()
        if content:
            sub_chunks.append({
                "id": f"{section_id}-part{sub_idx:02d}",
                "header": header,
                "content": content,
                "sub_chunk": sub_idx,
                "char_start": start,
                "char_end": min(end, len(text)),
            })
            sub_idx += 1
        start += max_chars - overlap

    return sub_chunks


def chunk_by_sections(
    blocks: List[Dict[str, Any]],
    pdf_name: str
) -> List[Dict[str, Any]]:
    """
    Group PDF text blocks into section-based chunks.

    Algorithm:
    1. Compute median font size across all blocks (baseline for header detection)
    2. Walk through blocks sequentially
    3. When a header is detected → start a new section
    4. Accumulate text spans into the current section
    5. When section text exceeds MAX_CHUNK_CHARS → sub-split with overlap
    6. Return list of chunk dicts with full metadata

    Returns list of chunks:
    {
        "id": "walmart-III-B-part00",
        "content": "B. Brand Types and Supplier Designations...",
        "metadata": {
            "source": "walmart_compliance.pdf",
            "section_header": "B. Brand Types and Supplier Designations",
            "page_start": 4,
            "page_end": 5,
            "chunk_type": "section"
        }
    }
    """
    # Compute median font size for header detection threshold
    font_sizes = [b["font_size"] for b in blocks if b["text"].strip()]
    font_sizes.sort()
    median_font = font_sizes[len(font_sizes) // 2] if font_sizes else 11.0
    print(f"   Median font size: {median_font}pt (used for header detection)")

    # Group blocks into sections
    sections = []          # list of {header, page_start, page_end, lines[]}
    current_section = {
        "header": "Preamble",
        "page_start": 1,
        "page_end": 1,
        "lines": []
    }

    for block in blocks:
        text = block["text"]
        page = block["page"]

        if is_section_header(block, median_font):
            # Save current section if it has content
            if current_section["lines"]:
                sections.append(current_section)
            # Start new section
            current_section = {
                "header": text,
                "page_start": page,
                "page_end": page,
                "lines": []
            }
        else:
            current_section["lines"].append(text)
            current_section["page_end"] = page

    # Don't forget the last section
    if current_section["lines"]:
        sections.append(current_section)

    print(f"   Detected {len(sections)} sections from headers")

    # Convert sections → chunks (sub-splitting if needed)
    chunks = []
    for sec_idx, section in enumerate(sections):
        full_text = " ".join(section["lines"])
        full_text = re.sub(r"\s+", " ", full_text).strip()

        if not full_text:
            continue

        header = section["header"]
        # Clean section ID for use as ChromaDB document ID
        section_id = f"walmart-sec{sec_idx:03d}"

        base_metadata = {
            "source": pdf_name,
            "section_header": header,
            "page_start": section["page_start"],
            "page_end": section["page_end"],
            "section_index": sec_idx,
            "chunk_type": "section"
        }

        if len(full_text) <= MAX_CHUNK_CHARS:
            # Section fits in one chunk
            chunks.append({
                "id": section_id,
                "content": f"{header}\n\n{full_text}",
                "metadata": {**base_metadata, "sub_chunk": 0, "total_sub_chunks": 1}
            })
        else:
            # Section too long — sub-split with overlap
            sub_chunks = sub_chunk(
                full_text, section_id, header, MAX_CHUNK_CHARS, OVERLAP_CHARS
            )
            for sc in sub_chunks:
                chunks.append({
                    "id": sc["id"],
                    "content": f"{header}\n\n{sc['content']}",
                    "metadata": {
                        **base_metadata,
                        "sub_chunk": sc["sub_chunk"],
                        "total_sub_chunks": len(sub_chunks),
                        "char_start": sc["char_start"],
                        "char_end": sc["char_end"],
                    }
                })

    print(f"✅ Created {len(chunks)} section-based chunks")
    return chunks


# =============================================================================
# STEP 3 — SAVE CHUNKS LOCALLY AS JSON
# =============================================================================
"""
CONCEPT: WHY SAVE CHUNKS TO JSON?

Two reasons:

1. TRANSPARENCY — You can open chunks.json and see exactly what went
   into ChromaDB. If retrieval is wrong, you inspect the JSON first.
   You can see: did the chunking split a section correctly?
   Did the header detection work? Is there missing text?

2. CACHING — PDF extraction + chunking takes time. By saving to JSON,
   on subsequent runs you can skip re-extraction and load directly.
   Especially important when processing hundreds of SEC filing PDFs.

The saved JSON is human-readable and maps 1:1 to what ChromaDB stores.
"""

CHUNKS_FILE = "walmart_chunks.json"

def save_chunks(chunks: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    size_kb = Path(path).stat().st_size / 1024
    print(f"💾 Saved {len(chunks)} chunks → {path} ({size_kb:.1f} KB)")
    print(f"   Open {path} to inspect chunk content and metadata")


def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"📂 Loaded {len(chunks)} chunks from {path}")
    return chunks


def preview_chunks(chunks: List[Dict[str, Any]], n: int = 5) -> None:
    """Print a preview of the first n chunks to show what was extracted."""
    print(f"\n{'='*60}")
    print(f"CHUNK PREVIEW (first {n} of {len(chunks)})")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks[:n]):
        meta = chunk["metadata"]
        print(f"\n[Chunk {i+1}] ID: {chunk['id']}")
        print(f"  Header   : {meta['section_header']}")
        print(f"  Pages    : {meta['page_start']} → {meta['page_end']}")
        print(f"  Sub-chunk: {meta['sub_chunk']} of {meta['total_sub_chunks']}")
        print(f"  Length   : {len(chunk['content'])} chars")
        print(f"  Preview  : {chunk['content'][:120].strip()}...")
    print(f"{'='*60}\n")


# =============================================================================
# STEP 4 — DATA MODEL
# =============================================================================

@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    retrieval_method: str   # "sparse_bm25" | "dense_semantic" | "hybrid_rrf"


# =============================================================================
# STEP 5 — HYBRID RAG PIPELINE (BM25 + ChromaDB + RRF + GPT-4o-mini)
# =============================================================================

class WalmartRAGPipeline:

    def __init__(self, openai_api_key: str):
        print("\n🔧 Initializing Walmart RAG Pipeline...")
        self.openai_client = OpenAI(api_key=openai_api_key)

        # BM25 sparse index state
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_store: List[Dict[str, Any]] = []

        # ChromaDB dense index
        self.chroma_client = chromadb.Client()

        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )

        try:
            self.chroma_client.delete_collection("walmart_compliance_v2")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name="walmart_compliance_v2",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        print("✅ ChromaDB collection ready (cosine space, text-embedding-3-small)")
        print("✅ BM25 index ready (will build on index_documents)")

    # ── TOKENIZER ──────────────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    # ── DUAL INDEXING ──────────────────────────────────────────────────────────
    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index chunks into BOTH BM25 and ChromaDB simultaneously.

        BM25:
          - Tokenizes each chunk into words
          - Builds inverted index + IDF weights
          - Lives in memory (Python object)

        ChromaDB:
          - Sends each chunk to OpenAI embedding API
          - Stores 1536-dim vectors in HNSW graph
          - Persists in memory (for this session)
        """
        print(f"\n📥 Indexing {len(chunks)} section-based chunks...")

        self.chunk_store = chunks

        # BM25 — tokenize and build index
        tokenized_corpus = [self._tokenize(chunk["content"]) for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # ChromaDB — batch add (triggers OpenAI embedding API calls)
        # ChromaDB accepts max 5000 items per .add() call
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                documents=[c["content"] for c in batch],
                ids=[c["id"] for c in batch],
                metadatas=[c["metadata"] for c in batch]
            )
            print(f"   Embedded batch {i // batch_size + 1} "
                  f"({min(i + batch_size, len(chunks))}/{len(chunks)} chunks)")

        print(f"\n✅ BM25 vocabulary : {len(self.bm25_index.idf):,} unique terms")
        print(f"✅ ChromaDB vectors: {self.collection.count()} stored")

    # ── SPARSE RETRIEVAL (BM25) ────────────────────────────────────────────────
    def sparse_search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            RetrievedChunk(
                chunk_id=self.chunk_store[i]["id"],
                content=self.chunk_store[i]["content"],
                score=float(scores[i]),
                metadata=self.chunk_store[i]["metadata"],
                retrieval_method="sparse_bm25"
            )
            for i in top_indices
        ]

    # ── DENSE RETRIEVAL (ChromaDB) ─────────────────────────────────────────────
    def dense_search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count())
        )

        retrieved = []
        for chunk_id, content, distance, metadata in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            similarity = 1.0 - distance   # cosine distance → similarity
            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=content,
                score=similarity,
                metadata=metadata,
                retrieval_method="dense_semantic"
            ))

        return retrieved

    # ── RECIPROCAL RANK FUSION ─────────────────────────────────────────────────
    def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[RetrievedChunk]],
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Merge BM25 and ChromaDB rankings using rank positions only.
        Score = Σ 1 / (k + rank)   for each retriever list.
        Documents appearing in both lists get a combined boost.
        """
        rrf_scores: Dict[str, float] = {}
        registry: Dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank_idx, chunk in enumerate(ranked_list, start=1):
                rrf_scores[chunk.chunk_id] = (
                    rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank_idx)
                )
                registry[chunk.chunk_id] = chunk

        fused = []
        for chunk_id, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            orig = registry[chunk_id]
            fused.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=orig.content,
                score=rrf_score,
                metadata=orig.metadata,
                retrieval_method="hybrid_rrf"
            ))

        return fused

    # ── HYBRID SEARCH ──────────────────────────────────────────────────────────
    def hybrid_search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        sparse  = self.sparse_search(query, k=k)
        dense   = self.dense_search(query, k=k)
        fused   = self.reciprocal_rank_fusion([sparse, dense])

        print(f"   BM25 top  → [{sparse[0].chunk_id}] "
              f"'{sparse[0].metadata.get('section_header', '')[:50]}'")
        print(f"   Dense top → [{dense[0].chunk_id}] "
              f"'{dense[0].metadata.get('section_header', '')[:50]}'")

        return fused[:k]

    # ── ANSWER GENERATION ──────────────────────────────────────────────────────
    def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Pass retrieved chunks as grounded context to GPT-4o-mini.
        System prompt enforces citation and prohibits hallucination.
        Each chunk is labeled with its section header for traceability.
        """
        context = "\n\n".join([
            f"[{chunk.chunk_id} | {chunk.metadata.get('section_header', 'Unknown')} "
            f"| Page {chunk.metadata.get('page_start', '?')} "
            f"| Score {chunk.score:.4f}]\n{chunk.content}"
            for chunk in chunks
        ])

        system_prompt = """You are a compliance assistant for Walmart's US Product Quality 
and Compliance Manual.

RULES:
1. Answer ONLY using the document chunks provided below.
2. Cite the chunk ID and section header for every claim e.g. [walmart-sec005 | F. Letters of Guarantee].
3. If the answer is not in the chunks, say exactly: "Not found in retrieved sections."
4. Use bullet points for lists. Be concise and precise.
5. Never invent information beyond what is explicitly in the chunks."""

        user_message = f"""RETRIEVED DOCUMENT CHUNKS:
{context}

QUESTION: {query}

Answer using only the chunks above. Cite chunk IDs and section headers."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=700
        )

        return response.choices[0].message.content

    # ── FULL PIPELINE ──────────────────────────────────────────────────────────
    def query(self, question: str, k: int = 4, verbose: bool = True) -> Dict[str, Any]:
        if verbose:
            print(f"\n🔍 Query: '{question}'")

        chunks  = self.hybrid_search(question, k=k)
        answer  = self.generate_answer(question, chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks
        }


# =============================================================================
# STEP 6 — RESULT DISPLAY
# =============================================================================

def print_result(result: Dict[str, Any]) -> None:
    """
    Display a RAG result with full evidence tracing.
    Each source shows: chunk ID, section header, page number, score, method.
    This is the 'glass box' output — every claim is traceable.
    """
    print("\n" + "═" * 70)
    print(f"❓  {result['question']}")
    print("═" * 70)
    print(f"\n💬  ANSWER:\n{result['answer']}")

    print("\n📚  EVIDENCE TRAIL:")
    print(f"    {'Rank':<5} {'Chunk ID':<25} {'Section':<35} {'Page':<6} {'Score':<8} {'Method'}")
    print(f"    {'─'*4} {'─'*24} {'─'*34} {'─'*5} {'─'*7} {'─'*15}")

    for i, chunk in enumerate(result["sources"], 1):
        section = chunk.metadata.get("section_header", "Unknown")[:33]
        page    = str(chunk.metadata.get("page_start", "?"))
        print(
            f"    {i:<5} {chunk.chunk_id:<25} {section:<35} "
            f"{page:<6} {chunk.score:<8.4f} {chunk.retrieval_method}"
        )

    print("═" * 70)


# =============================================================================
# STEP 7 — CHATBOT LOOP
# =============================================================================

DEMO_QUERIES = [
    ("BM25 wins — exact acronym",    "What is the SKPPT program?"),
    ("BM25 wins — exact term",       "What is WERCS used for?"),
    ("Dense wins — natural language","What happens if my product fails testing?"),
    ("Dense wins — conceptual",      "How does Walmart handle product recalls?"),
    ("Hybrid wins — mixed",          "What are the Letter of Guarantee requirements?"),
]

def run_chatbot(pipeline: WalmartRAGPipeline) -> None:
    print("\n" + "═" * 70)
    print("  WALMART COMPLIANCE RAG CHATBOT  (PDF-native, section-chunked)")
    print("  ChromaDB + BM25 + RRF + GPT-4o-mini")
    print("═" * 70)
    print("\n  Commands:")
    print("    ask any question about the Walmart compliance manual")
    print("    'demo'  → run 5 sample queries showing all retrieval strategies")
    print("    'exit'  → quit\n")

    while True:
        user_input = input("🧑 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        if user_input.lower() == "demo":
            print("\n🚀 Running demo queries...\n")
            for label, q in DEMO_QUERIES:
                print(f"\n── {label} ──")
                result = pipeline.query(q)
                print_result(result)
                input("  Press Enter for next query...")
            continue

        result = pipeline.query(user_input)
        print_result(result)


# =============================================================================
# MAIN
# =============================================================================

PDF_PATH = "u-s-product-quality-and-compliance-manual.pdf"

def main():
    # ── API KEY ────────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )

    # ── PDF EXTRACTION + CHUNKING ──────────────────────────────────────────────
    if Path(CHUNKS_FILE).exists():
        # Skip re-extraction if chunks already saved from a previous run
        print(f"\n📂 Found existing {CHUNKS_FILE} — loading cached chunks")
        print("   (Delete walmart_chunks.json to force re-extraction from PDF)")
        chunks = load_chunks(CHUNKS_FILE)
    else:
        # First run — extract from PDF
        if not Path(PDF_PATH).exists():
            raise FileNotFoundError(
                f"PDF not found at '{PDF_PATH}'.\n"
                f"Download it from:\n"
                f"https://corporate.walmart.com/content/dam/corporate/documents/"
                f"suppliers/requirements/compliance-areas/"
                f"u-s-product-quality-and-compliance-manual.pdf\n"
                f"Place it in the same folder as this script."
            )

        print(f"\n📄 Extracting text from PDF: {PDF_PATH}")
        blocks = extract_pdf_with_metadata(PDF_PATH)

        print(f"\n✂️  Chunking by sections...")
        chunks = chunk_by_sections(blocks, pdf_name=PDF_PATH)

        # Save locally for inspection and caching
        save_chunks(chunks, CHUNKS_FILE)

    # Show first 5 chunks so you can verify extraction quality
    preview_chunks(chunks, n=5)

    # ── BUILD PIPELINE ─────────────────────────────────────────────────────────
    pipeline = WalmartRAGPipeline(openai_api_key=api_key)
    pipeline.index_documents(chunks)

    print(f"\n✅ Pipeline ready — {len(chunks)} section-chunks indexed")

    # ── LAUNCH CHATBOT ─────────────────────────────────────────────────────────
    run_chatbot(pipeline)


if __name__ == "__main__":
    main()