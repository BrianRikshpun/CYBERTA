"""
CyberTA – RAG Pipeline
Handles: document ingestion (PDF / DOCX / PPTX), chunking,
open-source embeddings (sentence-transformers), ChromaDB vector store,
and multi-provider generation via llm_providers.
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

# ── Document parsers ──────────────────────────────────────────────────────────
import fitz                        # PyMuPDF  – PDF
from docx import Document          # python-docx – DOCX
from pptx import Presentation      # python-pptx – PPTX

# ── Embeddings (all open-source via sentence-transformers) ────────────────────
from sentence_transformers import SentenceTransformer

# ── Vector store ──────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings

# ── Multi-provider LLM ────────────────────────────────────────────────────────
from llm_providers import generate, SYSTEM_PROMPT, DEFAULT_MODEL

# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "cyberTA"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80
TOP_K           = 5

# ── Open-source embedding models (all run locally via sentence-transformers) ──
EMBED_MODELS: Dict[str, str] = {
    "MiniLM-L6-v2 (fast, 384d)"          : "all-MiniLM-L6-v2",
    "MiniLM-L12-v2 (balanced, 384d)"     : "all-MiniLM-L12-v2",
    "MPNet-base-v2 (best quality, 768d)" : "all-mpnet-base-v2",
    "BGE-small-en-v1.5 (fast, 384d)"     : "BAAI/bge-small-en-v1.5",
    "BGE-base-en-v1.5 (quality, 768d)"   : "BAAI/bge-base-en-v1.5",
    "E5-small-v2 (multilingual, 384d)"   : "intfloat/e5-small-v2",
    "E5-base-v2 (multilingual, 768d)"    : "intfloat/e5-base-v2",
    "GTE-small (384d)"                   : "thenlper/gte-small",
    "Nomic-embed-text-v1 (768d)"         : "nomic-ai/nomic-embed-text-v1",
}

DEFAULT_EMBED = "MiniLM-L6-v2 (fast, 384d)"

_embed_cache: Dict[str, SentenceTransformer] = {}
_chroma_client = None
_collection    = None
_current_embed_model_name: str = DEFAULT_EMBED


# ── Lazy singletons ───────────────────────────────────────────────────────────

def get_embed_model(display_name: str = DEFAULT_EMBED) -> SentenceTransformer:
    model_id = EMBED_MODELS.get(display_name, EMBED_MODELS[DEFAULT_EMBED])
    if model_id not in _embed_cache:
        _embed_cache[model_id] = SentenceTransformer(model_id)
    return _embed_cache[model_id]


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def _extract_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def _extract_pptx(path: str) -> str:
    prs = Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text)
    return "\n".join(parts)

def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":            return _extract_pdf(path)
    elif ext in (".docx",".doc"): return _extract_docx(path)
    elif ext in (".pptx",".ppt"): return _extract_pptx(path)
    else: raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_file(path: str, embed_model_name: str = DEFAULT_EMBED, progress_cb=None) -> Dict:
    filename = Path(path).name
    if progress_cb: progress_cb(f"📄 Extracting text from **{filename}** …")

    text   = extract_text(path)
    chunks = chunk_text(text)

    if progress_cb: progress_cb(f"✂️  {len(chunks)} chunks — embedding with **{embed_model_name}** …")

    model      = get_embed_model(embed_model_name)
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    collection = get_collection()
    ids        = [str(uuid.uuid4()) for _ in chunks]
    metadatas  = [{"source": filename, "chunk_index": i,
                   "embed_model": embed_model_name} for i, _ in enumerate(chunks)]

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=chunks, metadatas=metadatas)

    if progress_cb: progress_cb(f"✅ **{filename}** — {len(chunks)} chunks stored.")
    return {"filename": filename, "chunks": len(chunks), "chars": len(text)}


# ── Vector retrieval helpers ──────────────────────────────────────────────────

def get_all_vectors() -> Tuple[List, List, List]:
    col   = get_collection()
    if col.count() == 0: return [], [], []
    result = col.get(include=["embeddings", "documents", "metadatas"])
    return (result["embeddings"], result["documents"],
            [m["source"] for m in result["metadatas"]])


def collection_stats() -> Dict:
    col   = get_collection()
    count = col.count()
    if count == 0: return {"total_chunks": 0, "sources": []}
    result  = col.get(include=["metadatas"])
    sources = list({m["source"] for m in result["metadatas"]})
    return {"total_chunks": count, "sources": sources}


# ── RAG query ─────────────────────────────────────────────────────────────────

def rag_query(
    question: str,
    history: List[Dict],
    llm_display_name: str  = DEFAULT_MODEL,
    embed_display_name: str = DEFAULT_EMBED,
    top_k: int             = TOP_K,
) -> Tuple[str, List[str]]:
    """
    1. Embed question with chosen embedding model
    2. Retrieve top-k chunks from ChromaDB
    3. Build context
    4. Generate answer with chosen LLM
    Returns (answer, sources)
    """
    model = get_embed_model(embed_display_name)
    q_emb = model.encode([question]).tolist()

    col     = get_collection()
    results = col.query(
        query_embeddings=q_emb,
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks  = results["documents"][0]
    metas   = results["metadatas"][0]
    sources = list({m["source"] for m in metas})

    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}]\n{doc}"
        for doc, m in zip(chunks, metas)
    )

    # Build messages for the LLM
    messages = []
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    answer = generate(llm_display_name, SYSTEM_PROMPT, messages)
    return answer, sources


# ── Delete / reset ────────────────────────────────────────────────────────────

def reset_collection():
    global _chroma_client, _collection
    if _chroma_client is None: get_collection()
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = None
    get_collection()

def embed_model_list() -> List[str]:
    return list(EMBED_MODELS.keys())
