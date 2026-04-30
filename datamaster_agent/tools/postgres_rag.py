"""
tools/postgres_rag.py
──────────────────────
pgvector-backed vector store operations:
  - ingest_document: embed and store a document chunk
  - similarity_search: cosine ANN search returning top-k chunks
  - multi_query_search: generate N query variants, search, then RRF-rerank
  - ingest_from_url: crawl a Talon.One docs page and ingest it

Uses psycopg3 directly for fine-grained control over the vector type.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import psycopg
from psycopg.rows import dict_row

from core.state import RAGChunk
from tools.postgres_rag_helper import get_conn, RAG_AVAILABLE

logger = logging.getLogger(__name__)

_RAG_AVAILABLE = False

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://presales:presales_secret@localhost:5432/presales_db",
)
# psycopg3 uses postgresql:// (no +psycopg suffix)
_PSYCOPG_DSN = _DSN.replace("postgresql+psycopg://", "postgresql://")

# Chunk defaults (shared across all ingestion paths)
CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "800"))
CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "80"))


def chunk_text(text: str, size: int = 0, overlap: int = 0) -> list[str]:
    """Split text into word-based chunks with overlap."""
    size = size or CHUNK_SIZE_WORDS
    overlap = overlap or CHUNK_OVERLAP_WORDS
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# Embedding helper (cached to avoid duplicate API calls)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=2048)
def _embed_cached(text: str) -> tuple[float, ...]:
    """Return an embedding vector (as tuple for hashability)."""
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedder = GoogleGenerativeAIEmbeddings(model=model, output_dimensionality=768)
    return tuple(embedder.embed_query(text))


def _embed(text: str) -> list[float]:
    """Return an embedding vector for the given text using Gemini text-embedding-004."""
    return list(_embed_cached(text))


# ─────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────

def ingest_document(
    content: str,
    source: str,
    source_id: str,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    """
    Embed and store a document chunk.

    Returns the newly created row ID.
    """
    vector = _embed(content)
    meta_json = json.dumps(metadata or {})

    if not RAG_AVAILABLE:
        logger.warning("[pgvector] ingest_document skipped: DB unavailable")
        return -1

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source, source_id, title, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                RETURNING id
                """,
                (source, source_id, title, content, meta_json, vector),
            )
            row = cur.fetchone()
            conn.commit()

    doc_id: int = row["id"]  # type: ignore[index]
    logger.info("[pgvector] Ingested doc id=%d source=%s", doc_id, source)
    return doc_id


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────

def similarity_search(query: str, top_k: int = 8) -> list[RAGChunk]:
    """
    Cosine ANN search against the documents table.

    Returns up to top_k chunks ordered by descending similarity.
    """
    if not RAG_AVAILABLE:
        logger.warning("[pgvector] similarity_search fallback: no DB available")
        return []

    vector = _embed(query)
    vector_literal = "[" + ",".join(str(v) for v in vector) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    source,
                    source_id,
                    content,
                    1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal, vector_literal, top_k),
            )
            rows = cur.fetchall()

    chunks: list[RAGChunk] = [
        RAGChunk(
            content=row["content"],
            source=row["source"],
            source_id=row["source_id"] or "",
            score=float(row["score"]),
        )
        for row in rows
    ]

    logger.info("[pgvector] Query '%s' → %d chunks", query[:60], len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────
# Multi-query + RRF reranking (best-of-breed RAG)
# ─────────────────────────────────────────────────────────────

def multi_query_search(
    queries: list[str],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[RAGChunk]:
    """
    Run multiple queries against pgvector, then fuse results with
    Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = Σ  1 / (rrf_k + rank_i)
    This eliminates the need for a cross-encoder reranker while still
    producing better results than any single query.

    Returns top_k chunks ordered by fused score.
    """
    if not queries:
        return []

    # Collect per-query ranked lists
    ranked_lists: list[list[RAGChunk]] = []
    for q in queries:
        ranked_lists.append(similarity_search(q, top_k=top_k * 2))

    # RRF fusion
    scores: dict[str, float] = {}
    chunk_map: dict[str, RAGChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            uid = f"{chunk['source_id']}:{chunk['content'][:80]}"
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (rrf_k + rank)
            # Keep the chunk with the highest original score
            if uid not in chunk_map or chunk["score"] > chunk_map[uid]["score"]:
                chunk_map[uid] = chunk

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda uid: scores[uid], reverse=True)[:top_k]

    fused: list[RAGChunk] = []
    for uid in sorted_ids:
        c = chunk_map[uid]
        fused.append(
            RAGChunk(
                content=c["content"],
                source=c["source"],
                source_id=c["source_id"],
                score=round(scores[uid], 4),
            )
        )

    logger.info(
        "[pgvector] Multi-query RRF: %d queries → %d raw → %d fused",
        len(queries),
        sum(len(r) for r in ranked_lists),
        len(fused),
    )
    return fused


def is_url_ingested(url: str) -> bool:
    """Check if a URL has already been ingested into the knowledge base."""
    try:
        with psycopg.connect(_PSYCOPG_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE source_id = %s LIMIT 1",
                    (url,),
                )
                return cur.fetchone() is not None
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Convenience: ingest a web URL (Talon.One docs)
# ─────────────────────────────────────────────────────────────

def ingest_from_url(url: str, source_id: str | None = None) -> list[int]:
    """
    Fetch a documentation URL, strip HTML tags, chunk, and ingest.

    Returns the list of new document row IDs.
    """
    import re
    import httpx

    resp = httpx.get(url, timeout=15, follow_redirects=True)
    resp.raise_for_status()

    # Minimal HTML stripping — good enough for doc pages
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    chunks = chunk_text(text)
    ids: list[int] = []
    for i, chunk_piece in enumerate(chunks):
        doc_id = ingest_document(
            content=chunk_piece,
            source="talon_docs",
            source_id=f"{source_id or url}#chunk-{i}",
            title=url,
        )
        ids.append(doc_id)

    logger.info("[pgvector] Ingested %d chunks from %s", len(ids), url)
    return ids
