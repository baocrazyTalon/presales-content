"""
core/store.py
────────────
Persistent user insight store (sales playbook) for long-term agent memory.

This complements the existing pgvector document store (hard facts) with a user-specific
`user_insights` table used by the VSP coordinator + data architect.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from tools.postgres_rag import _embed

logger = logging.getLogger(__name__)

_STORE_AVAILABLE = False
_IN_MEMORY_INSIGHTS: list[dict[str, Any]] = []

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://presales:presales_secret@localhost:5432/presales_db",
)
_PSYCOPG_DSN = _DSN.replace("postgresql+psycopg://", "postgresql://")


def setup_store():
    """Ensure the user_insights table exists with embedding support."""
    global _STORE_AVAILABLE
    try:
        with psycopg.connect(_PSYCOPG_DSN, autocommit=True, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_insights (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        thread_id TEXT,
                        content TEXT NOT NULL,
                        tags JSONB,
                        embedding VECTOR(768) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """,
                )
                # Vector index for similarity search
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_user_insights_embedding
                    ON user_insights USING ivfflat (embedding vector_cosine_ops)
                    """,
                )
                _STORE_AVAILABLE = True
                logger.info("[Store] user_insights table ready")
    except Exception as exc:
        _STORE_AVAILABLE = False
        logger.warning("[Store] Could not set up user_insights table: %s", exc)
        logger.warning("[Store] Falling back to in-memory store")


def get_store():
    """Convenience helper to ensure the store is available."""
    setup_store()
    return True  # placeholder to indicate store is ready


def record_user_preference(
    user_id: str,
    namespace: str,
    insight: str,
    thread_id: str | None = None,
    tags: list[str] | None = None,
) -> int:
    """Persist a user preference/insight in the sales_playbook namespace."""
    if not user_id or not insight:
        raise ValueError("user_id and insight are required")

    vector = _embed(insight)
    tags_json = json.dumps(tags or [])

    if not _STORE_AVAILABLE:
        insight_id = len(_IN_MEMORY_INSIGHTS) + 1
        _IN_MEMORY_INSIGHTS.append({
            "id": insight_id,
            "user_id": user_id,
            "namespace": namespace,
            "thread_id": thread_id,
            "content": insight,
            "tags": tags or [],
            "embedding": vector,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
        logger.info("[Store] Saved insight in-memory id=%d user=%s namespace=%s", insight_id, user_id, namespace)
        return insight_id

    with psycopg.connect(_PSYCOPG_DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_insights
                    (user_id, namespace, thread_id, content, tags, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                RETURNING id
                """,
                (user_id, namespace, thread_id, insight, tags_json, vector),
            )
            row = cur.fetchone()
            conn.commit()

    insight_id = row["id"]
    logger.info("[Store] Saved insight id=%d user=%s namespace=%s", insight_id, user_id, namespace)
    return insight_id


def query_user_insights(
    user_id: str,
    namespace: str = "sales_playbook",
    query: str = "",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Semantic lookup for user insights for dynamic context injection."""
    if not user_id:
        return []

    if not query:
        if not _STORE_AVAILABLE:
            filtered = [i for i in _IN_MEMORY_INSIGHTS if i["user_id"] == user_id and i["namespace"] == namespace]
            return sorted(filtered, key=lambda d: d["id"], reverse=True)[:top_k]

        # fetch recent items if no query provided
        with psycopg.connect(_PSYCOPG_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, tags, thread_id, created_at
                    FROM user_insights
                    WHERE user_id = %s AND namespace = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, namespace, top_k),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    vector = _embed(query)
    vector_literal = "[" + ",".join(str(v) for v in vector) + "]"

    if not _STORE_AVAILABLE:
        filtered = [i for i in _IN_MEMORY_INSIGHTS if i["user_id"] == user_id and i["namespace"] == namespace]
        # simple keyword match fallback
        scored = []
        for item in filtered:
            score = 1.0 if query.lower() in item["content"].lower() else 0.0
            scored.append((score, item))
        scored = sorted(scored, key=lambda t: t[0], reverse=True)[:top_k]
        return [item for _, item in scored]

    with psycopg.connect(_PSYCOPG_DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, tags, thread_id, created_at,
                       1 - (embedding <=> %s::vector) AS score
                FROM user_insights
                WHERE user_id = %s AND namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal, user_id, namespace, vector_literal, top_k),
            )
            rows = cur.fetchall()

    insights = [dict(r) for r in rows]
    logger.info("[Store] query '%s' -> %d insights", query[:60], len(insights))
    return insights
