"""Helper utilities for robust pgvector connectivity with fallback."""

from __future__ import annotations

import logging
import os
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://presales:presales_secret@localhost:5432/presales_db",
)
_PSYCOPG_DSN = _DSN.replace("postgresql+psycopg://", "postgresql://")

RAG_AVAILABLE = False


def get_conn():
    global RAG_AVAILABLE
    try:
        conn = psycopg.connect(_PSYCOPG_DSN, row_factory=dict_row)
        RAG_AVAILABLE = True
        return conn
    except Exception as exc:
        RAG_AVAILABLE = False
        logger.warning("[pgvector] DB connection failed: %s (falling back to empty results)", exc)
        raise
