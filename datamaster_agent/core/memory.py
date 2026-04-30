"""
core/memory.py
──────────────
LangGraph persistent memory via PostgresSaver.

PostgreSQL serves triple duty in this project:
  ① Vector store  — pgvector `documents` table        (semantic / long-term memory)
  ② Graph state   — LangGraph checkpoint tables        (working / cross-run memory)
  ③ Output store  — `presentations` + `agent_runs`    (audit log / replay)

Memory layers
─────────────
  SHORT-TERM  :  `AgentState.messages` — the full conversation turn list,
                 accumulated within a single graph invocation via the
                 `add_messages` reducer.  Automatically held in the checkpoint.

  WORKING     :  The entire `AgentState` (tasks, retrieved_chunks, html_output,
                 validation, etc.) is serialised into the PostgreSQL checkpoint
                 table after every node transition.  If the process crashes or
                 the user resumes later, `graph.invoke(None, config=config)`
                 replays from the last saved state — no work is lost.

  LONG-TERM   :  The `documents` table (pgvector).  Every ingested chunk from
                 Talon.One docs, Google Drive, and Notion is embedded and stored
                 permanently.  The Data Architect queries this table on every
                 run via cosine ANN search — this IS the agent's long-term
                 semantic memory.

Thread model
────────────
  Each prospect gets a stable `thread_id` derived from their company slug.
  e.g. "My Muscle Chef" → "my-muscle-chef"
  Revisiting the same prospect resumes from the last checkpoint, letting the
  agent accumulate context across multiple presales sessions.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://presales:presales_secret@localhost:5432/presales_db",
)
# psycopg3 native driver uses postgresql:// (no +psycopg suffix)
_PSYCOPG_DSN = _DSN.replace("postgresql+psycopg://", "postgresql://")


# ─────────────────────────────────────────────────────────────
# Thread-ID helper
# ─────────────────────────────────────────────────────────────

def make_thread_id(company_name: str) -> str:
    """
    Derive a stable, URL-safe thread ID from a prospect company name.

        "My Muscle Chef"  →  "my-muscle-chef"
        "DALI / France"   →  "dali-france"

    The same thread_id on a future run causes LangGraph to resume from the
    last saved checkpoint rather than starting a fresh graph invocation.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    return slug or "default"


# ─────────────────────────────────────────────────────────────
# Checkpointer factory
# ─────────────────────────────────────────────────────────────

def get_checkpointer():
    """
    Return a PostgresSaver instance backed by the same PostgreSQL DB used
    for pgvector.

    On first call, `checkpointer.setup()` is invoked — it creates the
    LangGraph internal tables (`langgraph_checkpoints`, `langgraph_blobs`,
    `langgraph_writes`) idempotently via IF NOT EXISTS.

    Graceful degradation:  if the DB is not reachable (e.g. Docker not
    running during local development), the function logs a warning and
    returns None.  `build_graph().compile(checkpointer=None)` creates an
    in-memory-only graph — the agent still works, just without persistence.
    """
    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from core.store import setup_store

        # Initialize the long-term insight store table in parallel with checkpoint persistence.
        setup_store()

        conn = psycopg.connect(_PSYCOPG_DSN, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()  # idempotent — safe to call every startup
        logger.info("[Memory] PostgresSaver checkpointer initialised (thread-safe)")
        return checkpointer

    except Exception as exc:
        logger.warning(
            "[Memory] DB not reachable (%s) — running without checkpoint persistence. "
            "Start PostgreSQL to enable cross-run memory.",
            exc,
        )
        return None


# ─────────────────────────────────────────────────────────────
# Checkpoint introspection helpers
# ─────────────────────────────────────────────────────────────

def get_thread_history(graph, thread_id: str) -> list[dict]:
    """
    Return a list of past checkpoint summaries for a given thread.

    Useful for the UI to show "previous sessions for this prospect".
    Each dict has: step, ts, agent, iteration_count.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = []
    try:
        for snapshot in graph.get_state_history(config):
            state = snapshot.values
            history.append(
                {
                    "step": snapshot.config["configurable"].get("checkpoint_id", "?"),
                    "ts": snapshot.metadata.get("created_at", ""),
                    "agent": state.get("current_agent", ""),
                    "iteration": state.get("iteration_count", 0),
                    "html_ready": bool(state.get("html_output")),
                    "deployed": bool(state.get("vercel_url")),
                }
            )
    except Exception as exc:
        logger.warning("[Memory] Could not load thread history: %s", exc)
    return history
