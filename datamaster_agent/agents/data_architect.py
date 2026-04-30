"""
agents/data_architect.py
─────────────────────────
Data Architect agent node — the Knowledge Base / RAG specialist.

Responsibilities:
  1. Derive semantic queries from the task list.
  2. Search the pgvector store for relevant Talon.One documentation chunks.
  3. Pull additional assets from Google Drive (PPTX, XLSX, PDF, Images).
  4. Use Vision LLM to describe any image/slide assets.
  5. Return all retrieved chunks to the shared state.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.prompts import DATA_ARCHITECT_SYSTEM
from core.state import AgentState, RAGChunk
from tools.google_drive import list_drive_files, download_and_extract_text
from tools.notion import search_notion, ingest_notion_page
from tools.postgres_rag import similarity_search, multi_query_search, ingest_document, is_url_ingested
from tools.web_scraper import scrape_url, scrape_and_ingest

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LLM initialisation
# ─────────────────────────────────────────────────────────────

def _get_llm() -> Any:
    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.0)


def _strip_md_json(text: str) -> str:
    """Strip markdown code fences Gemini 2.5 Pro wraps around JSON."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


# ─────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────

def data_architect_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Data Architect.

    Generates RAG queries from the task list and retrieves the most
    relevant knowledge chunks from pgvector + Google Drive.
    """
    llm = _get_llm()
    logger.info("[DataArchitect] Starting retrieval")

    tasks = state.get("tasks", [])
    prospect = state.get("prospect", {})
    top_k = int(os.getenv("RAG_TOP_K", "8"))

    # Start with any RAG queries planned by Value Selling
    queries = state.get("rag_queries", []) or []

    if not queries:
        # ── Step 1: Generate multiple query variants via LLM ─────
        task_summary = "\n".join(f"- {t}" for t in tasks)
        messages = [
            SystemMessage(content=DATA_ARCHITECT_SYSTEM),
            HumanMessage(
                content=(
                    f"Generate semantic search queries for these tasks:\n{task_summary}\n\n"
                    f"Prospect context: {prospect.get('company_name')} — "
                    f"{prospect.get('industry')} — {prospect.get('use_case')}\n"
                    f"Competitor: {prospect.get('competitor', 'none')}\n"
                    f"Integrations: {', '.join(prospect.get('integrations', []))}\n\n"
                    "IMPORTANT: Generate 3-5 diverse query variants per task. "
                    "Rephrase each from different angles (technical, business, competitive) "
                    "to maximise recall across different document styles."
                )
            ),
        ]

        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        try:
            parsed = json.loads(_strip_md_json(raw))
            queries = parsed.get("rag_queries", [])
        except json.JSONDecodeError:
            logger.warning("[DataArchitect] Could not parse query JSON — falling back")
            queries = [f"Talon.One {prospect.get('use_case', 'integration')} documentation"]

    # ── Step 2: Multi-query vector search with RRF reranking ──
    all_chunks: list[RAGChunk] = []
    seen_ids: set[str] = set()

    if queries:
        fused_chunks = multi_query_search(queries, top_k=top_k)
        for chunk in fused_chunks:
            uid = f"{chunk['source_id']}:{chunk['content'][:60]}"
            if uid not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(uid)

    # Include persisted user coaching insights from sales_playbook as top-priority context
    user_insights = state.get("personal_insights", [])
    for insight in user_insights:
        insight_uid = f"sales_playbook:{insight.get('id', '')}:{insight.get('content', '')[:60]}"
        if insight_uid not in seen_ids:
            all_chunks.insert(0, RAGChunk(
                content=insight.get("content", ""),
                source="sales_playbook",
                source_id=str(insight.get("id", "")),
                score=1.0,
            ))
            seen_ids.add(insight_uid)

    # ── Step 3: Google Drive assets ───────────────────────────
    drive_root = os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")
    if drive_root:
        try:
            drive_files = list_drive_files(folder_id=drive_root, query=prospect.get("company_name", ""))
            for f in drive_files[:5]:  # cap at 5 Drive assets per run
                text = download_and_extract_text(f["id"], f["mimeType"])
                if text:
                    all_chunks.append(
                        RAGChunk(
                            content=text[:2000],  # truncate very large docs
                            source="google_drive",
                            source_id=f["id"],
                            score=0.0,
                        )
                    )
        except Exception as exc:
            logger.warning("[DataArchitect] Google Drive error: %s", exc)

    # ── Step 4: Notion pages ──────────────────────────────────
    notion_key = os.getenv("NOTION_API_KEY", "")
    if notion_key:
        try:
            notion_query = prospect.get("use_case") or prospect.get("company_name") or "Talon.One"
            notion_pages = search_notion(notion_query, max_pages=int(os.getenv("NOTION_MAX_PAGES", "5")))
            for npage in notion_pages:
                # Ingest into pgvector so future runs benefit from cached embeddings,
                # then also surface the raw text as an in-session RAG chunk.
                try:
                    from tools.notion import extract_page_text
                    page_text = extract_page_text(npage["id"])
                    if page_text.strip():
                        chunk_preview = page_text[:2000]
                        uid = f"notion:{npage['id']}"
                        if uid not in seen_ids:
                            all_chunks.append(
                                RAGChunk(
                                    content=chunk_preview,
                                    source="notion",
                                    source_id=npage["id"],
                                    score=0.0,
                                )
                            )
                            seen_ids.add(uid)
                        # Background-persist (best-effort — does not block response)
                        ingest_notion_page(npage["id"], title=npage["title"])
                except Exception as nexc:
                    logger.warning("[DataArchitect] Notion page %s error: %s", npage["id"], nexc)
        except Exception as exc:
            logger.warning("[DataArchitect] Notion search error: %s", exc)

    # ── Step 5: Web scraping (skip if already ingested) ─────────
    urls_to_scrape = prospect.get("urls_to_scrape", [])
    if urls_to_scrape:
        for page_url in urls_to_scrape:
            try:
                uid = f"web:{page_url}"
                if uid in seen_ids:
                    continue
                # Skip re-scraping on validation loops
                if is_url_ingested(page_url):
                    logger.info("[DataArchitect] URL already ingested, skipping: %s", page_url)
                    # Still add a preview chunk for the current session
                    cached = similarity_search(page_url, top_k=1)
                    if cached and uid not in seen_ids:
                        all_chunks.append(cached[0])
                        seen_ids.add(uid)
                    continue
                text = scrape_url(page_url)
                if text.strip():
                    all_chunks.append(
                        RAGChunk(
                            content=text[:2000],
                            source="web",
                            source_id=page_url,
                            score=0.0,
                        )
                    )
                    seen_ids.add(uid)
                    # Persist full content to pgvector for future runs
                    scrape_and_ingest(
                        page_url,
                        source="web",
                        title=page_url,
                    )
            except Exception as exc:
                logger.warning("[DataArchitect] Web scrape error for %s: %s", page_url, exc)

    logger.info("[DataArchitect] Retrieved %d unique chunks", len(all_chunks))

    return {
        **state,
        "rag_queries": queries,
        "retrieved_chunks": all_chunks,
        "current_agent": "data_architect",
        "messages": [
            AIMessage(content=f"Retrieved {len(all_chunks)} chunks across {len(queries)} queries.")
        ],
    }
