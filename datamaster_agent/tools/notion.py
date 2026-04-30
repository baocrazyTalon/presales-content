"""
tools/notion.py
────────────────
Notion integration for the Data Architect agent.

Capabilities:
  - search_notion(query)        → list matching pages/databases from Notion workspace
  - extract_page_text(page_id)  → recursively traverse blocks → plain text
  - ingest_notion_page(page_id) → extract + embed + store in pgvector
  - ingest_notion_search(query, page_ids=None)
                                 → search workspace + selectively ingest chosen pages

Authentication:
  Set NOTION_API_KEY in .env — get an Internal Integration token at:
  https://www.notion.so/my-integrations
  Then share the relevant page/database with your integration.

Optional:
  NOTION_ROOT_PAGE_ID — if set, search is scoped to descendants of that page.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Client initialisation
# ─────────────────────────────────────────────────────────────

def _get_client() -> Any:
    """Return an authenticated Notion client."""
    from notion_client import Client

    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NOTION_API_KEY is not set. "
            "Create an internal integration at https://www.notion.so/my-integrations "
            "and add its token to .env."
        )
    return Client(auth=api_key)


# ─────────────────────────────────────────────────────────────
# Block → plain-text extraction
# ─────────────────────────────────────────────────────────────

_RICH_TEXT_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "toggle",
    "quote", "callout", "code",
}

_SPECIAL_TYPES = {
    "table_row",      # cells contain arrays of rich_text
    "to_do",          # has a text field + checked bool
}


def _rich_text_to_str(rich_text_array: list[dict]) -> str:
    """Concatenate plain_text fields from a rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in rich_text_array)


def _block_to_text(block: dict) -> str:
    """Convert a single Notion block dict to a plain-text string."""
    btype = block.get("type", "")
    bdata = block.get(btype, {})

    if btype in _RICH_TEXT_TYPES:
        return _rich_text_to_str(bdata.get("rich_text", []))

    if btype == "to_do":
        checked = "✓" if bdata.get("checked") else "☐"
        text = _rich_text_to_str(bdata.get("rich_text", []))
        return f"{checked} {text}"

    if btype == "table_row":
        cells = bdata.get("cells", [])
        return " | ".join(_rich_text_to_str(cell) for cell in cells)

    if btype == "divider":
        return "---"

    if btype == "equation":
        return bdata.get("expression", "")

    # image / file / pdf / video / embed — return caption or URL
    if btype in ("image", "video", "file", "pdf", "embed", "bookmark"):
        caption = _rich_text_to_str(bdata.get("caption", []))
        url = (
            bdata.get("external", {}).get("url")
            or bdata.get("file", {}).get("url")
            or bdata.get("url", "")
        )
        return caption or url or ""

    return ""


def _fetch_all_blocks(client: Any, block_id: str, depth: int = 0) -> list[str]:
    """
    Recursively retrieve all blocks under block_id and return a list of text lines.
    Depth is capped at 3 to avoid runaway recursion on deeply nested pages.
    """
    if depth > 3:
        return []

    lines: list[str] = []
    cursor = None

    while True:
        kwargs: dict[str, Any] = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = client.blocks.children.list(**kwargs)
        blocks = resp.get("results", [])

        for block in blocks:
            text = _block_to_text(block)
            if text.strip():
                lines.append(text)
            # Recurse into children (toggles, synced blocks, columns, etc.)
            if block.get("has_children"):
                child_lines = _fetch_all_blocks(client, block["id"], depth + 1)
                lines.extend(child_lines)

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return lines


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def search_notion(query: str, max_pages: int = 20) -> list[dict[str, str]]:
    """
    Search the Notion workspace for pages matching *query*.

    Args:
        query:     Free-text search string.
        max_pages: Maximum number of results to return (default 20).

    Returns:
        List of dicts with keys: id, title, url, last_edited_time.
    """
    client = _get_client()
    results: list[dict[str, str]] = []
    cursor = None

    while len(results) < max_pages:
        kwargs: dict[str, Any] = {
            "query": query,
            "filter": {"value": "page", "property": "object"},
            "page_size": min(max_pages - len(results), 100),
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = client.search(**kwargs)
        pages = resp.get("results", [])

        for page in pages:
            # Extract title from title property (varies by page type)
            title = ""
            props = page.get("properties", {})
            for prop in props.values():
                if prop.get("type") == "title":
                    title = _rich_text_to_str(prop.get("title", []))
                    break
            if not title:
                # Fallback to page.title path (database-less pages)
                title_prop = page.get("properties", {}).get("title", {})
                title = _rich_text_to_str(title_prop.get("title", []))

            results.append(
                {
                    "id": page["id"],
                    "title": title or "(Untitled)",
                    "url": page.get("url", ""),
                    "last_edited_time": page.get("last_edited_time", ""),
                }
            )

        if not resp.get("has_more") or not pages:
            break
        cursor = resp.get("next_cursor")

    logger.info("[Notion] search('%s') → %d pages", query, len(results))
    return results[:max_pages]


def extract_page_text(page_id: str) -> str:
    """
    Retrieve the full plain-text content of a Notion page.

    Args:
        page_id: Notion page UUID (with or without dashes).

    Returns:
        Multi-line string of the page's text content.
    """
    client = _get_client()
    lines = _fetch_all_blocks(client, page_id)
    return "\n".join(lines)


def ingest_notion_page(page_id: str, title: str = "") -> int:
    """
    Extract a Notion page and store it in pgvector.

    Chunks the page into segments of up to 800 words each to respect
    embedding token limits and improve retrieval granularity.

    Args:
        page_id: Notion page UUID.
        title:   Optional human-readable title (used as metadata).

    Returns:
        Number of chunks ingested.
    """
    from tools.postgres_rag import ingest_document

    text = extract_page_text(page_id)
    if not text.strip():
        logger.warning("[Notion] Page %s yielded no text — skipping", page_id)
        return 0

    # Chunk into ~800-word segments with 80-word overlap
    words = text.split()
    chunk_size = 800
    overlap = 80
    step = chunk_size - overlap

    chunks_ingested = 0
    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        chunk_text = " ".join(chunk_words)
        if not chunk_text.strip():
            continue

        ingest_document(
            content=chunk_text,
            source="notion",
            source_id=f"{page_id}::chunk_{i}",
            title=title or page_id,
            metadata={
                "notion_page_id": page_id,
                "chunk_index": i,
                "title": title,
            },
        )
        chunks_ingested += 1

    logger.info("[Notion] Ingested %d chunks from page %s", chunks_ingested, page_id)
    return chunks_ingested


def ingest_notion_search(
    query: str,
    page_ids: list[str] | None = None,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Notion for *query*, optionally restrict to *page_ids*, and ingest.

    Args:
        query:     Search keyword (e.g. "loyalty programme integration").
        page_ids:  If provided, only ingest pages whose IDs are in this list.
                   This lets the user pre-select which pages to store.
        max_pages: Max pages to fetch from search before filtering (default 10).

    Returns:
        List of dicts — one per ingested page — with keys:
          id, title, url, chunks_ingested.
    """
    pages = search_notion(query, max_pages=max_pages)

    if page_ids:
        # Normalise IDs (strip dashes) for comparison
        wanted = {pid.replace("-", "") for pid in page_ids}
        pages = [p for p in pages if p["id"].replace("-", "") in wanted]

    report: list[dict[str, Any]] = []
    for page in pages:
        try:
            n = ingest_notion_page(page["id"], title=page["title"])
            report.append({**page, "chunks_ingested": n})
        except Exception as exc:
            logger.warning("[Notion] Failed to ingest %s: %s", page["id"], exc)
            report.append({**page, "chunks_ingested": 0, "error": str(exc)})

    logger.info(
        "[Notion] ingest_notion_search('%s') → %d pages ingested",
        query,
        len(report),
    )
    return report
