"""
tools/web_scraper.py
─────────────────────
Web scraping and ingestion tool for the Data Architect agent.

Use cases:
  - Competitor API documentation (e.g. Salesforce Loyalty API, Yotpo docs)
  - Prospect loyalty / promotions / programme pages
  - Any public web page the presales engineer wants ingested as knowledge

Capabilities:
  scrape_url(url)                → clean plain-text extraction of a single page
  scrape_and_ingest(url, …)      → scrape + chunk + embed + store in pgvector
  scrape_sitemap(sitemap_url, …) → read a sitemap.xml and ingest matching pages
  crawl_links(url, …)            → scrape a page + follow internal links up to depth

Anti-abuse:
  - Respects robots.txt via urllib.robotparser
  - Configurable rate-limit delay between requests
  - Default max pages cap (WEB_SCRAPER_MAX_PAGES env, default 20)
  - User-Agent identifies the bot so site owners can block if needed
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = "TalonOnePresalesBot/1.0 (+https://talon.one)"
_REQUEST_TIMEOUT = 20
_DELAY_BETWEEN_REQUESTS = float(os.getenv("WEB_SCRAPER_DELAY", "1.0"))
_MAX_PAGES = int(os.getenv("WEB_SCRAPER_MAX_PAGES", "20"))


# ─────────────────────────────────────────────────────────────
# robots.txt check
# ─────────────────────────────────────────────────────────────

def _is_allowed(url: str) -> bool:
    """Check robots.txt for the target URL. Allow by default on failure."""
    from urllib.robotparser import RobotFileParser

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(_USER_AGENT, url)
    except Exception:
        return True  # permissive on network errors


# ─────────────────────────────────────────────────────────────
# HTML → clean plain text
# ─────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """
    Convert raw HTML to clean readable text.

    Uses BeautifulSoup if available for high-quality extraction,
    falls back to regex stripping otherwise.
    Detects pre-processed text (e.g. converted JSON/YAML) and returns as-is.
    """
    # If this is already plain text (no HTML tags), skip parsing entirely
    if "<html" not in html.lower() and html.count("<") < 10:
        text = re.sub(r"\n{3,}", "\n\n", html)
        return text.strip()
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header — noise for knowledge extraction
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

    except ImportError:
        # Fallback: naive regex stripping
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute href links from HTML, scoped to the same domain."""
    base_domain = urlparse(base_url).netloc
    links: list[str] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if urlparse(href).netloc == base_domain and href not in links:
                links.append(href)
    except ImportError:
        for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = urljoin(base_url, match.group(1))
            if urlparse(href).netloc == base_domain and href not in links:
                links.append(href)
    return links


# ─────────────────────────────────────────────────────────────
# Core fetch
# ─────────────────────────────────────────────────────────────

def _strip_fragment(url: str) -> str:
    """Remove the #fragment portion from a URL — fragments are client-side only."""
    return url.split("#")[0]


def _json_to_text(raw: str, url: str) -> str:
    """
    Convert a JSON or YAML file to flat readable text.
    Handles OpenAPI / Swagger specs by extracting path descriptions,
    summaries, and operation details into a searchable string.
    """
    import json

    data: Any = None

    # Try JSON first
    try:
        data = json.loads(raw)
    except Exception:
        pass

    # Try YAML if JSON failed
    if data is None:
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(raw)
        except Exception:
            return raw[:8000]  # Last resort: return raw text truncated

    if not isinstance(data, dict):
        return str(data)[:8000]

    lines: list[str] = []

    # OpenAPI / Swagger spec
    if "openapi" in data or "swagger" in data:
        info = data.get("info", {})
        lines.append(f"API: {info.get('title', url)}")
        lines.append(f"Version: {info.get('version', '')}")
        if info.get("description"):
            lines.append(info["description"])
        lines.append("")
        for path, methods in (data.get("paths") or {}).items():
            for method, op in (methods or {}).items():
                if not isinstance(op, dict):
                    continue
                summary = op.get("summary", "")
                desc = op.get("description", "")
                tags = ", ".join(op.get("tags", []))
                lines.append(f"{method.upper()} {path} [{tags}] — {summary}")
                if desc:
                    lines.append(f"  {desc[:400]}")
        return "\n".join(lines)

    # Generic JSON — just pretty-print key/value pairs
    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}: " if prefix else f"{k}: ")
        elif isinstance(obj, list):
            for item in obj[:20]:
                _flatten(item, prefix)
        else:
            lines.append(f"{prefix}{obj}")

    _flatten(data)
    return "\n".join(lines[:2000])


def _fetch(url: str) -> str | None:
    """
    Fetch a URL and return text content (HTML, JSON, or YAML), or None on failure.
    Respects robots.txt and rate-limit delay.
    Automatically strips URL fragments (#anchor) before requesting.
    """
    url = _strip_fragment(url)

    if not _is_allowed(url):
        logger.info("[WebScraper] Blocked by robots.txt: %s", url)
        return None

    try:
        resp = httpx.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type or "application/xhtml" in content_type:
            return resp.text
        if "application/json" in content_type or "text/json" in content_type:
            return _json_to_text(resp.text, url)
        if "yaml" in content_type or url.endswith(".yaml") or url.endswith(".yml"):
            return _json_to_text(resp.text, url)
        logger.info("[WebScraper] Skipping unsupported content-type (%s): %s", content_type, url)
        return None
    except Exception as exc:
        logger.warning("[WebScraper] Fetch failed for %s: %s", url, exc)
        return None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def scrape_url(url: str) -> str:
    """
    Scrape a single web page and return clean plain text.

    Args:
        url: Full HTTP(S) URL to scrape.

    Returns:
        Extracted plain text, or empty string on failure.
    """
    html = _fetch(url)
    if not html:
        return ""
    text = _html_to_text(html)
    logger.info("[WebScraper] Scraped %s → %d chars", url, len(text))
    return text


def scrape_and_ingest(
    url: str,
    source: str = "web",
    title: str = "",
    chunk_size: int = 800,
    overlap: int = 80,
) -> int:
    # Normalise URL — remove fragment so storage key is canonical
    url = _strip_fragment(url)
    """
    Scrape a web page and ingest its content into pgvector.

    The text is chunked at ~chunk_size words (with overlap) to match
    the existing Notion ingestion strategy.

    Args:
        url:        Page URL to scrape.
        source:     Source label in the DB (e.g. "competitor_docs", "prospect_site").
        title:      Human-readable title (defaults to the URL).
        chunk_size: Words per chunk (default 800).
        overlap:    Word overlap between chunks (default 80).

    Returns:
        Number of chunks ingested.
    """
    from tools.postgres_rag import ingest_document

    text = scrape_url(url)
    if not text:
        return 0

    words = text.split()
    step = max(chunk_size - overlap, 1)
    chunks_ingested = 0

    for i, start in enumerate(range(0, len(words), step)):
        chunk_text = " ".join(words[start : start + chunk_size])
        if not chunk_text.strip():
            continue
        ingest_document(
            content=chunk_text,
            source=source,
            source_id=f"{url}::chunk_{i}",
            title=title or url,
            metadata={"url": url, "chunk_index": i, "source_type": source},
        )
        chunks_ingested += 1

    logger.info("[WebScraper] Ingested %d chunks from %s", chunks_ingested, url)
    return chunks_ingested


def scrape_sitemap(
    sitemap_url: str,
    url_filter: str = "",
    source: str = "web",
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """
    Parse a sitemap.xml and ingest pages matching the filter.

    Args:
        sitemap_url: URL to the sitemap.xml.
        url_filter:  Regex pattern to filter URLs (e.g. r"/api/" for API docs only).
        source:      Source label for the DB.
        max_pages:   Override maximum pages (default from WEB_SCRAPER_MAX_PAGES).

    Returns:
        List of dicts: [{"url": ..., "chunks_ingested": int}, ...]
    """
    cap = max_pages or _MAX_PAGES
    try:
        resp = httpx.get(
            sitemap_url,
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[WebScraper] Failed to fetch sitemap: %s", exc)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.warning("[WebScraper] Sitemap XML parse error: %s", exc)
        return []

    # Handle namespace (most sitemaps use the sitemap protocol namespace)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    loc_elements = root.findall(".//sm:loc", ns)
    if not loc_elements:
        # Try without namespace
        loc_elements = root.findall(".//loc")

    urls: list[str] = []
    for loc in loc_elements:
        if loc.text:
            page_url = loc.text.strip()
            if url_filter and not re.search(url_filter, page_url):
                continue
            urls.append(page_url)

    report: list[dict[str, Any]] = []
    for page_url in urls[:cap]:
        n = scrape_and_ingest(page_url, source=source, title=page_url)
        report.append({"url": page_url, "chunks_ingested": n})
        time.sleep(_DELAY_BETWEEN_REQUESTS)

    logger.info("[WebScraper] Sitemap: ingested %d pages from %s", len(report), sitemap_url)
    return report


def crawl_links(
    start_url: str,
    depth: int = 1,
    url_filter: str = "",
    source: str = "web",
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """
    Scrape a page and follow internal links up to *depth* levels.

    Useful for competitor docs that don't provide a sitemap. For example:
        crawl_links("https://docs.competitor.com/api/", depth=2,
                     url_filter=r"/api/", source="competitor_docs")

    Args:
        start_url:  Starting page URL.
        depth:      How many link levels to follow (1 = start page only,
                    2 = start + linked pages, etc.).
        url_filter: Regex to restrict which discovered links are followed.
        source:     Source label for the DB.
        max_pages:  Override maximum pages (default from WEB_SCRAPER_MAX_PAGES).

    Returns:
        List of dicts: [{"url": ..., "chunks_ingested": int}, ...]
    """
    cap = max_pages or _MAX_PAGES
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    report: list[dict[str, Any]] = []

    while queue and len(visited) < cap:
        url, current_depth = queue.pop(0)

        # Normalize: strip fragment
        url = url.split("#")[0]
        if url in visited:
            continue
        if url_filter and not re.search(url_filter, url):
            continue

        visited.add(url)

        # Fetch raw HTML for both text extraction and link discovery
        html = _fetch(url)
        if not html:
            continue

        text = _html_to_text(html)
        if text:
            n = _ingest_text(url, text, source)
            report.append({"url": url, "chunks_ingested": n})
        else:
            report.append({"url": url, "chunks_ingested": 0})

        # Discover links for deeper crawl
        if current_depth < depth - 1:
            links = _extract_links(html, url)
            for link in links:
                if link.split("#")[0] not in visited:
                    queue.append((link, current_depth + 1))

        time.sleep(_DELAY_BETWEEN_REQUESTS)

    logger.info("[WebScraper] Crawl from %s: %d pages scraped", start_url, len(report))
    return report


# ─────────────────────────────────────────────────────────────
# Internal helper (avoid circular import at module level)
# ─────────────────────────────────────────────────────────────

def _ingest_text(url: str, text: str, source: str, chunk_size: int = 800, overlap: int = 80) -> int:
    """Chunk and ingest already-extracted text."""
    from tools.postgres_rag import ingest_document

    words = text.split()
    step = max(chunk_size - overlap, 1)
    count = 0
    for i, start in enumerate(range(0, len(words), step)):
        chunk_text = " ".join(words[start : start + chunk_size])
        if not chunk_text.strip():
            continue
        ingest_document(
            content=chunk_text,
            source=source,
            source_id=f"{url}::chunk_{i}",
            title=url,
            metadata={"url": url, "chunk_index": i, "source_type": source},
        )
        count += 1
    return count
