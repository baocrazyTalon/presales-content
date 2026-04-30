"""
agents/document_engineer.py
────────────────────────────
Document Engineer agent node — the Full-Stack Design specialist.

Responsibilities:
  1. Consume retrieved RAG chunks and prospect context.
  2. Generate a self-contained, responsive HTML/Tailwind CSS document.
  3. Embed Mermaid.js integration flow diagrams.
  4. Optionally mirror a competitor/prospect UI style if assets were provided.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.prompts import DOCUMENT_ENGINEER_SYSTEM
from core.state import AgentState

logger = logging.getLogger(__name__)

# Maximum characters of RAG context we inject into the prompt
# Gemini 1.5 Pro has a 1M-token context window — 100K chars is ~25K tokens
_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "100000"))


# ─────────────────────────────────────────────────────────────
# LLM initialisation
# ─────────────────────────────────────────────────────────────

def _get_llm() -> Any:
    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.4)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _build_context_block(state: AgentState) -> str:
    """Condense retrieved chunks into a single context string for the prompt."""
    chunks = state.get("retrieved_chunks", [])
    lines: list[str] = []
    total = 0
    for chunk in chunks:
        entry = f"[{chunk['source']} / {chunk['source_id']}]\n{chunk['content']}\n"
        if total + len(entry) > _MAX_CONTEXT_CHARS:
            break
        lines.append(entry)
        total += len(entry)
    return "\n---\n".join(lines)


def _extract_diagrams(html: str) -> list[str]:
    """Pull out raw Mermaid diagram definitions from the generated HTML."""
    return re.findall(r"class=\"mermaid\">(.*?)</div>", html, re.DOTALL)


def _derive_filename(state: AgentState) -> str:
    company = state.get("prospect", {}).get("company_name", "prospect")
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", company).lower().strip("-")
    return f"{safe}-presales.html"


# ─────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────

def document_engineer_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Document Engineer.

    Uses the LLM to produce a publication-ready HTML/Tailwind presentation
    grounded in the retrieved knowledge chunks.
    """
    llm = _get_llm()
    prospect = state.get("prospect", {})
    validation = state.get("validation")

    logger.info("[DocumentEngineer] Generating HTML for %s", prospect.get("company_name"))

    context_block = _build_context_block(state)

    # If we have prior validation notes, ask the LLM to fix the gaps
    fix_prompt = ""
    if validation and not validation["passed"]:
        missing = [n for n in validation["notes"] if n.startswith("❌")]
        fix_prompt = (
            "\n\n## IMPORTANT — previous iteration failed these checks:\n"
            + "\n".join(missing)
            + "\nPlease fix ALL of the above in this revision."
        )

    # Build agenda guidance from Value Selling agent (if available)
    agenda_block = ""
    selling_agenda = state.get("selling_agenda", [])
    narrative_arc = state.get("narrative_arc", "")
    key_talking_points = state.get("key_talking_points", [])
    competitor_kill_shots = state.get("competitor_kill_shots", [])

    if selling_agenda:
        sections = []
        for s in selling_agenda:
            pts = "\n".join(f"      - {p}" for p in s.get("talking_points", []))
            sections.append(
                f"  {s.get('section_number', '?')}. **{s.get('title', 'Section')}**\n"
                f"     Purpose: {s.get('purpose', '')}\n"
                f"     Storytelling: {s.get('storytelling_note', '')}\n"
                f"     Must-include talking points:\n{pts}"
            )
        agenda_block = (
            "\n\n## VALUE SELLING STRATEGY (follow this structure exactly)\n"
            f"### Narrative Arc\n{narrative_arc}\n\n"
            f"### Document Agenda\n" + "\n\n".join(sections)
        )

    personal_insights = state.get("personal_insights", [])
    insights_block = "No personal selling insights found."
    if personal_insights:
        insights_block = "\n".join(
            f"- {item.get('content', '')}" for item in personal_insights
        )
        if key_talking_points:
            must_haves = [tp["point"] for tp in key_talking_points if tp.get("priority") == "must_have"]
            if must_haves:
                agenda_block += "\n\n### Must-Have Talking Points\n" + "\n".join(f"- {p}" for p in must_haves)
        if competitor_kill_shots:
            kills = [f"- {ks['claim']} (Evidence: {ks['evidence']})" for ks in competitor_kill_shots]
            agenda_block += "\n\n### Competitor Kill-Shots\n" + "\n".join(kills)

    user_message = f"""Generate a complete presales HTML document for the following prospect.

## Prospect
- Company: {prospect.get('company_name', 'Unknown')}
- Industry: {prospect.get('industry', 'Unknown')}
- Use-case: {prospect.get('use_case', 'Unknown')}
- Competitor to displace: {prospect.get('competitor', 'N/A')}
- Key integrations: {', '.join(prospect.get('integrations', []))}

## Document Template
Use template: **{prospect.get('template_type', 'sales-pitch')}**
Follow the section structure for this template EXACTLY as defined in your system prompt.
The fixed nav links and table-of-contents MUST reference the exact section IDs from this template.

## Knowledge Base Context
{context_block}

## Personal Insights
{insights_block}

{agenda_block}
{fix_prompt}

## Output format
Return ONLY a valid JSON object with keys:
  - "html_output": the complete HTML string (no markdown fences)
  - "diagram_definitions": list of raw Mermaid diagram strings
"""

    messages = [
        SystemMessage(content=DOCUMENT_ENGINEER_SYSTEM),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Strip accidental markdown fences the LLM may add
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
        html_output: str = parsed.get("html_output", "")
        diagrams: list[str] = parsed.get("diagram_definitions", [])
    except json.JSONDecodeError:
        logger.warning("[DocumentEngineer] JSON parse failed — treating entire response as HTML")
        html_output = raw
        diagrams = _extract_diagrams(raw)

    filename = _derive_filename(state)

    return {
        **state,
        "html_output": html_output,
        "diagram_definitions": diagrams,
        "output_filename": filename,
        "current_agent": "document_engineer",
        "validation": None,  # reset so coordinator re-validates this fresh output
        "messages": [
            AIMessage(
                content=f"Generated '{filename}' ({len(html_output):,} chars, {len(diagrams)} diagrams)."
            )
        ],
    }
