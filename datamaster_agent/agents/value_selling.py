"""
agents/value_selling.py
────────────────────────
Value Selling Proposition (VSP) agent node — the Strategic Storyteller.

Position in graph: runs AFTER the Data Architect (has retrieved chunks),
BEFORE the Document Engineer (tells it what to write).

Responsibilities:
  1. Analyse prospect context + retrieved knowledge to identify the strongest
     value-selling angles for Talon.One.
  2. Produce a recommended agenda (table of contents) tailored to the prospect.
  3. Surface must-have talking points: integrations, competitive kill-shots,
     customer proof-points, technical differentiators.
  4. Craft a narrative arc (storytelling flow) so the final document reads as
     a persuasive story, not a feature dump.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.prompts import VALUE_SELLING_SYSTEM
from core.state import AgentState
from core.store import record_user_preference, query_user_insights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LLM initialisation
# ─────────────────────────────────────────────────────────────

def _get_llm() -> Any:
    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.3)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _summarise_chunks(state: AgentState, max_chars: int = 6000) -> str:
    """Build a condensed knowledge summary from retrieved RAG chunks."""
    chunks = state.get("retrieved_chunks", [])
    lines: list[str] = []
    total = 0
    for chunk in chunks:
        entry = f"[{chunk['source']}] {chunk['content'][:300]}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def _build_prospect_summary(state: AgentState) -> str:
    """One-paragraph prospect summary for the LLM."""
    p = state.get("prospect", {})
    parts = [
        f"Company: {p.get('company_name', 'Unknown')}",
        f"Industry: {p.get('industry', 'Unknown')}",
        f"Use-case: {p.get('use_case', 'Unknown')}",
        f"Competitor to displace: {p.get('competitor', 'None')}",
        f"Integrations: {', '.join(p.get('integrations', []))}",
    ]
    urls = p.get("urls_to_scrape", [])
    if urls:
        parts.append(f"Web sources scraped: {', '.join(urls)}")
    notes = p.get("raw_notes", "")
    if notes:
        parts.append(f"Notes: {notes[:500]}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────

def value_selling_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Value Selling Proposition.

    Analyses the prospect + retrieved knowledge and outputs:
      - selling_agenda: ordered list of section dicts (title + purpose + talking points)
      - narrative_arc: storytelling guidance for the Document Engineer
      - key_talking_points: must-mention items (integrations, differentiators, proof)
    """
    llm = _get_llm()
    prospect = state.get("prospect", {})

    logger.info(
        "[ValueSelling] Crafting strategy for %s (%s vs %s)",
        prospect.get("company_name"),
        prospect.get("use_case"),
        prospect.get("competitor", "none"),
    )

    prospect_summary = _build_prospect_summary(state)
    knowledge_summary = _summarise_chunks(state)
    tasks = state.get("tasks", [])

    personal_insights = state.get("personal_insights", [])
    if not personal_insights:
        try:
            personal_insights = query_user_insights(
                user_id=state.get("thread_id", "unknown"),
                namespace="sales_playbook",
                query=prospect.get("use_case", ""),
                top_k=5,
            )
        except Exception as exc:
            logger.warning("[ValueSelling] could not query personal insights: %s", exc)
            personal_insights = []

    # Seed RAG query generation from coordinator tasks + value selling heuristics
    rag_queries = [
        f"Talon.One {prospect.get('use_case', 'integration')} {prospect.get('industry', '')}"
    ]
    for t in tasks:
        rag_queries.append(t)

    insight_block = "No personal insights available."
    if personal_insights:
        insight_block = "\n".join(f"- {item.get('content', '')}" for item in personal_insights)

    user_message = f"""Analyse the following prospect and knowledge base, then produce a value-selling
strategy for a Talon.One presales presentation.

## Prospect
{prospect_summary}

## Tasks from Coordinator
{chr(10).join(f'- {t}' for t in tasks)}

## Personal Insights
{insight_block}

## Retrieved Knowledge (condensed)
{knowledge_summary}

## Your deliverables
Return a JSON object with exactly these keys:

1. "selling_agenda" — An ordered list of section objects. Each object has:
   - "section_number": int (1-indexed)
   - "title": string (the section heading for the document)
   - "purpose": string (one sentence explaining WHY this section exists — what buying signal does it address?)
   - "talking_points": list of strings (the specific facts, claims, or proof-points the Document Engineer MUST include)
   - "storytelling_note": string (tone/narrative guidance — e.g. "start with the prospect's pain, then reveal the solution")

2. "narrative_arc" — A string describing the overall storytelling flow in 3-5 sentences.
   Think of it as the "red thread" connecting the sections. Great presales docs follow a
   Problem → Insight → Solution → Proof → Call-to-Action arc.

3. "key_talking_points" — A flat list of the absolute must-mention items across the whole document.
   Group them as objects with:
   - "category": one of "integration", "competitive_advantage", "customer_proof", "technical_differentiator", "business_value"
   - "point": the specific talking point text
   - "priority": "must_have" | "nice_to_have"

4. "competitor_kill_shots" — If a competitor is named, list 2-4 specific, factual
   differentiators where Talon.One wins. Each object: {{"claim": "...", "evidence": "..."}}.
   If no competitor is named, return an empty list.

Important: base every talking point on FACTS from the retrieved knowledge. Never fabricate claims.
If you lack enough data for a section, note it as a gap so the Data Architect can fill it on the next iteration.
"""

    messages = [
        SystemMessage(content=VALUE_SELLING_SYSTEM),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Strip markdown fences
    import re
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[ValueSelling] JSON parse failed — using fallback structure")
        parsed = {
            "selling_agenda": [
                {
                    "section_number": 1,
                    "title": "Executive Summary",
                    "purpose": "Hook the reader with the core value proposition",
                    "talking_points": [f"Talon.One for {prospect.get('use_case', 'promotions')}"],
                    "storytelling_note": "Lead with the prospect's business challenge",
                },
                {
                    "section_number": 2,
                    "title": "Proposed Solution",
                    "purpose": "Show how Talon.One solves the prospect's challenge",
                    "talking_points": ["MAC architecture", "Sub-50ms latency", "Unified rule engine"],
                    "storytelling_note": "Position Talon.One as the inevitable answer",
                },
            ],
            "narrative_arc": "Problem → Insight → Solution → Proof → Next Steps",
            "key_talking_points": [],
            "competitor_kill_shots": [],
        }

    selling_agenda = parsed.get("selling_agenda", [])
    narrative_arc = parsed.get("narrative_arc", "")
    key_talking_points = parsed.get("key_talking_points", [])
    competitor_kill_shots = parsed.get("competitor_kill_shots", [])

    # Build a human-readable agenda summary for the message log
    agenda_text = "\n".join(
        f"  {s.get('section_number', i+1)}. {s.get('title', 'Section')}"
        for i, s in enumerate(selling_agenda)
    )

    logger.info(
        "[ValueSelling] Produced %d-section agenda, %d talking points, %d kill-shots",
        len(selling_agenda),
        len(key_talking_points),
        len(competitor_kill_shots),
    )

    return {
        **state,
        "selling_agenda": selling_agenda,
        "narrative_arc": narrative_arc,
        "key_talking_points": key_talking_points,
        "competitor_kill_shots": competitor_kill_shots,
        "rag_queries": rag_queries,
        "personal_insights": personal_insights,
        "current_agent": "value_selling",
        "messages": [
            AIMessage(
                content=(
                    f"Value Selling strategy ready — {len(selling_agenda)} sections:\n"
                    f"{agenda_text}\n"
                    f"Narrative: {narrative_arc[:200]}"
                )
            )
        ],
    }
