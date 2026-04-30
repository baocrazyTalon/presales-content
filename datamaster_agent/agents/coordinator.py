"""
agents/coordinator.py
─────────────────────
Coordinator agent node — the Project Manager of the multi-agent system.

Responsibilities:
  1. Parse the initial user prompt into an ordered task list.
  2. Validate generated HTML against the Talon.One Best Practices checklist.
  3. Emit routing signals consumed by core/graph.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from core.prompts import COORDINATOR_SYSTEM
from core.state import AgentState, ProspectContext, ValidationResult
from core.store import query_user_insights, record_user_preference

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Best-practices validation checklist
# ─────────────────────────────────────────────────────────────

# Base checks — always required
BASE_CHECKLIST = [
    ("mac_architecture", r"(?:Talon\.?One|our|the)\s.*MAC|MAC\s.*(?:architecture|multi.application)", "MAC architecture reference"),
    ("session_v2", r"Session\s*V2|session.v2", "Integration API Session V2"),
    ("latency_claim", r"sub.?50\s*m?s|<\s*50\s*m?s|under\s+50\s*m?s", "Sub-50ms latency claim"),
    ("unified_rule_engine", r"unified\s+rule\s+engine|single\s+rule\s+engine", "Unified rule engine"),
    ("mermaid_diagram", r"class=\"mermaid\"|mermaid\.init|```mermaid", "Mermaid.js diagram"),
    ("cdp_cep_ecosystem", r"Segment|mParticle|Braze|Iterable|Bloomreach|Tealium", "CDP/CEP ecosystem reference"),
    ("tailwind", r"tailwindcss|tailwind\.css|cdn\.tailwindcss", "Tailwind CSS"),
]


def _validate_html(html: str, prospect: ProspectContext | None = None) -> ValidationResult:
    """Run the checklist against a generated HTML document."""
    import re

    notes: list[str] = []
    passed = True

    for key, pattern, label in BASE_CHECKLIST:
        if re.search(pattern, html, re.IGNORECASE):
            notes.append(f"✅  {key}: {label} found")
        else:
            notes.append(f"❌  {key}: MISSING — {label}")
            passed = False

    # Conditional: competitor comparison required if competitor specified
    if prospect and prospect.get("competitor"):
        comp = prospect["competitor"]
        comp_pattern = rf"(?:vs\.?|versus|compared?\s+to|over|advantage|win)\s.*{re.escape(comp)}|{re.escape(comp)}.*(?:vs\.?|versus|lack|doesn.t)"
        if re.search(comp_pattern, html, re.IGNORECASE):
            notes.append(f"✅  competitor_comparison: {comp} comparison found")
        else:
            notes.append(f"❌  competitor_comparison: MISSING — should compare against {comp}")
            passed = False

    # Conditional: integration mentions if integrations specified
    if prospect and prospect.get("integrations"):
        found_integrations = []
        for integ in prospect["integrations"]:
            if re.search(re.escape(integ), html, re.IGNORECASE):
                found_integrations.append(integ)
        if found_integrations:
            notes.append(f"✅  integrations: {', '.join(found_integrations)} mentioned")
        else:
            notes.append(f"⚠️  integrations: none of {prospect['integrations']} mentioned (non-blocking)")

    return {"passed": passed, "notes": notes}


# ─────────────────────────────────────────────────────────────
# LLM initialisation (lazy — avoids import errors if keys missing)
# ─────────────────────────────────────────────────────────────

def _get_llm() -> Any:
    model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.2)


def _strip_md_json(text: str) -> str:
    """Strip markdown code fences Gemini 2.5 Pro wraps around JSON."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


# ─────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────

def coordinator_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Coordinator.

    Phase A — task decomposition (first call, no html_output yet).
    Phase B — validation (subsequent calls, html_output exists).
    """
    llm = _get_llm()
    iteration = state.get("iteration_count", 0) + 1
    logger.info("[Coordinator] iteration=%d", iteration)

    # Active learning: persist explicit user coaching from state
    user_feedback = state.get("user_feedback")
    if user_feedback:
        user_id = state.get("thread_id", "unknown")
        try:
            record_user_preference(
                user_id=user_id,
                namespace="sales_playbook",
                insight=user_feedback,
                thread_id=state.get("thread_id"),
            )
            logger.info("[Coordinator] recorded user_feedback for %s", user_id)
        except Exception as exc:
            logger.warning("[Coordinator] failed to persist user_feedback: %s", exc)

    html = state.get("html_output")

    # ── Phase B: Validate existing document ───────────────────
    if html:
        prospect = state.get("prospect", {})
        validation = _validate_html(html, prospect=prospect)
        logger.info(
            "[Coordinator] validation passed=%s notes=%s",
            validation["passed"],
            validation["notes"],
        )
        return {
            **state,
            "validation": validation,
            "current_agent": "coordinator",
            "iteration_count": iteration,
        }

    # ── Phase A: Decompose user prompt into tasks ─────────────
    prospect = state.get("prospect", {})
    user_brief = (
        f"Company: {prospect.get('company_name', 'Unknown')}\n"
        f"Industry: {prospect.get('industry', 'Unknown')}\n"
        f"Use-case: {prospect.get('use_case', 'Unknown')}\n"
        f"Competitor: {prospect.get('competitor', 'None specified')}\n"
        f"Integrations: {', '.join(prospect.get('integrations', []))}\n\n"
        f"Raw notes:\n{prospect.get('raw_notes', '')}"
    )

    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM),
        HumanMessage(content=f"Decompose the following presales brief into tasks:\n\n{user_brief}"),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        parsed = json.loads(_strip_md_json(raw))
        tasks: list[str] = parsed.get("tasks", [])
    except json.JSONDecodeError:
        logger.warning("[Coordinator] Could not parse task JSON — using raw text")
        tasks = [raw]

    # Fetch prior coaching for this thread to bootstrap the strategy
    thread_id = state.get("thread_id", "unknown")
    prior_insights = []
    try:
        prior_insights = query_user_insights(
            user_id=thread_id,
            namespace="sales_playbook",
            query=prospect.get("use_case", ""),
            top_k=5,
        )
    except Exception as exc:
        logger.warning("[Coordinator] error loading prior insights: %s", exc)

    return {
        **state,
        "tasks": tasks,
        "completed_tasks": [],
        "personal_insights": prior_insights,
        "current_agent": "coordinator",
        "iteration_count": iteration,
        "messages": [AIMessage(content=f"Tasks decomposed: {tasks}")],
    }
