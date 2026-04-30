"""
core/state.py
─────────────
LangGraph shared state (TypedDict) that flows through every node.
All agents read from and write to this single object — LangGraph
merges list fields automatically via the Annotated[..., add_messages]
reducer where indicated.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────
# Sub-schemas (plain TypedDicts — not nodes, just structured data)
# ─────────────────────────────────────────────────────────────

class ProspectContext(TypedDict, total=False):
    """Captured from the initial user prompt."""
    company_name: str             # e.g. "Dali"
    industry: str                 # e.g. "Retail", "QSR", "iGaming"
    use_case: str                 # e.g. "Loyalty POS integration"
    competitor: Optional[str]     # e.g. "Salesforce Loyalty", "Yotpo"
    raw_notes: str                # unstructured intake notes / pasted text
    integrations: list[str]       # e.g. ["Braze", "Segment", "Talon.One API"]
    urls_to_scrape: list[str]     # web pages to scrape (competitor docs, prospect pages)
    presentation_type: str        # "client" | "competition" | "partner"
    presentation_category: str    # e.g. "LOYALTY", "CEP", "CDP" (used for competition/partner subfolders)
    doc_status: str               # "draft" | "published"
    template_type: str            # "sales-pitch" | "technical-integration" | "business-case" | "battle-card" | "poc-scope" | "partner-gtm"


class RAGChunk(TypedDict):
    """A single retrieved document chunk."""
    content: str
    source: str        # origin: "talon_docs" | "google_drive" | "notion" | "local"
    source_id: str     # Drive file ID, doc slug, Notion page ID, or local path
    score: float       # cosine similarity score


class SellingAgendaSection(TypedDict, total=False):
    """One section in the VSP agent's recommended agenda."""
    section_number: int
    title: str
    purpose: str
    talking_points: list[str]
    storytelling_note: str


class TalkingPoint(TypedDict, total=False):
    """A categorised talking point from the VSP agent."""
    category: str    # integration | competitive_advantage | customer_proof | technical_differentiator | business_value
    point: str
    priority: str    # must_have | nice_to_have


class CompetitorKillShot(TypedDict, total=False):
    """A competitive differentiator where Talon.One wins."""
    claim: str
    evidence: str


class ValidationResult(TypedDict):
    passed: bool
    notes: list[str]   # human-readable pass/fail items


# ─────────────────────────────────────────────────────────────
# Master AgentState — the single source of truth for the graph
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Conversation history (auto-merged by LangGraph) ───────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Prospect intake ───────────────────────────────────────
    prospect: ProspectContext

    # ── Coordinator task tracking ─────────────────────────────
    tasks: list[str]            # ordered list of sub-tasks
    completed_tasks: list[str]  # tasks finished so far

    # ── RAG / knowledge retrieval ─────────────────────────────
    rag_queries: list[str]          # queries issued to the vector store
    retrieved_chunks: list[RAGChunk]  # all chunks from data_architect

    # ── Value Selling Proposition outputs ─────────────────────
    selling_agenda: list[SellingAgendaSection]  # ordered sections from VSP agent
    narrative_arc: Optional[str]                # storytelling guidance
    key_talking_points: list[TalkingPoint]      # must-mention items
    competitor_kill_shots: list[CompetitorKillShot]  # {claim, evidence} pairs

    # ── Document engineer outputs ─────────────────────────────
    html_output: Optional[str]           # final rendered HTML string
    diagram_definitions: list[str]       # raw Mermaid.js diagram blocks

    # ── Coordinator validation ────────────────────────────────
    validation: Optional[ValidationResult]

    # ── Deployment ────────────────────────────────────────────
    output_filename: Optional[str]   # e.g. "dali-presales-2026-03.html"
    github_pr_url: Optional[str]
    vercel_url: Optional[str]

    # ── Control flow metadata ─────────────────────────────────
    current_agent: str          # name of the node currently executing
    iteration_count: int        # guard against infinite loops
    error: Optional[str]        # set by any node on fatal failure

    # ── Memory / checkpointing ────────────────────────────────
    # thread_id is stored in the LangGraph config, not the state, but
    # we mirror it here so every node can log/reference it easily.
    thread_id: Optional[str]    # e.g. "my-muscle-chef" — one per prospect
    # persisted Active Learning content
    personal_insights: list[dict[str, Any]]
    user_feedback: str | None
    # ── Arbitrary carry-through data ──────────────────────────
    extra: dict[str, Any]       # escape hatch for future fields
