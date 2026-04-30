"""
core/graph.py
─────────────
LangGraph StateGraph definition: wires the three agent nodes together
and defines conditional routing logic.

Graph topology:
  START
    │
    ▼
  coordinator ──────────────────────────────────────────────────────► deploy ──► END
    │   ▲                                                                 ▲
    │   │ (validation failed – request more data)                         │
    ▼   │                                                                 │
  data_architect                                                          │
    │                                                                     │
    ▼                                                                     │
  document_engineer ───────────────────────────────────────────────────►─┘
                       (coordinator validates and approves)
"""

from __future__ import annotations

import os
from langgraph.graph import StateGraph, END, START

from core.state import AgentState
from agents.coordinator import coordinator_node
from agents.data_architect import data_architect_node
from agents.value_selling import value_selling_node
from agents.document_engineer import document_engineer_node
from agents.user_preference import record_user_preference_node
from tools.deployment import deploy_node
from core.memory import get_checkpointer
from core.store import get_store


# ─────────────────────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────────────────────

def route_after_coordinator(state: AgentState) -> str:
    """
    Called after the coordinator node runs.
    New flow:
      - Initial or validation loop: go to value_selling first.
      - If validation passes: deploy.
      - If the document fails validation: loop back to value_selling after fresh retrieval.
    """
    max_iter = int(os.getenv("MAX_ITERATIONS", "10"))
    iteration = state.get("iteration_count", 0)

    if state.get("error"):
        return END  # type: ignore[return-value]

    if iteration >= max_iter:
        import logging
        logging.getLogger(__name__).warning(
            "[Graph] Max iterations (%d) reached — force-deploying best-effort document",
            max_iter,
        )
        if state.get("html_output"):
            return "deploy"
        return END  # type: ignore[return-value]

    if state.get("user_feedback"):
        return "user_preference"
    if not state.get("html_output"):
        return "value_selling"

    validation = state.get("validation")
    if validation and validation.get("passed"):
        return "deploy"

    if iteration >= 3 and state.get("html_output"):
        import logging
        logging.getLogger(__name__).warning(
            "[Graph] 3+ validation failures — force-deploying best-effort document"
        )
        return "deploy"

    # Re-evaluate under value selling guidance after failed validation
    return "value_selling"


def route_after_document_engineer(state: AgentState) -> str:
    """After document generation, always return to coordinator for validation."""
    return "coordinator"


# ─────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("user_preference", record_user_preference_node)
    workflow.add_node("value_selling", value_selling_node)
    workflow.add_node("data_architect", data_architect_node)
    workflow.add_node("document_engineer", document_engineer_node)
    workflow.add_node("deploy", deploy_node)

    # Entry point
    workflow.add_edge(START, "coordinator")

    # Coordinator routes conditionally
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "user_preference": "user_preference",
            "value_selling": "value_selling",
            "deploy": "deploy",
            END: END,
        },
    )

    # Value selling -> Data architect -> Document engineer
    workflow.add_edge("value_selling", "data_architect")
    workflow.add_edge("data_architect", "document_engineer")

    # Feedback node path: user preference saved then strategy continues
    workflow.add_edge("user_preference", "value_selling")

    # Document engineer loops back to coordinator for validation
    workflow.add_conditional_edges(
        "document_engineer",
        route_after_document_engineer,
        {"coordinator": "coordinator"},
    )

    # Deploy is terminal
    workflow.add_edge("deploy", END)

    return workflow


def route_after_value_selling(state: AgentState) -> str:
    """After value selling strategy, go to data architect for retrieval."""
    return "data_architect"


def route_after_data_architect(state: AgentState) -> str:
    """After data architect has fetched knowledge, send to document engineer."""
    return "document_engineer"


def route_after_document_engineer(state: AgentState) -> str:
    """After document generation, validate with coordinator."""
    return "coordinator"
    # Document engineer always loops back to coordinator for validation
    workflow.add_conditional_edges(
        "document_engineer",
        route_after_document_engineer,
        {"coordinator": "coordinator"},
    )

    # Deploy is a terminal node
    workflow.add_edge("deploy", END)

    return workflow


# ─────────────────────────────────────────────────────────────
# Lazy compilation — deferred so DB failures don't crash import
# ─────────────────────────────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    """
    Return the compiled LangGraph, building it on first call.

    Attaches a PostgresSaver checkpointer so every node transition is
    persisted to PostgreSQL.  Falls back to no checkpointer (in-memory
    only) when the DB is not yet available.
    """
    global _compiled_graph
    if _compiled_graph is None:
        checkpointer = get_checkpointer()
        store = get_store()
        _compiled_graph = build_graph().compile(checkpointer=checkpointer)
        # Attach store object to the compiled graph for node access if needed
        try:
            setattr(_compiled_graph, 'store', store)
        except Exception:
            pass
    return _compiled_graph


# Legacy alias — kept for backwards compatibility
@property
def compiled_graph():
    return get_compiled_graph()
