"""
agents/user_preference.py
────────────────────────
Active learning node / tool for recording user feedback into the persistent store.
"""

from __future__ import annotations

import logging
from core.store import record_user_preference
from core.state import AgentState

logger = logging.getLogger(__name__)


def record_user_preference_node(state: AgentState) -> AgentState:
    """Persist an incoming user feedback text as a long-term sales playbook insight."""
    feedback = state.get("user_feedback")
    if not feedback:
        return state

    user_id = state.get("thread_id", "unknown")
    try:
        record_user_preference(
            user_id=user_id,
            namespace="sales_playbook",
            insight=feedback,
            thread_id=state.get("thread_id"),
        )
        logger.info("[UserPreference] Saved user feedback for %s", user_id)
    except Exception as exc:
        logger.warning("[UserPreference] Could not save user feedback: %s", exc)

    # Clear the feedback after persisting so it is not re-processed every loop
    return {
        **state,
        "user_feedback": None,
    }
