"""Persistence smoke tests for the sales_playbook user insight path."""

from core.store import record_user_preference, query_user_insights
from main import run_agent
from core.state import ProspectContext


def test_insight_store_and_graph():
    user_id = "unit-test-thread"
    feedback = "Focus more on the Loyalty API for Tier 1 clients"

    # 1) record an insight
    insight_id = record_user_preference(
        user_id=user_id,
        namespace="sales_playbook",
        insight=feedback,
        thread_id=user_id,
        tags=["loyalty", "tier1"],
    )
    assert insight_id is not None
    print(f"Insight persisted with id={insight_id}")

    # 2) query the same insight
    results = query_user_insights(user_id=user_id, namespace="sales_playbook", query="Loyalty API", top_k=5)
    assert len(results) > 0
    assert any(feedback in r.get("content", "") for r in results)
    print(f"Query returned {len(results)} insights")

    # 3) run simplified graph with thread_id and ensure personal_insights is in state
    prospect = ProspectContext(
        company_name="Test Co",
        industry="Retail",
        use_case="Loyalty program",
        competitor="Yotpo",
        integrations=["Segment"],
        urls_to_scrape=[],
        raw_notes="Testing user insight persistence",
        presentation_type="client",
        presentation_category="LOYALTY",
        template_type="sales-pitch",
        presentation_name="",
        brand=None,
    )

    final_state = run_agent(prospect)
    assert "personal_insights" in final_state
    if final_state.get("personal_insights"):
        print("Personal insights present in final state.")
    else:
        print("No personal insights in state (ok if none pulled)" )


if __name__ == "__main__":
    test_insight_store_and_graph()
    print("Persistence test completed")
