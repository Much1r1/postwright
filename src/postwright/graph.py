from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from postwright.config import get_settings
from postwright.nodes import (
    capture_node,
    critic_node,
    draft_node,
    extract_ideas_node,
    pick_angles_node,
)
from postwright.state import PostwrightState
from postwright.store import AngleHistoryStore


def should_revise(state: PostwrightState | dict[str, Any]) -> str:
    """Determine whether to route back to draft or end based on scores and revision count."""
    if isinstance(state, dict):
        candidate_drafts = state.get("candidate_drafts", [])
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", get_settings().max_revisions)
        threshold = state.get("critic_score_threshold", get_settings().critic_score_threshold)
    else:
        candidate_drafts = state.candidate_drafts
        revision_count = state.revision_count
        max_revisions = state.max_revisions
        threshold = state.critic_score_threshold

    has_failing = any(d.score is None or d.score < threshold for d in candidate_drafts)

    if has_failing and revision_count < max_revisions:
        return "draft"

    return END


def create_graph(
    llm: BaseChatModel | None = None,
    store: AngleHistoryStore | None = None,
    critic_score_threshold: int | None = None,
    max_revisions: int | None = None,
) -> Any:
    """Build and return the compiled LangGraph workflow graph."""
    settings = get_settings()
    eff_threshold = (
        critic_score_threshold
        if critic_score_threshold is not None
        else settings.critic_score_threshold
    )
    eff_max_revisions = (
        max_revisions if max_revisions is not None else settings.max_revisions
    )

    workflow = StateGraph(PostwrightState)

    def capture_fn(state: PostwrightState) -> dict[str, Any]:
        return capture_node(state)

    def extract_ideas_fn(state: PostwrightState) -> dict[str, Any]:
        return extract_ideas_node(state, llm=llm)

    def pick_angles_fn(state: PostwrightState) -> dict[str, Any]:
        return pick_angles_node(state, store=store)

    def draft_fn(state: PostwrightState) -> dict[str, Any]:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        return draft_node(state, llm=llm)

    def critic_fn(state: PostwrightState) -> dict[str, Any]:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        return critic_node(state, llm=llm)

    def router_fn(state: PostwrightState) -> str:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        return should_revise(state)

    workflow.add_node("capture", capture_fn)
    workflow.add_node("extract_ideas", extract_ideas_fn)
    workflow.add_node("pick_angles", pick_angles_fn)
    workflow.add_node("draft", draft_fn)
    workflow.add_node("critic", critic_fn)

    workflow.add_edge(START, "capture")
    workflow.add_edge("capture", "extract_ideas")
    workflow.add_edge("extract_ideas", "pick_angles")
    workflow.add_edge("pick_angles", "draft")
    workflow.add_edge("draft", "critic")

    workflow.add_conditional_edges(
        "critic",
        router_fn,
        {
            "draft": "draft",
            END: END,
        },
    )

    return workflow.compile()
