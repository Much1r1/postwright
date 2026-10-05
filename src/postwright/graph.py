from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from postwright.nodes import (
    capture_node,
    draft_node,
    extract_ideas_node,
    pick_angles_node,
)
from postwright.state import PostwrightState
from postwright.store import AngleHistoryStore


def create_graph(
    llm: BaseChatModel | None = None,
    store: AngleHistoryStore | None = None,
) -> Any:
    """Build and return the compiled LangGraph workflow graph."""
    workflow = StateGraph(PostwrightState)

    # Wrap nodes to inject optional dependencies if supplied
    def capture_fn(state: PostwrightState) -> dict[str, Any]:
        return capture_node(state)

    def extract_ideas_fn(state: PostwrightState) -> dict[str, Any]:
        return extract_ideas_node(state, llm=llm)

    def pick_angles_fn(state: PostwrightState) -> dict[str, Any]:
        return pick_angles_node(state, store=store)

    def draft_fn(state: PostwrightState) -> dict[str, Any]:
        return draft_node(state, llm=llm)

    workflow.add_node("capture", capture_fn)
    workflow.add_node("extract_ideas", extract_ideas_fn)
    workflow.add_node("pick_angles", pick_angles_fn)
    workflow.add_node("draft", draft_fn)

    workflow.add_edge(START, "capture")
    workflow.add_edge("capture", "extract_ideas")
    workflow.add_edge("extract_ideas", "pick_angles")
    workflow.add_edge("pick_angles", "draft")
    workflow.add_edge("draft", END)

    return workflow.compile()
