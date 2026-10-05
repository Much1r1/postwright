import sqlite3
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from postwright.config import get_settings
from postwright.nodes import (
    capture_node,
    critic_node,
    draft_node,
    extract_ideas_node,
    human_review_node,
    pick_angles_node,
    pick_slot_node,
    record_node,
    schedule_node,
)
from postwright.scheduler import SlotPolicy
from postwright.state import PostwrightState
from postwright.store import AngleHistoryStore, ApprovalHistoryStore, QueueStore


def get_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    """Create and set up a SqliteSaver checkpointer instance."""
    if db_path is None:
        db_path = get_settings().postwright_checkpoint_db
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def should_revise_critic(state: PostwrightState | dict[str, Any]) -> str:
    """Determine whether to route back to draft or proceed to pick_slot based on scores."""
    if isinstance(state, dict):
        candidate_drafts = state.get("candidate_drafts", [])
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", get_settings().max_revisions)
        threshold = state.get(
            "critic_score_threshold", get_settings().critic_score_threshold
        )
    else:
        candidate_drafts = state.candidate_drafts
        revision_count = state.revision_count
        max_revisions = state.max_revisions
        threshold = state.critic_score_threshold

    has_failing = any(d.score is None or d.score < threshold for d in candidate_drafts)

    if has_failing and revision_count < max_revisions:
        return "draft"

    return "pick_slot"


def should_continue_human(state: PostwrightState | dict[str, Any]) -> str:
    """Route graph based on human review decision."""
    if isinstance(state, dict):
        human_decision = state.get("human_decision")
        candidate_drafts = state.get("candidate_drafts", [])
    else:
        human_decision = state.human_decision
        candidate_drafts = state.candidate_drafts

    if human_decision in ("approve", "edit"):
        return "schedule"
    elif human_decision == "rewrite":
        return "draft"
    elif human_decision == "reject":
        if candidate_drafts:
            return "human_review"
        return END

    return END


def create_graph(
    llm: BaseChatModel | None = None,
    store: AngleHistoryStore | None = None,
    approval_store: ApprovalHistoryStore | None = None,
    queue_store: QueueStore | None = None,
    slot_policy: SlotPolicy | None = None,
    critic_score_threshold: int | None = None,
    max_revisions: int | None = None,
    max_human_rewrites: int | None = None,
    checkpointer: Any | None = None,
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
    eff_max_human_rewrites = (
        max_human_rewrites
        if max_human_rewrites is not None
        else settings.max_human_rewrites
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
        state.max_human_rewrites = eff_max_human_rewrites
        return draft_node(state, llm=llm)

    def critic_fn(state: PostwrightState) -> dict[str, Any]:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        state.max_human_rewrites = eff_max_human_rewrites
        return critic_node(state, llm=llm)

    def pick_slot_fn(state: PostwrightState) -> dict[str, Any]:
        return pick_slot_node(state, policy=slot_policy, queue_store=queue_store)

    def human_review_fn(state: PostwrightState) -> dict[str, Any]:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        state.max_human_rewrites = eff_max_human_rewrites
        return human_review_node(state, queue_store=queue_store)

    def schedule_fn(state: PostwrightState) -> dict[str, Any]:
        return schedule_node(state, queue_store=queue_store)

    def record_fn(state: PostwrightState) -> dict[str, Any]:
        return record_node(state, store=approval_store)

    def router_critic_fn(state: PostwrightState) -> str:
        state.critic_score_threshold = eff_threshold
        state.max_revisions = eff_max_revisions
        return should_revise_critic(state)

    def router_human_fn(state: PostwrightState) -> str:
        return should_continue_human(state)

    workflow.add_node("capture", capture_fn)
    workflow.add_node("extract_ideas", extract_ideas_fn)
    workflow.add_node("pick_angles", pick_angles_fn)
    workflow.add_node("draft", draft_fn)
    workflow.add_node("critic", critic_fn)
    workflow.add_node("pick_slot", pick_slot_fn)
    workflow.add_node("human_review", human_review_fn)
    workflow.add_node("schedule", schedule_fn)
    workflow.add_node("record", record_fn)

    workflow.add_edge(START, "capture")
    workflow.add_edge("capture", "extract_ideas")
    workflow.add_edge("extract_ideas", "pick_angles")
    workflow.add_edge("pick_angles", "draft")
    workflow.add_edge("draft", "critic")

    workflow.add_conditional_edges(
        "critic",
        router_critic_fn,
        {
            "draft": "draft",
            "pick_slot": "pick_slot",
        },
    )

    workflow.add_edge("pick_slot", "human_review")

    workflow.add_conditional_edges(
        "human_review",
        router_human_fn,
        {
            "schedule": "schedule",
            "draft": "draft",
            "human_review": "human_review",
            END: END,
        },
    )

    workflow.add_edge("schedule", "record")
    workflow.add_edge("record", END)

    if checkpointer is True or checkpointer is None:
        cp = get_checkpointer()
    elif checkpointer is False:
        cp = None
    else:
        cp = checkpointer

    compiled = workflow.compile(checkpointer=cp)

    if cp is not None:
        orig_invoke = compiled.invoke

        def invoke_with_default_thread(input: Any, config: Any = None, **kwargs: Any) -> Any:
            cfg = dict(config or {})
            configurable = dict(cfg.get("configurable") or {})
            if "thread_id" not in configurable:
                tid = None
                if isinstance(input, PostwrightState):
                    tid = input.thread_id
                elif isinstance(input, dict):
                    tid = input.get("thread_id")
                configurable["thread_id"] = tid or f"thread-{uuid.uuid4().hex[:8]}"
                cfg["configurable"] = configurable
            return orig_invoke(input, config=cfg, **kwargs)  # type: ignore[call-overload]

        compiled.invoke = invoke_with_default_thread  # type: ignore[method-assign]

    return compiled
