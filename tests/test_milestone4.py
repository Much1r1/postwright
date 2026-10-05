from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from pydantic import Field
from typer.testing import CliRunner

from postwright.cli import app
from postwright.graph import create_graph, get_checkpointer
from postwright.nodes import (
    CandidateDraftsOutput,
    CriticOutput,
    ExtractedIdeasOutput,
    SingleDraftCritiqueOutput,
    record_node,
)
from postwright.state import CandidateDraft, ExtractedIdea, InputNote, PostwrightState
from postwright.store import ApprovalHistoryStore


class FakeStructuredLLM(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)

    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        super().__init__(responses=list(responses), **kwargs)

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> Any:
        return None

    @property
    def _llm_type(self) -> str:
        return "fake_structured_llm"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        def _invoke(messages: Any) -> Any:
            if self.responses:
                return self.responses.pop(0)
            return None

        mock = MagicMock()
        mock.invoke.side_effect = _invoke
        return mock


def build_mock_llm() -> FakeStructuredLLM:
    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="Implemented LangGraph Human Review")]
    )
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Built Milestone 4 human review interrupt mechanism.",
            )
        ]
    )
    fake_critique = CriticOutput(
        critiques=[
            SingleDraftCritiqueOutput(
                draft_id="draft-1",
                voice_match_score=4,
                clarity_score=4,
                specificity_score=4,
                hook_strength_score=4,
                critique="Great post.",
            )
        ]
    )
    return FakeStructuredLLM(responses=[fake_ideas, fake_drafts, fake_critique])


def test_graph_pauses_at_human_review_without_proceeding(tmp_path: Any) -> None:
    cp_file = tmp_path / "checkpoints.db"
    store_file = tmp_path / "store.db"

    checkpointer = get_checkpointer(cp_file)
    approval_store = ApprovalHistoryStore(store_file)
    llm = build_mock_llm()

    graph = create_graph(llm=llm, checkpointer=checkpointer, approval_store=approval_store)
    config = {"configurable": {"thread_id": "test-pause-thread"}}

    initial_state = PostwrightState(
        thread_id="test-pause-thread",
        raw_note=InputNote(content="Test human review interrupt."),
    )

    graph.invoke(initial_state, config=config)

    # Verify execution stopped at interrupt
    st = graph.get_state(config)
    assert st.next == ("human_review",)
    assert len(st.tasks) > 0
    assert len(st.tasks[0].interrupts) > 0

    # Ensure no approval was recorded yet
    assert len(approval_store.get_history()) == 0


def test_resume_from_new_graph_instance_simulating_closed_terminal(tmp_path: Any) -> None:
    cp_file = tmp_path / "checkpoints.db"
    store_file = tmp_path / "store.db"

    llm1 = build_mock_llm()
    checkpointer1 = get_checkpointer(cp_file)
    approval_store1 = ApprovalHistoryStore(store_file)

    graph1 = create_graph(llm=llm1, checkpointer=checkpointer1, approval_store=approval_store1)
    config = {"configurable": {"thread_id": "thread-closed-terminal"}}

    initial_state = PostwrightState(
        thread_id="thread-closed-terminal",
        raw_note=InputNote(content="Testing resume from fresh process."),
    )

    graph1.invoke(initial_state, config=config)

    # Simulate closing terminal / process by instantiating a completely NEW graph object
    checkpointer2 = get_checkpointer(cp_file)
    approval_store2 = ApprovalHistoryStore(store_file)
    graph2 = create_graph(checkpointer=checkpointer2, approval_store=approval_store2)

    # Resume from NEW graph instance
    graph2.invoke(
        Command(resume={"decision": "approve", "draft_id": "draft-1"}), config=config
    )

    st2 = graph2.get_state(config)
    assert not st2.next  # Completed
    assert len(approval_store2.get_history()) == 1
    rec = approval_store2.get_history()[0]
    assert rec.thread_id == "thread-closed-terminal"
    assert rec.draft_id == "draft-1"


def test_approve_edit_reject_and_rewrite_decisions(tmp_path: Any) -> None:
    cp_file = tmp_path / "checkpoints.db"
    store_file = tmp_path / "store.db"

    # 1. Test Edit
    llm = build_mock_llm()
    checkpointer = get_checkpointer(cp_file)
    approval_store = ApprovalHistoryStore(store_file)
    graph = create_graph(llm=llm, checkpointer=checkpointer, approval_store=approval_store)
    config_edit = {"configurable": {"thread_id": "t-edit"}}

    graph.invoke(PostwrightState(thread_id="t-edit", raw_note=InputNote(content="Test edit.")), config=config_edit)

    edited_content = "Built Milestone 4 human review interrupt mechanism with explicit diff."
    graph.invoke(
        Command(resume={"decision": "edit", "draft_id": "draft-1", "edit_text": edited_content}),
        config=config_edit,
    )

    records = approval_store.get_history()
    assert len(records) == 1
    assert records[0].final_text == edited_content
    assert "-Built Milestone 4 human review interrupt mechanism." in records[0].unified_diff
    assert "+Built Milestone 4 human review interrupt mechanism with explicit diff." in records[0].unified_diff

    # 2. Test Reject
    config_rej = {"configurable": {"thread_id": "t-reject"}}
    graph.invoke(PostwrightState(thread_id="t-reject", raw_note=InputNote(content="Test reject.")), config=config_rej)

    res_rej = graph.invoke(
        Command(resume={"decision": "reject", "draft_id": "draft-1"}), config=config_rej
    )
    assert res_rej.get("status") == "rejected"
    assert len(res_rej.get("candidate_drafts", [])) == 0

    # 3. Test Rewrite and Cap Enforcement
    fake_ideas = ExtractedIdeasOutput(ideas=[ExtractedIdea(id="idea-1", summary="Idea 1")])
    fake_draft1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="d1", idea_id="idea-1", platform="x", angle_format="build_log", content="Draft 1")])
    fake_crit1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="d1", voice_match_score=4, clarity_score=4, specificity_score=4, hook_strength_score=4, critique="Good")])

    fake_draft_rw1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="d1", idea_id="idea-1", platform="x", angle_format="build_log", content="Rewrite 1")])
    fake_crit_rw1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="d1", voice_match_score=4, clarity_score=4, specificity_score=4, hook_strength_score=4, critique="Good")])

    fake_draft_rw2 = CandidateDraftsOutput(drafts=[CandidateDraft(id="d1", idea_id="idea-1", platform="x", angle_format="build_log", content="Rewrite 2")])
    fake_crit_rw2 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="d1", voice_match_score=4, clarity_score=4, specificity_score=4, hook_strength_score=4, critique="Good")])

    fake_draft_rw3 = CandidateDraftsOutput(drafts=[CandidateDraft(id="d1", idea_id="idea-1", platform="x", angle_format="build_log", content="Rewrite 3")])
    fake_crit_rw3 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="d1", voice_match_score=4, clarity_score=4, specificity_score=4, hook_strength_score=4, critique="Good")])

    llm_rw = FakeStructuredLLM(responses=[
        fake_ideas, fake_draft1, fake_crit1,
        fake_draft_rw1, fake_crit_rw1,
        fake_draft_rw2, fake_crit_rw2,
        fake_draft_rw3, fake_crit_rw3,
    ])

    graph_rw = create_graph(llm=llm_rw, checkpointer=checkpointer, approval_store=approval_store, max_human_rewrites=3)
    config_rw = {"configurable": {"thread_id": "t-rewrite"}}

    graph_rw.invoke(PostwrightState(thread_id="t-rewrite", raw_note=InputNote(content="Test rewrite cap.")), config=config_rw)

    # Request rewrite 1
    graph_rw.invoke(Command(resume={"decision": "rewrite", "draft_id": "d1", "rewrite_note": "Note 1"}), config=config_rw)
    st = graph_rw.get_state(config_rw)
    assert st.next == ("human_review",)

    # Request rewrite 2
    graph_rw.invoke(Command(resume={"decision": "rewrite", "draft_id": "d1", "rewrite_note": "Note 2"}), config=config_rw)
    st = graph_rw.get_state(config_rw)
    assert st.next == ("human_review",)

    # Request rewrite 3
    graph_rw.invoke(Command(resume={"decision": "rewrite", "draft_id": "d1", "rewrite_note": "Note 3"}), config=config_rw)
    st = graph_rw.get_state(config_rw)
    assert st.next == ("human_review",)

    # 4th rewrite should fail due to cap = 3
    with pytest.raises(ValueError, match="Human-requested rewrites limit \\(3\\) exceeded"):
        graph_rw.invoke(Command(resume={"decision": "rewrite", "draft_id": "d1", "rewrite_note": "Note 4"}), config=config_rw)


def test_no_path_to_record_without_a_decision(tmp_path: Any) -> None:
    store = ApprovalHistoryStore(tmp_path / "store.db")

    state_unapproved = PostwrightState(
        thread_id="t-unapproved",
        selected_draft=CandidateDraft(id="d1", idea_id="i1", platform="x", angle_format="build_log", content="Text"),
        human_decision=None,
    )

    with pytest.raises(ValueError, match="Cannot record post without a valid resume approval/edit decision"):
        record_node(state_unapproved, store=store)


def test_cli_run_review_and_history(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    cp_file = tmp_path / "cli_checkpoints.db"
    store_file = tmp_path / "cli_store.db"

    monkeypatch.setenv("POSTWRIGHT_CHECKPOINT_DB", str(cp_file))
    monkeypatch.setenv("POSTWRIGHT_STORE_DB", str(store_file))

    fake_ideas = ExtractedIdeasOutput(ideas=[ExtractedIdea(id="idea-1", summary="CLI Summary")])
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="CLI original draft content.",
            )
        ]
    )
    fake_critique = CriticOutput(
        critiques=[
            SingleDraftCritiqueOutput(
                draft_id="draft-1",
                voice_match_score=4,
                clarity_score=4,
                specificity_score=4,
                hook_strength_score=4,
                critique="Looks good",
            )
        ]
    )
    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts, fake_critique])

    def mock_create_graph(**kwargs: Any) -> Any:
        from postwright.graph import create_graph
        return create_graph(
            llm=fake_llm,
            store=kwargs.get("store"),
            approval_store=kwargs.get("approval_store"),
            checkpointer=kwargs.get("checkpointer"),
        )

    monkeypatch.setattr("postwright.cli.create_graph", mock_create_graph)

    runner = CliRunner()

    # Run command stops at interrupt
    run_res = runner.invoke(app, ["run", "--note", "CLI test note", "--thread-id", "cli-thread-1"])
    assert run_res.exit_code == 0
    assert "Run paused at human review" in run_res.stdout
    assert "cli-thread-1" in run_res.stdout

    # History prior to approval should be empty
    hist_res1 = runner.invoke(app, ["history"])
    assert "No approved post history found" in hist_res1.stdout

    # Review command approves post
    # Input simulation: decision 'approve', change proposed slot 'n'
    rev_res = runner.invoke(app, ["review", "--thread-id", "cli-thread-1"], input="approve\nn\n")
    assert rev_res.exit_code == 0
    assert "Decision 'approve' applied successfully" in rev_res.stdout

    # History after approval
    hist_res2 = runner.invoke(app, ["history"])
    assert hist_res2.exit_code == 0
    assert "Approved Post History" in hist_res2.stdout
    assert "cli-thread-1" in hist_res2.stdout
