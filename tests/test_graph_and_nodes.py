from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

from postwright.graph import create_graph
from postwright.llm import get_llm
from postwright.nodes import (
    CandidateDraftsOutput,
    CriticOutput,
    ExtractedIdeasOutput,
    SingleDraftCritiqueOutput,
    capture_node,
    draft_node,
    extract_ideas_node,
    pick_angles_node,
)
from postwright.prompts import (
    get_critic_system_prompt,
    get_drafter_system_prompt,
)
from postwright.state import (
    AngledIdea,
    CandidateDraft,
    ExtractedIdea,
    InputNote,
    PostwrightState,
)
from postwright.store import AngleHistoryStore


class FakeStructuredLLM(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)

    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        super().__init__(responses=list(responses), **kwargs)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
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


def test_llm_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    llm = get_llm()
    assert llm is not None


def test_capture_node() -> None:
    res1 = capture_node({"raw_note": "My build note"})
    assert isinstance(res1["raw_note"], InputNote)
    assert res1["raw_note"].content == "My build note"
    assert res1["raw_note"].source == "cli"

    note = InputNote(content="Test note", source="file.md", project_tag="postwright")
    state2 = PostwrightState(raw_note=note)
    res2 = capture_node(state2)
    assert res2["raw_note"].content == "Test note"
    assert res2["raw_note"].source == "file.md"
    assert res2["raw_note"].project_tag == "postwright"


def test_extract_ideas_node() -> None:
    fake_ideas = ExtractedIdeasOutput(
        ideas=[
            ExtractedIdea(
                id="idea-1",
                summary="Debugged SQLite checkpointing",
                what_was_built="Added SQLite checkpointer",
                what_broke="State dropped across re-invocations",
                what_was_learned="Connection pooling needed",
                numerical_result="Fixed 100% state loss",
            ),
            ExtractedIdea(
                id="idea-2",
                summary="Implemented CLI with Typer",
                what_was_built="Typer CLI interface",
                what_broke="None",
                what_was_learned="Rich panels work well",
                numerical_result="Ran 3 tests",
            ),
        ]
    )
    fake_llm = FakeStructuredLLM(responses=[fake_ideas])

    state = PostwrightState(raw_note=InputNote(content="Built SQLite checkpointer and Typer CLI."))
    res = extract_ideas_node(state, llm=fake_llm)

    assert len(res["extracted_ideas"]) == 2
    assert res["extracted_ideas"][0].summary == "Debugged SQLite checkpointing"
    assert res["total_llm_calls"] == 1


def test_pick_angles_node_rotation(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    ideas = [
        ExtractedIdea(id="1", summary="Idea 1"),
        ExtractedIdea(id="2", summary="Idea 2"),
        ExtractedIdea(id="3", summary="Idea 3"),
    ]

    state = PostwrightState(extracted_ideas=ideas)
    res1 = pick_angles_node(state, store=store)

    angled1 = res1["angled_ideas"]
    assert len(angled1) == 6
    formats1 = [a.angle_format for a in angled1[::2]]
    assert formats1 == ["build_log", "lesson", "opinion"]

    ideas2 = [
        ExtractedIdea(id="4", summary="Idea 4"),
        ExtractedIdea(id="5", summary="Idea 5"),
    ]
    state2 = PostwrightState(extracted_ideas=ideas2)
    res2 = pick_angles_node(state2, store=store)
    angled2 = res2["angled_ideas"]
    formats2 = [a.angle_format for a in angled2[::2]]
    assert formats2 == ["diagram_prompt", "build_log"]


def test_draft_node() -> None:
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Spent 3 hours debugging SQLite checkpointer. Fixed by thread-scoped connection pooling.",
            ),
            CandidateDraft(
                id="draft-2",
                idea_id="idea-1",
                platform="linkedin",
                angle_format="build_log",
                content="Here is how we solved state persistence issues with LangGraph and SQLite in production...",
            ),
        ]
    )
    fake_llm = FakeStructuredLLM(responses=[fake_drafts])

    angled_idea = AngledIdea(
        idea=ExtractedIdea(id="idea-1", summary="SQLite fix"),
        angle_format="build_log",
        target_platform="x",
    )
    state = PostwrightState(angled_ideas=[angled_idea])
    res = draft_node(state, llm=fake_llm)

    assert len(res["candidate_drafts"]) == 2
    assert res["candidate_drafts"][0].platform == "x"
    assert res["candidate_drafts"][1].platform == "linkedin"
    assert res["total_llm_calls"] == 1


def test_pass_on_first_try(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="LangGraph core implementation")]
    )
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Built core LangGraph nodes today with 100% test coverage.",
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
                critique="Solid post with clear metric.",
            )
        ]
    )

    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts, fake_critique])
    graph = create_graph(llm=fake_llm, store=store, critic_score_threshold=14, max_revisions=2)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    assert final_state["revision_count"] == 0
    assert final_state["below_threshold"] is False
    assert final_state["total_llm_calls"] == 3
    assert len(final_state["candidate_drafts"]) == 1
    assert final_state["candidate_drafts"][0].score == 16


def test_failing_draft_revised_and_better_version_kept(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="LangGraph core implementation")]
    )
    # Draft 1 initial (gets score 10 < 14)
    fake_drafts_v1 = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Built core LangGraph nodes.",
            )
        ]
    )
    fake_critique_v1 = CriticOutput(
        critiques=[
            SingleDraftCritiqueOutput(
                draft_id="draft-1",
                voice_match_score=2,
                clarity_score=3,
                specificity_score=2,
                hook_strength_score=3,
                critique="Missing specific metrics and weak hook.",
            )
        ]
    )
    # Revision 1 (gets score 16 >= 14)
    fake_drafts_v2 = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Spent 2 hours wiring 4 LangGraph nodes. Reached 100% test coverage.",
            )
        ]
    )
    fake_critique_v2 = CriticOutput(
        critiques=[
            SingleDraftCritiqueOutput(
                draft_id="draft-1",
                voice_match_score=4,
                clarity_score=4,
                specificity_score=4,
                hook_strength_score=4,
                critique="Great revision with clear numbers and strong hook.",
            )
        ]
    )

    fake_llm = FakeStructuredLLM(
        responses=[fake_ideas, fake_drafts_v1, fake_critique_v1, fake_drafts_v2, fake_critique_v2]
    )
    graph = create_graph(llm=fake_llm, store=store, critic_score_threshold=14, max_revisions=2)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    assert final_state["revision_count"] == 1
    assert final_state["below_threshold"] is False
    assert final_state["candidate_drafts"][0].score == 16
    assert "Spent 2 hours wiring 4 LangGraph nodes" in final_state["candidate_drafts"][0].content


def test_loop_stops_at_cap_and_flags_below_threshold(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="LangGraph core implementation")]
    )

    # Initial draft + 2 revisions, all failing
    fake_drafts_1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="V1 draft")])
    fake_critique_1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=2, specificity_score=2, hook_strength_score=2, critique="V1 weak")])

    fake_drafts_2 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="V2 draft")])
    fake_critique_2 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=3, specificity_score=2, hook_strength_score=2, critique="V2 still weak")])

    fake_drafts_3 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="V3 draft")])
    fake_critique_3 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=3, clarity_score=3, specificity_score=2, hook_strength_score=2, critique="V3 still below threshold")])

    fake_llm = FakeStructuredLLM(
        responses=[
            fake_ideas,
            fake_drafts_1, fake_critique_1,
            fake_drafts_2, fake_critique_2,
            fake_drafts_3, fake_critique_3,
        ]
    )

    graph = create_graph(llm=fake_llm, store=store, critic_score_threshold=14, max_revisions=2)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    assert final_state["revision_count"] == 2
    assert final_state["below_threshold"] is True
    assert final_state["candidate_drafts"][0].below_threshold is True
    assert final_state["candidate_drafts"][0].score == 10  # score of V3 (score 10)


def test_revision_never_overwrites_higher_scoring_earlier_version(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="LangGraph core implementation")]
    )

    # Initial draft gets score 13 (< 14 threshold)
    fake_drafts_1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="V1 draft with score 13")])
    fake_critique_1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=3, clarity_score=3, specificity_score=3, hook_strength_score=4, critique="V1 score 13")])

    # Revision 1 gets score 8 (< 13)
    fake_drafts_2 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Worse V2 draft with score 8")])
    fake_critique_2 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=2, specificity_score=2, hook_strength_score=2, critique="V2 score 8")])

    # Revision 2 gets score 16 (>= 14)
    fake_drafts_3 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Great V3 draft with score 16")])
    fake_critique_3 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=4, clarity_score=4, specificity_score=4, hook_strength_score=4, critique="V3 score 16")])

    fake_llm = FakeStructuredLLM(
        responses=[
            fake_ideas,
            fake_drafts_1, fake_critique_1,
            fake_drafts_2, fake_critique_2,
            fake_drafts_3, fake_critique_3,
        ]
    )

    graph = create_graph(llm=fake_llm, store=store, critic_score_threshold=14, max_revisions=2)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    # V2 was ignored because score 8 < score 13. Then V3 (score 16) was kept!
    assert final_state["revision_count"] == 2
    assert final_state["below_threshold"] is False
    assert final_state["candidate_drafts"][0].score == 16
    assert "Great V3 draft" in final_state["candidate_drafts"][0].content


def test_lower_scoring_revision_preserves_earlier_draft_content(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="LangGraph core implementation")]
    )

    fake_drafts_1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Original score 12 content")])
    fake_critique_1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=3, clarity_score=3, specificity_score=3, hook_strength_score=3, critique="V1 score 12")])

    fake_drafts_2 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Worse score 8 content")])
    fake_critique_2 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=2, specificity_score=2, hook_strength_score=2, critique="V2 score 8")])

    fake_drafts_3 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Worse score 10 content")])
    fake_critique_3 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=3, specificity_score=2, hook_strength_score=3, critique="V3 score 10")])

    fake_llm = FakeStructuredLLM(
        responses=[
            fake_ideas,
            fake_drafts_1, fake_critique_1,
            fake_drafts_2, fake_critique_2,
            fake_drafts_3, fake_critique_3,
        ]
    )

    graph = create_graph(llm=fake_llm, store=store, critic_score_threshold=14, max_revisions=2)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    assert final_state["revision_count"] == 2
    assert final_state["below_threshold"] is True
    assert final_state["candidate_drafts"][0].score == 12
    assert final_state["candidate_drafts"][0].content == "Original score 12 content"


def test_critic_prompt_separate_and_no_drafter_prompt() -> None:
    critic_prompt = get_critic_system_prompt()
    drafter_prompt = get_drafter_system_prompt()

    assert "EVALUATION CRITERIA" in critic_prompt
    assert "voice_match" in critic_prompt
    assert drafter_prompt not in critic_prompt
