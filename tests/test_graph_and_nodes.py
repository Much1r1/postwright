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
    ExtractedIdeasOutput,
    capture_node,
    draft_node,
    extract_ideas_node,
    pick_angles_node,
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
    # Test dictionary state with string note
    res1 = capture_node({"raw_note": "My build note"})
    assert isinstance(res1["raw_note"], InputNote)
    assert res1["raw_note"].content == "My build note"
    assert res1["raw_note"].source == "cli"

    # Test InputNote object state
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
    # 3 ideas x 2 platforms (x, linkedin) = 6 angled ideas
    assert len(angled1) == 6
    formats1 = [a.angle_format for a in angled1[::2]]
    assert formats1 == ["build_log", "lesson", "opinion"]

    # Run again for 2 ideas, should continue rotation starting from diagram_prompt
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


def test_full_graph_end_to_end(tmp_path: Any) -> None:
    db_file = tmp_path / "test_store.db"
    store = AngleHistoryStore(db_path=db_file)

    fake_ideas = ExtractedIdeasOutput(
        ideas=[
            ExtractedIdea(
                id="idea-1",
                summary="LangGraph core implementation",
                what_was_built="StateGraph with nodes",
                what_broke="Missing edge",
                what_was_learned="Wire nodes explicitly",
                numerical_result="4 nodes implemented",
            )
        ]
    )
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Built core LangGraph nodes today.",
            )
        ]
    )

    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts])

    graph = create_graph(llm=fake_llm, store=store)

    initial_state = PostwrightState(raw_note=InputNote(content="Implemented LangGraph core nodes."))
    final_state = graph.invoke(initial_state)

    assert final_state["raw_note"].content == "Implemented LangGraph core nodes."
    assert len(final_state["extracted_ideas"]) == 1
    assert len(final_state["angled_ideas"]) == 2  # x + linkedin
    assert len(final_state["candidate_drafts"]) == 1
    assert final_state["candidate_drafts"][0].content == "Built core LangGraph nodes today."
