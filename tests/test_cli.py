from typing import Any

import pytest
from test_graph_and_nodes import (
    CandidateDraftsOutput,
    CriticOutput,
    ExtractedIdeasOutput,
    FakeStructuredLLM,
    SingleDraftCritiqueOutput,
)
from typer.testing import CliRunner

from postwright.cli import app
from postwright.state import CandidateDraft, ExtractedIdea

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Process build notes and generate social media drafts" in result.stdout or "Usage:" in result.stdout


def test_cli_missing_args() -> None:
    result = runner.invoke(app, ["run", "--note", ""])
    assert result.exit_code == 1
    assert "Either --note or --file must be provided" in result.stdout


def test_cli_file_not_found() -> None:
    result = runner.invoke(app, ["run", "--file", "non_existent.md"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_cli_run_with_note(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="CLI Test")]
    )
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="x",
                angle_format="build_log",
                content="Test draft content for CLI",
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
                critique="Looks great",
            )
        ]
    )
    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts, fake_critique])

    def mock_create_graph(**kwargs: Any) -> Any:
        from postwright.graph import create_graph
        return create_graph(llm=fake_llm, store=kwargs.get("store"))

    monkeypatch.setattr("postwright.cli.create_graph", mock_create_graph)

    result = runner.invoke(app, ["run", "--note", "CLI note content"])
    assert result.exit_code == 0
    assert "Generated 1 candidate draft(s)" in result.stdout
    assert "Test draft content for CLI" in result.stdout
    assert "Score: 16/20" in result.stdout
    assert "Revisions: 0" in result.stdout


def test_cli_run_below_threshold_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="Below threshold test")]
    )
    fake_drafts_1 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Draft v1")])
    fake_critique_1 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=2, specificity_score=2, hook_strength_score=2, critique="Needs improvement")])
    fake_drafts_2 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Draft v2")])
    fake_critique_2 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=3, specificity_score=2, hook_strength_score=2, critique="Still needs work")])
    fake_drafts_3 = CandidateDraftsOutput(drafts=[CandidateDraft(id="draft-1", idea_id="idea-1", platform="x", angle_format="build_log", content="Draft v3")])
    fake_critique_3 = CriticOutput(critiques=[SingleDraftCritiqueOutput(draft_id="draft-1", voice_match_score=2, clarity_score=3, specificity_score=2, hook_strength_score=3, critique="Weak hook")])

    fake_llm = FakeStructuredLLM(
        responses=[
            fake_ideas,
            fake_drafts_1, fake_critique_1,
            fake_drafts_2, fake_critique_2,
            fake_drafts_3, fake_critique_3,
        ]
    )

    def mock_create_graph(**kwargs: Any) -> Any:
        from postwright.graph import create_graph
        return create_graph(llm=fake_llm, store=kwargs.get("store"), critic_score_threshold=14, max_revisions=2)

    monkeypatch.setattr("postwright.cli.create_graph", mock_create_graph)

    result = runner.invoke(app, ["run", "--note", "Note for below threshold"])
    assert result.exit_code == 0
    assert "Score: 10/20" in result.stdout
    assert "Revisions: 2" in result.stdout
    assert "BELOW THRESHOLD" in result.stdout
