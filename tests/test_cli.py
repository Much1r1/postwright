from pathlib import Path
from typing import Any

import pytest
from test_graph_and_nodes import CandidateDraftsOutput, ExtractedIdeasOutput, FakeStructuredLLM
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
    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts])

    def mock_create_graph(**kwargs: Any) -> Any:
        from postwright.graph import create_graph
        return create_graph(llm=fake_llm, store=kwargs.get("store"))

    monkeypatch.setattr("postwright.cli.create_graph", mock_create_graph)

    result = runner.invoke(app, ["run", "--note", "CLI note content"])
    assert result.exit_code == 0
    assert "Generated 1 candidate draft(s)" in result.stdout
    assert "Test draft content for CLI" in result.stdout


def test_cli_run_with_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    note_file = tmp_path / "notes.md"
    note_file.write_text("Note content from file", encoding="utf-8")

    fake_ideas = ExtractedIdeasOutput(
        ideas=[ExtractedIdea(id="idea-1", summary="File Test")]
    )
    fake_drafts = CandidateDraftsOutput(
        drafts=[
            CandidateDraft(
                id="draft-1",
                idea_id="idea-1",
                platform="linkedin",
                angle_format="lesson",
                content="Test draft content from file",
            )
        ]
    )
    fake_llm = FakeStructuredLLM(responses=[fake_ideas, fake_drafts])

    def mock_create_graph(**kwargs: Any) -> Any:
        from postwright.graph import create_graph
        return create_graph(llm=fake_llm, store=kwargs.get("store"))

    monkeypatch.setattr("postwright.cli.create_graph", mock_create_graph)

    result = runner.invoke(app, ["run", "--file", str(note_file)])
    assert result.exit_code == 0
    assert "Generated 1 candidate draft(s)" in result.stdout
    assert "Test draft content from file" in result.stdout
