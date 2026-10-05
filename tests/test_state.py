from postwright.state import (
    DraftCritique,
    InputNote,
    PostwrightState,
)


def test_input_note_creation() -> None:
    note = InputNote(content="Test note content", source="file", project_tag="postwright")
    assert note.content == "Test note content"
    assert note.source == "file"
    assert note.project_tag == "postwright"


def test_draft_critique_total_score() -> None:
    critique = DraftCritique(
        draft_id="draft-1",
        voice_match_score=4,
        clarity_score=5,
        specificity_score=3,
        hook_strength_score=4,
        critique_notes="Good hook, needs more specific numbers.",
        needs_revision=False,
    )
    assert critique.total_score == 16


def test_postwright_state_initialization() -> None:
    state = PostwrightState()
    assert state.raw_note is None
    assert state.extracted_ideas == []
    assert state.revision_count == 0
    assert state.status == "pending"
