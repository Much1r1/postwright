from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class InputNote(BaseModel):
    content: str
    source: str = "cli"
    captured_at: datetime = Field(default_factory=utc_now)
    project_tag: str | None = None


class ExtractedIdea(BaseModel):
    id: str
    summary: str
    what_was_built: str | None = None
    what_broke: str | None = None
    what_was_learned: str | None = None
    numerical_result: str | None = None


class AngledIdea(BaseModel):
    idea: ExtractedIdea
    angle_format: Literal["build_log", "lesson", "opinion", "diagram_prompt"]
    target_platform: Literal["x", "linkedin"]


class ScheduledSlot(BaseModel):
    platform: Literal["x", "linkedin"]
    scheduled_at_utc: datetime
    scheduled_at_local: datetime
    user_timezone: str = "Africa/Nairobi"


class CandidateDraft(BaseModel):
    id: str
    idea_id: str
    platform: Literal["x", "linkedin"]
    angle_format: Literal["build_log", "lesson", "opinion", "diagram_prompt"]
    content: str
    is_thread: bool = False
    thread_parts: list[str] = Field(default_factory=list)
    score: int | None = None
    critique: str | None = None
    revision_count: int = 0
    below_threshold: bool = False
    previous_best_content: str | None = None
    previous_best_score: int | None = None
    proposed_slot: ScheduledSlot | None = None


class DraftCritique(BaseModel):
    draft_id: str
    voice_match_score: int = Field(ge=1, le=5)
    clarity_score: int = Field(ge=1, le=5)
    specificity_score: int = Field(ge=1, le=5)
    hook_strength_score: int = Field(ge=1, le=5)
    critique_notes: str
    needs_revision: bool

    @property
    def total_score(self) -> int:
        return (
            self.voice_match_score
            + self.clarity_score
            + self.specificity_score
            + self.hook_strength_score
        )


class HumanReviewDecision(BaseModel):
    decision: Literal["approve", "edit", "reject", "rewrite"]
    draft_id: str | None = None
    edit_text: str | None = None
    rewrite_note: str | None = None
    slot_override: ScheduledSlot | str | None = None


class PostwrightState(BaseModel):
    thread_id: str | None = None
    raw_note: InputNote | str | dict[str, Any] | None = None
    extracted_ideas: list[ExtractedIdea] = Field(default_factory=list)
    angled_ideas: list[AngledIdea] = Field(default_factory=list)
    candidate_drafts: list[CandidateDraft] = Field(default_factory=list)
    critiques: list[DraftCritique] = Field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 2
    critic_score_threshold: int = 14
    total_llm_calls: int = 0
    draft_revision_counts: dict[str, int] = Field(default_factory=dict)
    human_rewrite_counts: dict[str, int] = Field(default_factory=dict)
    max_human_rewrites: int = 3
    below_threshold: bool = False
    selected_draft: CandidateDraft | None = None
    proposed_slot: ScheduledSlot | None = None
    human_decision: Literal["approve", "edit", "reject", "rewrite"] | None = None
    user_feedback: str | None = None
    edit_diff: str | dict[str, Any] | None = None
    status: str = "pending"
