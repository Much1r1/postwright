import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from postwright.llm import get_llm
from postwright.prompts import get_voice_examples, get_voice_profile
from postwright.state import (
    AngledIdea,
    CandidateDraft,
    ExtractedIdea,
    InputNote,
    PostwrightState,
    utc_now,
)
from postwright.store import AngleHistoryStore


class ExtractedIdeasOutput(BaseModel):
    ideas: list[ExtractedIdea] = Field(
        description="3 to 5 postable ideas extracted from the build note."
    )


class CandidateDraftsOutput(BaseModel):
    drafts: list[CandidateDraft] = Field(
        description="Candidate social media drafts generated from angled ideas."
    )


def capture_node(state: PostwrightState | dict[str, Any]) -> dict[str, Any]:
    """Capture and normalize raw input note."""
    if isinstance(state, dict):
        raw_note = state.get("raw_note")
    else:
        raw_note = state.raw_note

    if isinstance(raw_note, InputNote):
        normalized = raw_note
    elif isinstance(raw_note, dict):
        normalized = InputNote(
            content=raw_note.get("content", ""),
            source=raw_note.get("source", "cli"),
            captured_at=raw_note.get("captured_at") or utc_now(),
            project_tag=raw_note.get("project_tag"),
        )
    elif isinstance(raw_note, str):
        normalized = InputNote(content=raw_note, source="cli")
    else:
        normalized = InputNote(content="", source="cli")

    return {"raw_note": normalized}


def extract_ideas_node(
    state: PostwrightState | dict[str, Any],
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Extract 3-5 postable ideas from the raw note as structured output."""
    if isinstance(state, dict):
        raw_note = state.get("raw_note")
    else:
        raw_note = state.raw_note

    note_content = raw_note.content if isinstance(raw_note, InputNote) else str(raw_note)

    if llm is None:
        llm = get_llm()

    system_prompt = (
        "You are an expert technical editor for an AI engineer building in public. "
        "Extract 3 to 5 distinct, postable ideas from the provided build note. "
        "Each idea MUST include concise details on: summary, what was built, what broke, "
        "what was learned, and any concrete numbers/metrics/results if present."
    )

    user_prompt = f"Build Note:\n{note_content}"

    structured_llm = llm.with_structured_output(ExtractedIdeasOutput)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    res = structured_llm.invoke(messages)
    if isinstance(res, ExtractedIdeasOutput):
        ideas = res.ideas
    elif isinstance(res, dict) and "ideas" in res:
        ideas = [ExtractedIdea(**i) if isinstance(i, dict) else i for i in res["ideas"]]
    else:
        ideas = []

    # Ensure each idea has a unique ID if not generated
    for idx, idea in enumerate(ideas):
        if not idea.id:
            idea.id = f"idea-{idx + 1}-{uuid.uuid4().hex[:6]}"

    return {"extracted_ideas": ideas}


def pick_angles_node(
    state: PostwrightState | dict[str, Any],
    store: AngleHistoryStore | None = None,
) -> dict[str, Any]:
    """Assign each extracted idea formats and target platforms rotating history."""
    if isinstance(state, dict):
        extracted_ideas = state.get("extracted_ideas", [])
    else:
        extracted_ideas = state.extracted_ideas

    if store is None:
        store = AngleHistoryStore()

    num_ideas = len(extracted_ideas)
    if num_ideas == 0:
        return {"angled_ideas": []}

    formats = store.pick_next_formats(num_ideas)

    angled_ideas: list[AngledIdea] = []
    for idea, angle_format in zip(extracted_ideas, formats):
        # Create angled ideas for both X and LinkedIn for comprehensive coverage
        for platform in ["x", "linkedin"]:
            angled_ideas.append(
                AngledIdea(
                    idea=idea,
                    angle_format=angle_format,
                    target_platform=platform,  # type: ignore[arg-type]
                )
            )

    return {"angled_ideas": angled_ideas}


def draft_node(
    state: PostwrightState | dict[str, Any],
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Write platform-aware candidate posts per angled idea using voice profile and examples."""
    if isinstance(state, dict):
        angled_ideas = state.get("angled_ideas", [])
    else:
        angled_ideas = state.angled_ideas

    if not angled_ideas:
        return {"candidate_drafts": []}

    if llm is None:
        llm = get_llm()

    voice_profile = get_voice_profile()
    voice_examples = get_voice_examples()

    system_prompt = f"""You are a content ghostwriter for an AI Engineer.
Your job is to generate candidate social media posts matching the creator's voice and style.

VOICE PROFILE:
{voice_profile}

EXAMPLE POSTS:
{voice_examples}

GUIDELINES:
- Generate 2 to 3 candidate posts for each given angled idea.
- For X (Twitter): max 280 characters for single post OR a structured thread (list of parts). Direct, punchy hook.
- For LinkedIn: longer-form (analytical, key takeaways, structured bullet points).
- STRICTLY avoid buzzwords ("game-changer", "delve", "revolutionary", etc.).
- Return structured output containing candidate drafts.
"""

    formatted_ideas = []
    for idx, angled in enumerate(angled_ideas):
        formatted_ideas.append(
            f"Idea #{idx + 1} (ID: {angled.idea.id}):\n"
            f"- Summary: {angled.idea.summary}\n"
            f"- What was built: {angled.idea.what_was_built}\n"
            f"- What broke: {angled.idea.what_broke}\n"
            f"- What was learned: {angled.idea.what_was_learned}\n"
            f"- Numerical result: {angled.idea.numerical_result}\n"
            f"- Assigned Angle Format: {angled.angle_format}\n"
            f"- Target Platform: {angled.target_platform}\n"
        )

    user_prompt = "Generate candidate drafts for these angled ideas:\n\n" + "\n---\n".join(
        formatted_ideas
    )

    structured_llm = llm.with_structured_output(CandidateDraftsOutput)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    res = structured_llm.invoke(messages)
    if isinstance(res, CandidateDraftsOutput):
        drafts = res.drafts
    elif isinstance(res, dict) and "drafts" in res:
        drafts = [CandidateDraft(**d) if isinstance(d, dict) else d for d in res["drafts"]]
    else:
        drafts = []

    for idx, draft in enumerate(drafts):
        if not draft.id:
            draft.id = f"draft-{idx + 1}-{uuid.uuid4().hex[:6]}"

    return {"candidate_drafts": drafts}
