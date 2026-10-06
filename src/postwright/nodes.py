import difflib
import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from postwright.config import get_settings
from postwright.llm import get_llm
from postwright.platforms.registry import get_adapter
from postwright.prompts import (
    get_critic_system_prompt,
    get_drafter_system_prompt,
    get_revision_system_prompt,
)
from postwright.scheduler import SlotPolicy, UniversalPriorPolicy
from postwright.state import (
    AngledIdea,
    CandidateDraft,
    DraftCritique,
    ExtractedIdea,
    HumanReviewDecision,
    InputNote,
    PostwrightState,
    ScheduledSlot,
    utc_now,
)
from postwright.store import AngleHistoryStore, ApprovalHistoryStore, QueueStore


class ExtractedIdeasOutput(BaseModel):
    ideas: list[ExtractedIdea] = Field(
        description="3 to 5 postable ideas extracted from the build note."
    )


class CandidateDraftsOutput(BaseModel):
    drafts: list[CandidateDraft] = Field(
        description="Candidate social media drafts generated from angled ideas."
    )


class SingleDraftCritiqueOutput(BaseModel):
    draft_id: str
    voice_match_score: int = Field(ge=1, le=5)
    clarity_score: int = Field(ge=1, le=5)
    specificity_score: int = Field(ge=1, le=5)
    hook_strength_score: int = Field(ge=1, le=5)
    critique: str


class CriticOutput(BaseModel):
    critiques: list[SingleDraftCritiqueOutput] = Field(
        description="Critiques and scores for candidate drafts."
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
        total_llm_calls = state.get("total_llm_calls", 0)
    else:
        raw_note = state.raw_note
        total_llm_calls = state.total_llm_calls

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

    for idx, idea in enumerate(ideas):
        if not idea.id:
            idea.id = f"idea-{idx + 1}-{uuid.uuid4().hex[:6]}"

    return {"extracted_ideas": ideas, "total_llm_calls": total_llm_calls + 1}


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
    """Write platform-aware candidate posts or regenerate failing drafts."""
    if isinstance(state, dict):
        angled_ideas = state.get("angled_ideas", [])
        existing_drafts = state.get("candidate_drafts", [])
        total_llm_calls = state.get("total_llm_calls", 0)
        draft_revision_counts = dict(state.get("draft_revision_counts", {}))
        revision_count = state.get("revision_count", 0)
        threshold = state.get("critic_score_threshold", get_settings().critic_score_threshold)
    else:
        angled_ideas = state.angled_ideas
        existing_drafts = state.candidate_drafts
        total_llm_calls = state.total_llm_calls
        draft_revision_counts = dict(state.draft_revision_counts)
        revision_count = state.revision_count
        threshold = state.critic_score_threshold

    if not angled_ideas:
        return {"candidate_drafts": []}

    if llm is None:
        llm = get_llm()

    if not existing_drafts:
        system_prompt = get_drafter_system_prompt()
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

        return {
            "candidate_drafts": drafts,
            "total_llm_calls": total_llm_calls + 1,
            "draft_revision_counts": draft_revision_counts,
        }

    user_feedback = (
        state.get("user_feedback")
        if isinstance(state, dict)
        else getattr(state, "user_feedback", None)
    )

    failing_drafts = [d for d in existing_drafts if d.score is None or d.score < threshold or user_feedback]
    if not failing_drafts:
        return {"candidate_drafts": existing_drafts, "user_feedback": None}

    system_prompt = get_revision_system_prompt()

    failing_descriptions = []
    for d in failing_drafts:
        fb_text = d.critique or "Needs higher specificity, clarity, or hook strength."
        if user_feedback:
            fb_text += f"\nHuman Feedback / Note for Rewrite: {user_feedback}"
        failing_descriptions.append(
            f"Draft ID: {d.id}\n"
            f"Platform: {d.platform}\n"
            f"Angle Format: {d.angle_format}\n"
            f"Current Content:\n{d.content}\n"
            f"Critique / Feedback:\n{fb_text}\n"
        )

    user_prompt = (
        "Please revise only the following social media drafts addressing the feedback provided:\n\n"
        + "\n---\n".join(failing_descriptions)
    )

    structured_llm = llm.with_structured_output(CandidateDraftsOutput)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    res = structured_llm.invoke(messages)
    if isinstance(res, CandidateDraftsOutput):
        new_drafts = res.drafts
    elif isinstance(res, dict) and "drafts" in res:
        new_drafts = [CandidateDraft(**d) if isinstance(d, dict) else d for d in res["drafts"]]
    else:
        new_drafts = []

    updated_drafts = list(existing_drafts)
    for idx, failing_d in enumerate(failing_drafts):
        matching_new = None
        for nd in new_drafts:
            if nd.id == failing_d.id:
                matching_new = nd
                break
        if not matching_new and idx < len(new_drafts):
            matching_new = new_drafts[idx]

        if matching_new:
            draft_revision_counts[failing_d.id] = (
                draft_revision_counts.get(failing_d.id, 0) + 1
            )
            best_content = failing_d.previous_best_content or failing_d.content
            best_score = (
                failing_d.previous_best_score
                if failing_d.previous_best_score is not None
                else failing_d.score
            )

            for i, d in enumerate(updated_drafts):
                if d.id == failing_d.id:
                    updated_drafts[i] = CandidateDraft(
                        id=failing_d.id,
                        idea_id=failing_d.idea_id,
                        platform=failing_d.platform,
                        angle_format=failing_d.angle_format,
                        content=matching_new.content,
                        is_thread=matching_new.is_thread,
                        thread_parts=matching_new.thread_parts,
                        score=failing_d.score,
                        critique=failing_d.critique,
                        revision_count=draft_revision_counts[failing_d.id],
                        below_threshold=failing_d.below_threshold,
                        previous_best_content=best_content,
                        previous_best_score=best_score,
                        proposed_slot=failing_d.proposed_slot,
                    )
                    break

    return {
        "candidate_drafts": updated_drafts,
        "total_llm_calls": total_llm_calls + 1,
        "draft_revision_counts": draft_revision_counts,
        "revision_count": revision_count + 1,
        "user_feedback": None,
    }


def critic_node(
    state: PostwrightState | dict[str, Any],
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Score candidate drafts on voice, clarity, specificity, and hook strength."""
    if isinstance(state, dict):
        candidate_drafts = state.get("candidate_drafts", [])
        raw_note = state.get("raw_note")
        threshold = state.get("critic_score_threshold", get_settings().critic_score_threshold)
        max_revisions = state.get("max_revisions", get_settings().max_revisions)
        total_llm_calls = state.get("total_llm_calls", 0)
        revision_count = state.get("revision_count", 0)
        existing_critiques = list(state.get("critiques", []))
    else:
        candidate_drafts = state.candidate_drafts
        raw_note = state.raw_note
        threshold = state.critic_score_threshold
        max_revisions = state.max_revisions
        total_llm_calls = state.total_llm_calls
        revision_count = state.revision_count
        existing_critiques = list(state.critiques)

    if not candidate_drafts:
        return {"critiques": [], "below_threshold": False}

    note_content = raw_note.content if isinstance(raw_note, InputNote) else str(raw_note)

    if llm is None:
        llm = get_llm()

    system_prompt = get_critic_system_prompt()

    draft_descriptions = []
    for d in candidate_drafts:
        parts_str = ""
        if d.is_thread and d.thread_parts:
            parts_str = "\nThread parts: " + " | ".join(d.thread_parts)
        draft_descriptions.append(
            f"Draft ID: {d.id}\n"
            f"Platform: {d.platform}\n"
            f"Angle Format: {d.angle_format}\n"
            f"Content: {d.content}{parts_str}\n"
        )

    user_prompt = (
        f"Raw Build Note:\n{note_content}\n\n"
        "Candidate Drafts to Evaluate:\n"
        + "\n---\n".join(draft_descriptions)
    )

    structured_llm = llm.with_structured_output(CriticOutput)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    res = structured_llm.invoke(messages)
    if isinstance(res, CriticOutput):
        items = res.critiques
    elif isinstance(res, dict) and "critiques" in res:
        items = [
            SingleDraftCritiqueOutput(**c) if isinstance(c, dict) else c
            for c in res["critiques"]
        ]
    elif isinstance(res, list):
        items = [
            SingleDraftCritiqueOutput(**c) if isinstance(c, dict) else c
            for c in res
        ]
    else:
        items = []

    updated_drafts: list[CandidateDraft] = []
    new_critiques: list[DraftCritique] = list(existing_critiques)
    any_below_threshold = False

    for idx, d in enumerate(candidate_drafts):
        critique_item = None
        for item in items:
            if item.draft_id == d.id:
                critique_item = item
                break
        if not critique_item and idx < len(items):
            critique_item = items[idx]

        if critique_item:
            total_score = (
                critique_item.voice_match_score
                + critique_item.clarity_score
                + critique_item.specificity_score
                + critique_item.hook_strength_score
            )
            critique_notes = critique_item.critique

            crit_obj = DraftCritique(
                draft_id=d.id,
                voice_match_score=critique_item.voice_match_score,
                clarity_score=critique_item.clarity_score,
                specificity_score=critique_item.specificity_score,
                hook_strength_score=critique_item.hook_strength_score,
                critique_notes=critique_notes,
                needs_revision=(total_score < threshold),
            )
            new_critiques.append(crit_obj)

            prev_best_score = (
                d.previous_best_score if d.previous_best_score is not None else d.score
            )
            prev_best_content = (
                d.previous_best_content if d.previous_best_content is not None else d.content
            )

            if prev_best_score is not None and prev_best_score > total_score:
                is_below = prev_best_score < threshold
                best_draft = CandidateDraft(
                    id=d.id,
                    idea_id=d.idea_id,
                    platform=d.platform,
                    angle_format=d.angle_format,
                    content=prev_best_content,
                    is_thread=d.is_thread,
                    thread_parts=d.thread_parts,
                    score=prev_best_score,
                    critique=d.critique,
                    revision_count=d.revision_count,
                    below_threshold=is_below if revision_count >= max_revisions else False,
                    previous_best_content=prev_best_content,
                    previous_best_score=prev_best_score,
                    proposed_slot=d.proposed_slot,
                )
            else:
                is_below = total_score < threshold
                best_draft = CandidateDraft(
                    id=d.id,
                    idea_id=d.idea_id,
                    platform=d.platform,
                    angle_format=d.angle_format,
                    content=d.content,
                    is_thread=d.is_thread,
                    thread_parts=d.thread_parts,
                    score=total_score,
                    critique=critique_notes,
                    revision_count=d.revision_count,
                    below_threshold=is_below if revision_count >= max_revisions else False,
                    previous_best_content=d.content,
                    previous_best_score=total_score,
                    proposed_slot=d.proposed_slot,
                )

            if (
                revision_count >= max_revisions
                and best_draft.score is not None
                and best_draft.score < threshold
            ):
                best_draft = CandidateDraft(
                    id=best_draft.id,
                    idea_id=best_draft.idea_id,
                    platform=best_draft.platform,
                    angle_format=best_draft.angle_format,
                    content=best_draft.content,
                    is_thread=best_draft.is_thread,
                    thread_parts=best_draft.thread_parts,
                    score=best_draft.score,
                    critique=best_draft.critique,
                    revision_count=best_draft.revision_count,
                    below_threshold=True,
                    previous_best_content=best_draft.previous_best_content,
                    previous_best_score=best_draft.previous_best_score,
                    proposed_slot=best_draft.proposed_slot,
                )
                any_below_threshold = True

            updated_drafts.append(best_draft)
        else:
            updated_drafts.append(d)

    return {
        "candidate_drafts": updated_drafts,
        "critiques": new_critiques,
        "total_llm_calls": total_llm_calls + 1,
        "below_threshold": any_below_threshold,
    }


def pick_slot_node(
    state: PostwrightState | dict[str, Any],
    policy: SlotPolicy | None = None,
    queue_store: QueueStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pick proposed slots for candidate drafts after critic loop."""
    if isinstance(state, dict):
        candidate_drafts = state.get("candidate_drafts", [])
    else:
        candidate_drafts = state.candidate_drafts

    if not candidate_drafts:
        return {"candidate_drafts": []}

    if policy is None:
        policy = UniversalPriorPolicy()
    if queue_store is None:
        queue_store = QueueStore()
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    queued_records = queue_store.get_queued_posts(status="queued")
    already_booked = [
        datetime.fromisoformat(r.slot_utc).replace(tzinfo=UTC)
        for r in queued_records
    ]

    assigned_drafts = []
    assigned_slots_during_run: list[ScheduledSlot] = []

    for d in candidate_drafts:
        all_queued = list(already_booked) + [s.scheduled_at_utc for s in assigned_slots_during_run]
        slot = policy.suggest(platform=d.platform, now=now, queued_slots=all_queued)  # type: ignore[arg-type]

        updated_d = CandidateDraft(
            id=d.id,
            idea_id=d.idea_id,
            platform=d.platform,
            angle_format=d.angle_format,
            content=d.content,
            is_thread=d.is_thread,
            thread_parts=d.thread_parts,
            score=d.score,
            critique=d.critique,
            revision_count=d.revision_count,
            below_threshold=d.below_threshold,
            previous_best_content=d.previous_best_content,
            previous_best_score=d.previous_best_score,
            proposed_slot=slot,
        )
        assigned_drafts.append(updated_d)
        assigned_slots_during_run.append(slot)

    return {"candidate_drafts": assigned_drafts}


def human_review_node(
    state: PostwrightState | dict[str, Any],
    queue_store: QueueStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pause graph execution for human review and process resume decision."""
    if isinstance(state, dict):
        candidate_drafts = state.get("candidate_drafts", [])
        thread_id = state.get("thread_id")
        human_rewrite_counts = dict(state.get("human_rewrite_counts", {}))
        max_human_rewrites = state.get(
            "max_human_rewrites", get_settings().max_human_rewrites
        )
    else:
        candidate_drafts = state.candidate_drafts
        thread_id = state.thread_id
        human_rewrite_counts = dict(state.human_rewrite_counts)
        max_human_rewrites = state.max_human_rewrites

    top_drafts_payload = []
    for d in candidate_drafts:
        slot_dict = d.proposed_slot.model_dump(mode="json") if d.proposed_slot else None

        # Validate and split via platform adapter to show thread splits / errors in review
        adapter = get_adapter(d.platform)
        val = adapter.validate(d.content)

        top_drafts_payload.append(
            {
                "id": d.id,
                "idea_id": d.idea_id,
                "platform": d.platform,
                "angle_format": d.angle_format,
                "content": d.content,
                "score": d.score,
                "critique": d.critique,
                "revision_count": d.revision_count,
                "below_threshold": d.below_threshold,
                "proposed_slot": slot_dict,
                "is_valid": val.is_valid,
                "validation_errors": val.errors,
                "thread_parts": val.thread_parts,
            }
        )

    interrupt_data = {
        "drafts": top_drafts_payload,
        "thread_id": thread_id,
    }

    resume_input = interrupt(interrupt_data)

    if isinstance(resume_input, HumanReviewDecision):
        decision = resume_input
    elif isinstance(resume_input, dict):
        decision = HumanReviewDecision(**resume_input)
    else:
        raise TypeError("Invalid resume decision provided to human_review node.")

    decision_type = decision.decision

    if decision_type in ("approve", "edit"):
        selected_draft = None
        if decision.draft_id:
            for d in candidate_drafts:
                if d.id == decision.draft_id:
                    selected_draft = d
                    break
        if not selected_draft and candidate_drafts:
            selected_draft = candidate_drafts[0]

        if not selected_draft:
            raise ValueError("No candidate draft available to approve/edit.")

        original_text = selected_draft.content
        final_text = (
            decision.edit_text
            if decision_type == "edit" and decision.edit_text is not None
            else original_text
        )

        # Validate replacement text with adapter - refuse edit if it breaks limits
        adapter = get_adapter(selected_draft.platform)
        val_result = adapter.validate(final_text)
        if not val_result.is_valid:
            err_msg = "; ".join(val_result.errors)
            raise ValueError(
                f"Edit refused: Content breaks platform limits for {selected_draft.platform.upper()}. ({err_msg})"
            )

        # Handle slot override if provided
        final_slot = selected_draft.proposed_slot
        if decision.slot_override:
            if queue_store is None:
                queue_store = QueueStore()
            if now is None:
                now_utc = datetime.now(UTC)
            elif now.tzinfo is None:
                now_utc = now.replace(tzinfo=UTC)
            else:
                now_utc = now.astimezone(UTC)

            if isinstance(decision.slot_override, ScheduledSlot):
                override_slot = decision.slot_override
            elif isinstance(decision.slot_override, str):
                dt = datetime.fromisoformat(decision.slot_override)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                else:
                    dt = dt.astimezone(UTC)
                user_tz = ZoneInfo("Africa/Nairobi")
                override_slot = ScheduledSlot(
                    platform=selected_draft.platform,
                    scheduled_at_utc=dt,
                    scheduled_at_local=dt.astimezone(user_tz),
                    user_timezone="Africa/Nairobi",
                )
            else:
                raise ValueError(f"Invalid slot_override type: {type(decision.slot_override)}")

            if override_slot.scheduled_at_utc <= now_utc:
                raise ValueError(
                    f"Slot override '{override_slot.scheduled_at_utc.isoformat()}' is in the past."
                )

            queued_records = queue_store.get_queued_posts(status="queued")
            for rec in queued_records:
                rec_dt = datetime.fromisoformat(rec.slot_utc)
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=UTC)
                else:
                    rec_dt = rec_dt.astimezone(UTC)
                if rec_dt == override_slot.scheduled_at_utc:
                    raise ValueError(
                        f"Slot override '{override_slot.scheduled_at_utc.isoformat()}' is already taken in post queue."
                    )

            final_slot = override_slot

        edit_diff = ""
        if decision_type == "edit":
            diff_lines = list(
                difflib.unified_diff(
                    original_text.splitlines(keepends=True),
                    final_text.splitlines(keepends=True),
                    fromfile="original",
                    tofile="edited",
                )
            )
            edit_diff = "".join(diff_lines)

        final_draft = CandidateDraft(
            id=selected_draft.id,
            idea_id=selected_draft.idea_id,
            platform=selected_draft.platform,
            angle_format=selected_draft.angle_format,
            content=final_text,
            is_thread=len(val_result.thread_parts) > 1,
            thread_parts=val_result.thread_parts,
            score=selected_draft.score,
            critique=selected_draft.critique,
            revision_count=selected_draft.revision_count,
            below_threshold=selected_draft.below_threshold,
            previous_best_content=original_text,
            previous_best_score=selected_draft.score,
            proposed_slot=final_slot,
        )

        return {
            "selected_draft": final_draft,
            "proposed_slot": final_slot,
            "human_decision": decision_type,
            "status": "approved",
            "edit_diff": edit_diff,
        }

    elif decision_type == "reject":
        target_id = decision.draft_id
        if target_id:
            updated_candidate_drafts = [d for d in candidate_drafts if d.id != target_id]
        else:
            updated_candidate_drafts = []

        return {
            "candidate_drafts": updated_candidate_drafts,
            "human_decision": "reject",
            "status": "rejected" if not updated_candidate_drafts else "pending_review",
        }

    elif decision_type == "rewrite":
        target_id = decision.draft_id or (
            candidate_drafts[0].id if candidate_drafts else "default"
        )
        current_rewrites = human_rewrite_counts.get(target_id, 0)
        if current_rewrites >= max_human_rewrites:
            raise ValueError(
                f"Human-requested rewrites limit ({max_human_rewrites}) exceeded for draft {target_id}."
            )

        human_rewrite_counts[target_id] = current_rewrites + 1

        return {
            "human_rewrite_counts": human_rewrite_counts,
            "user_feedback": decision.rewrite_note,
            "human_decision": "rewrite",
            "status": "needs_rewrite",
        }

    else:
        raise ValueError(f"Unknown human decision type: {decision_type}")


def schedule_node(
    state: PostwrightState | dict[str, Any],
    queue_store: QueueStore | None = None,
) -> dict[str, Any]:
    """Write approved post into SQLite queue table."""
    if isinstance(state, dict):
        human_decision = state.get("human_decision")
        selected_draft = state.get("selected_draft")
        proposed_slot = state.get("proposed_slot")
        thread_id = state.get("thread_id") or "unknown"
    else:
        human_decision = state.human_decision
        selected_draft = state.selected_draft
        proposed_slot = state.proposed_slot
        thread_id = state.thread_id or "unknown"

    if human_decision not in ("approve", "edit") or selected_draft is None:
        raise ValueError("Cannot schedule post without an approved draft.")

    if isinstance(selected_draft, dict):
        selected_draft = CandidateDraft(**selected_draft)

    slot = proposed_slot or selected_draft.proposed_slot
    if slot is None:
        raise ValueError("Cannot schedule post without a valid proposed slot.")

    if isinstance(slot, dict):
        slot = ScheduledSlot(**slot)

    if queue_store is None:
        queue_store = QueueStore()

    slot_utc_str = slot.scheduled_at_utc.isoformat()
    slot_local_str = slot.scheduled_at_local.isoformat()

    queue_store.enqueue(
        thread_id=thread_id,
        draft_id=selected_draft.id,
        platform=selected_draft.platform,
        final_text=selected_draft.content,
        slot_utc=slot_utc_str,
        slot_local=slot_local_str,
        user_timezone=slot.user_timezone,
        status="queued",
    )

    return {"status": "queued"}


def record_node(
    state: PostwrightState | dict[str, Any],
    store: ApprovalHistoryStore | None = None,
) -> dict[str, Any]:
    """Store approved draft, diff, scores, timestamp, and thread_id in SQLite."""
    if isinstance(state, dict):
        human_decision = state.get("human_decision")
        selected_draft = state.get("selected_draft")
        thread_id = state.get("thread_id") or "unknown"
        edit_diff = state.get("edit_diff") or ""
    else:
        human_decision = state.human_decision
        selected_draft = state.selected_draft
        thread_id = state.thread_id or "unknown"
        edit_diff = state.edit_diff or ""

    if human_decision not in ("approve", "edit") or selected_draft is None:
        raise ValueError("Cannot record post without a valid resume approval/edit decision.")

    if store is None:
        store = ApprovalHistoryStore()

    if isinstance(selected_draft, dict):
        selected_draft = CandidateDraft(**selected_draft)

    original_draft = selected_draft.previous_best_content or selected_draft.content
    final_text = selected_draft.content
    diff_str = str(edit_diff) if isinstance(edit_diff, str) else str(edit_diff or "")

    store.record_approval(
        thread_id=thread_id,
        draft_id=selected_draft.id,
        original_draft=original_draft,
        final_text=final_text,
        unified_diff=diff_str,
        score=selected_draft.score,
    )

    return {"status": "recorded"}
