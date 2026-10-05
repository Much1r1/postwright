from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command

from postwright.graph import create_graph
from postwright.nodes import human_review_node
from postwright.scheduler import UniversalPriorPolicy, check_staleness
from postwright.state import (
    CandidateDraft,
    HumanReviewDecision,
    ScheduledSlot,
)
from postwright.store import QueueStore


class FakeLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return None

    @property
    def _llm_type(self):
        return "fake"

    def with_structured_output(self, schema, **kwargs):
        class StructuredFake:
            def __init__(self, schema):
                self.schema = schema

            def invoke(self, input_msgs):
                schema_name = getattr(self.schema, "__name__", str(self.schema))
                if "ExtractedIdeasOutput" in schema_name:
                    return {
                        "ideas": [
                            {
                                "id": "idea-1",
                                "summary": "Built a scheduling graph node in Postwright.",
                                "what_was_built": "UniversalPriorPolicy and queueing table",
                                "what_broke": "DST conversion edge case",
                                "what_was_learned": "zoneinfo handles DST seamlessly",
                                "numerical_result": "100% test coverage",
                            }
                        ]
                    }
                elif "CandidateDraftsOutput" in schema_name:
                    return {
                        "drafts": [
                            {
                                "id": "draft-1",
                                "idea_id": "idea-1",
                                "platform": "x",
                                "angle_format": "build_log",
                                "content": "Just built scheduling in Postwright! Deterministic queues are awesome.",
                                "is_thread": False,
                                "thread_parts": [],
                            }
                        ]
                    }
                elif "CriticOutput" in schema_name:
                    return {
                        "critiques": [
                            {
                                "draft_id": "draft-1",
                                "voice_match_score": 5,
                                "clarity_score": 5,
                                "specificity_score": 4,
                                "hook_strength_score": 4,
                                "critique": "Solid technical build log post.",
                            }
                        ]
                    }
                return {}

        return StructuredFake(schema)


def test_audience_timezone_conversion_and_dst(tmp_path):
    """Test audience timezone to Nairobi conversion across DST boundary."""
    config_file = tmp_path / "schedule.yaml"
    config_file.write_text(
        """
user_timezone: "Africa/Nairobi"
x:
  - day_of_week: "Monday"
    time: "14:00"
    audience_timezone: "America/New_York"
    source: "Test benchmark"
    retrieved_on: "2025-01-01"
"""
    )
    policy = UniversalPriorPolicy(config_path=config_file)

    # Standard time (Winter - NY is UTC-5, Nairobi is UTC+3 -> 14:00 NY = 19:00 UTC = 22:00 Nairobi)
    winter_now = datetime(2025, 1, 12, 10, 0, tzinfo=UTC)  # Sunday
    slot_winter = policy.suggest("x", now=winter_now)
    assert slot_winter.scheduled_at_utc == datetime(2025, 1, 13, 19, 0, tzinfo=UTC)
    assert slot_winter.scheduled_at_local.hour == 22

    # Daylight Saving Time (Summer - NY is UTC-4, Nairobi is UTC+3 -> 14:00 NY = 18:00 UTC = 21:00 Nairobi)
    summer_now = datetime(2025, 6, 8, 10, 0, tzinfo=UTC)  # Sunday
    slot_summer = policy.suggest("x", now=summer_now)
    assert slot_summer.scheduled_at_utc == datetime(2025, 6, 9, 18, 0, tzinfo=UTC)
    assert slot_summer.scheduled_at_local.hour == 21


def test_next_free_slot_and_no_double_booking(tmp_path):
    """Test picking next free slot without double-booking taken slots."""
    config_file = tmp_path / "schedule.yaml"
    config_file.write_text(
        """
user_timezone: "Africa/Nairobi"
x:
  - day_of_week: "Monday"
    time: "14:00"
    audience_timezone: "UTC"
  - day_of_week: "Wednesday"
    time: "15:00"
    audience_timezone: "UTC"
"""
    )
    policy = UniversalPriorPolicy(config_path=config_file)
    fixed_now = datetime(2025, 1, 12, 10, 0, tzinfo=UTC)  # Sunday

    slot1 = policy.suggest("x", now=fixed_now)
    assert slot1.scheduled_at_utc == datetime(2025, 1, 13, 14, 0, tzinfo=UTC)

    slot2 = policy.suggest("x", now=fixed_now, queued_slots=[slot1.scheduled_at_utc])
    assert slot2.scheduled_at_utc == datetime(2025, 1, 15, 15, 0, tzinfo=UTC)


def test_slot_rotation_across_consecutive_posts(tmp_path):
    """Test rotation through platform slot templates across consecutive posts."""
    config_file = tmp_path / "schedule.yaml"
    config_file.write_text(
        """
user_timezone: "Africa/Nairobi"
x:
  - day_of_week: "Monday"
    time: "14:00"
    audience_timezone: "UTC"
  - day_of_week: "Wednesday"
    time: "15:00"
    audience_timezone: "UTC"
"""
    )
    policy = UniversalPriorPolicy(config_path=config_file)
    fixed_now = datetime(2025, 1, 12, 10, 0, tzinfo=UTC)

    slot1 = policy.suggest("x", now=fixed_now)
    slot2 = policy.suggest("x", now=fixed_now, queued_slots=[slot1])

    assert slot1.scheduled_at_utc == datetime(2025, 1, 13, 14, 0, tzinfo=UTC)
    assert slot2.scheduled_at_utc == datetime(2025, 1, 15, 15, 0, tzinfo=UTC)


def test_override_validation_past_and_taken_rejected(tmp_path):
    """Test that slot overrides in the past or already taken are rejected."""
    db_path = tmp_path / "test_queue.db"
    q_store = QueueStore(db_path=db_path)

    fixed_now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)

    taken_dt = datetime(2025, 1, 20, 14, 0, tzinfo=UTC)
    q_store.enqueue(
        thread_id="t-1",
        draft_id="d-1",
        platform="x",
        final_text="taken",
        slot_utc=taken_dt.isoformat(),
        slot_local=taken_dt.isoformat(),
    )

    draft = CandidateDraft(
        id="d-2",
        idea_id="i-1",
        platform="x",
        angle_format="build_log",
        content="new post",
        proposed_slot=ScheduledSlot(
            platform="x",
            scheduled_at_utc=datetime(2025, 1, 22, 14, 0, tzinfo=UTC),
            scheduled_at_local=datetime(2025, 1, 22, 17, 0, tzinfo=UTC),
        ),
    )

    past_decision = HumanReviewDecision(
        decision="approve",
        draft_id="d-2",
        slot_override="2025-01-10T14:00:00+00:00",
    )
    with patch("postwright.nodes.interrupt", return_value=past_decision), pytest.raises(
        ValueError, match="in the past"
    ):
        human_review_node({"candidate_drafts": [draft]}, queue_store=q_store, now=fixed_now)

    taken_decision = HumanReviewDecision(
        decision="approve",
        draft_id="d-2",
        slot_override=taken_dt.isoformat(),
    )
    with patch("postwright.nodes.interrupt", return_value=taken_decision), pytest.raises(
        ValueError, match="already taken"
    ):
        human_review_node({"candidate_drafts": [draft]}, queue_store=q_store, now=fixed_now)


def test_nothing_queued_without_approval(tmp_path):
    """Test that rejection or rewrite decisions do not enqueue anything."""
    db_path = tmp_path / "test_store.db"
    q_store = QueueStore(db_path=db_path)
    llm = FakeLLM()

    graph = create_graph(
        llm=llm,
        queue_store=q_store,
        checkpointer=True,
    )

    tid = "test-no-queue"
    config = {"configurable": {"thread_id": tid}}

    initial_input = {"raw_note": {"content": "Building stuff", "source": "cli"}}
    graph.invoke(initial_input, config=config)

    graph.invoke(Command(resume={"decision": "reject"}), config=config)

    assert len(q_store.get_queued_posts()) == 0


def test_staleness_warning_triggers(tmp_path):
    """Test that staleness warning triggers when prior retrieved_on > threshold days old."""
    config_file = tmp_path / "schedule.yaml"
    config_file.write_text(
        """
user_timezone: "Africa/Nairobi"
x:
  - day_of_week: "Monday"
    time: "14:00"
    audience_timezone: "UTC"
    retrieved_on: "2024-01-01"
"""
    )
    fixed_now = datetime(2025, 1, 1, tzinfo=UTC)
    warnings = check_staleness(config_path=config_file, days_threshold=90, now=fixed_now)

    assert len(warnings) == 1
    assert "is 366 days old" in warnings[0]
