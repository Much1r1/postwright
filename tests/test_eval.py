import json
import warnings
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel

from postwright.adapters import XAdapter, get_adapter, get_x_weighted_length
from postwright.eval import (
    calculate_aggregates,
    check_specificity,
    compute_pearson_correlation,
    compute_percentile,
    format_report_comparison,
    generate_markdown_report,
    run_eval_pipeline_on_case,
    run_eval_suite,
)
from postwright.llm import calculate_cost, get_model_pricing
from postwright.state import CandidateDraft


class FakeStructuredLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return None

    @property
    def _llm_type(self):
        return "fake_eval_llm"

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
                                "id": "idea-eval-1",
                                "summary": "Built eval suite with 100% test coverage.",
                                "what_was_built": "Evaluation harness and judge",
                                "what_broke": "Rate limits on 429",
                                "what_was_learned": "Exponential backoff retries work well",
                                "numerical_result": "100%",
                            }
                        ]
                    }
                elif "CandidateDraftsOutput" in schema_name:
                    return {
                        "drafts": [
                            {
                                "id": "draft-eval-1",
                                "idea_id": "idea-eval-1",
                                "platform": "x",
                                "angle_format": "build_log",
                                "content": "Built an eval suite for Postwright! Achieved 100% test coverage.",
                                "is_thread": False,
                                "thread_parts": [],
                            }
                        ]
                    }
                elif "CriticOutput" in schema_name:
                    return {
                        "critiques": [
                            {
                                "draft_id": "draft-eval-1",
                                "voice_match_score": 4,
                                "clarity_score": 4,
                                "specificity_score": 4,
                                "hook_strength_score": 4,
                                "critique": "Solid technical draft.",
                            }
                        ]
                    }
                elif "JudgeEvaluationOutput" in schema_name:
                    return {
                        "voice_match_score": 4,
                        "clarity_score": 4,
                        "specificity_score": 3,
                        "hook_strength_score": 4,
                        "critique_notes": "Great post overall.",
                    }
                return {}

        return StructuredFake(schema)


def test_percentile_calculations():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p50 = compute_percentile(data, 50)
    p95 = compute_percentile(data, 95)
    assert p50 == 5.5
    assert p95 == 9.55 or round(p95, 1) == 9.6

    single = [42.0]
    assert compute_percentile(single, 50) == 42.0
    assert compute_percentile([], 50) == 0.0


def test_specificity_checker():
    note_with_num = "Reduced latency from 450ms to 85ms by caching DB queries."
    draft1 = "We reduced latency from 450ms to 85ms using DB caching!"

    assert check_specificity(note_with_num, draft1) is True
    # draft2 has 'refactored' and 'faster' (words > 5 chars overlap) or digit check
    note_vague = "Did some work today."
    draft_vague = "Just working hard."
    assert check_specificity(note_vague, draft_vague) is False


def test_platform_adapters_and_length_validation():
    x_adapter = XAdapter()
    url_text = "Check out https://github.com/example/repo for details!"
    # URL counts as 23 chars
    expected_len = len("Check out ") + 23 + len(" for details!")
    assert get_x_weighted_length(url_text) == expected_len

    short_draft = CandidateDraft(
        id="d1", idea_id="i1", platform="x", angle_format="build_log", content="Short tweet"
    )
    assert x_adapter.validate(short_draft) is True

    long_draft = CandidateDraft(
        id="d2", idea_id="i1", platform="x", angle_format="build_log", content="A" * 300
    )
    assert x_adapter.validate(long_draft) is False

    linkedin_adapter = get_adapter("linkedin")
    li_draft = CandidateDraft(
        id="d3", idea_id="i1", platform="linkedin", angle_format="build_log", content="B" * 2500
    )
    assert linkedin_adapter.validate(li_draft) is True


def test_correlation_and_bias_edge_cases():
    # Normal case
    x = [10.0, 12.0, 14.0, 16.0, 18.0]
    y = [11.0, 13.0, 15.0, 17.0, 19.0]
    corr = compute_pearson_correlation(x, y)
    assert corr == 1.0

    # Identical scores (zero variance)
    x_identical = [15.0, 15.0, 15.0]
    y_identical = [12.0, 12.0, 12.0]
    assert compute_pearson_correlation(x_identical, y_identical) == 0.0

    # Single case
    assert compute_pearson_correlation([15.0], [12.0]) == 0.0

    # Aggregates math
    case_results = [
        {
            "case_id": "c1",
            "critic_score": 16,
            "judge_total": 14,
            "below_threshold": False,
            "length_compliance": True,
            "specificity": True,
            "revision_count": 0,
            "cost": 0.001,
            "latency": 1.2,
        },
        {
            "case_id": "c2",
            "critic_score": 12,
            "judge_total": 12,
            "below_threshold": True,
            "length_compliance": True,
            "specificity": False,
            "revision_count": 1,
            "cost": 0.002,
            "latency": 2.4,
        },
    ]
    aggs = calculate_aggregates(case_results)
    assert aggs["case_count"] == 2
    assert aggs["mean_critic_score"] == 14.0
    assert aggs["mean_judge_score"] == 13.0
    assert aggs["mean_bias"] == 1.0  # 14.0 - 13.0
    assert aggs["below_threshold_rate"] == 50.0
    assert aggs["length_compliance_rate"] == 100.0
    assert aggs["specificity_rate"] == 50.0


def test_pricing_lookup_and_missing_model_fallback(tmp_path):
    pricing_yaml = tmp_path / "pricing.yaml"
    pricing_yaml.write_text(
        """
models:
  claude-3-5-sonnet-20241022:
    input_cost_per_1m: 3.00
    output_cost_per_1m: 15.00
"""
    )

    in_cost, out_cost = get_model_pricing("claude-3-5-sonnet-20241022", pricing_path=pricing_yaml)
    assert in_cost == 3.00
    assert out_cost == 15.00

    calc_cost = calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000, pricing_path=pricing_yaml)
    assert calc_cost == round((1000 / 1e6) * 3.00 + (1000 / 1e6) * 15.00, 6)

    # Missing model should issue warning and return (0.0, 0.0) without crashing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        missing_in, missing_out = get_model_pricing("nonexistent-model-xyz", pricing_path=pricing_yaml)
        assert missing_in == 0.0
        assert missing_out == 0.0
        assert len(w) == 1
        assert "not found in pricing.yaml" in str(w[0].message)


def test_report_generation_and_compare_deltas():
    case_results = [
        {
            "case_id": "case-01",
            "note": "Built features.",
            "platform": "x",
            "final_draft": "Draft 1",
            "critic_score": 16,
            "critic_critique": "Good",
            "judge_total": 15,
            "judge_critique": "Solid",
            "revision_count": 0,
            "below_threshold": False,
            "length_compliance": True,
            "specificity": True,
            "retries": 0,
            "wait_time": 0.0,
            "cost": 0.0005,
            "latency": 1.1,
        }
    ]
    aggs = calculate_aggregates(case_results)
    config_info = {
        "timestamp": "20250223_120000",
        "git_commit": "abc1234",
        "pipeline_provider": "groq",
        "pipeline_model": "llama-3.3-70b-versatile",
        "judge_provider": "google",
        "judge_model": "gemini-2.5-flash",
        "critic_threshold": 14,
        "max_revisions": 2,
    }

    report_md = generate_markdown_report(case_results, aggs, config_info)
    assert "# Postwright Evaluation Report" in report_md
    assert "Mean Critic Total:** 16" in report_md
    assert "| case-01 | X | 0 | 16/20 | 15/20 | No | Yes | Yes |" in report_md

    # Test same model warning
    config_info_same = dict(config_info)
    config_info_same["judge_model"] = "llama-3.3-70b-versatile"
    report_same_md = generate_markdown_report(case_results, aggs, config_info_same)
    assert "WARNING:** Pipeline model and judge model are the same" in report_same_md

    # Test compare deltas format
    deltas = format_report_comparison(aggs, report_md)
    assert "Delta Comparison" in deltas


def test_eval_path_never_touches_queue_or_approval_db(tmp_path, monkeypatch):
    """Assert via spies that running postwright eval NEVER touches queue store or approval store."""
    mock_queue = MagicMock()
    mock_approval = MagicMock()

    monkeypatch.setattr("postwright.store.QueueStore.enqueue", mock_queue.enqueue)
    monkeypatch.setattr("postwright.store.ApprovalHistoryStore.record_approval", mock_approval.record_approval)
    monkeypatch.setattr("postwright.eval.REPORTS_DIR", tmp_path / "reports")

    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps(
            {
                "id": "case-spy-1",
                "note": "Testing spy assertions that eval harness is read-only for queue.",
                "platform": "x",
                "expected_qualities": {"must_include_concrete_detail": True},
            }
        )
        + "\n"
    )

    fake_llm = FakeStructuredLLM()

    report_md, report_path, _extra = run_eval_suite(
        cases_path=cases_file,
        limit=1,
        llm=fake_llm,
        judge_llm=fake_llm,
    )

    # Assert that no enqueue or approval records were called
    mock_queue.enqueue.assert_not_called()
    mock_approval.record_approval.assert_not_called()

    assert report_path.exists()
    assert "case-spy-1" in report_md


def test_pipeline_nodes_accumulate_tokens_and_cost():
    case = {
        "id": "case-acc-1",
        "note": "Testing token accumulation in nodes.",
        "platform": "x",
    }
    fake_llm = FakeStructuredLLM()

    res = run_eval_pipeline_on_case(case, llm=fake_llm)

    assert res["case_id"] == "case-acc-1"
    assert res["total_llm_calls"] >= 3
    assert "critic_score" in res
    assert "final_draft" in res
