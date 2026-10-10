import json
import logging
import math
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from postwright.config import get_settings
from postwright.llm import get_llm, invoke_with_resilience
from postwright.nodes import (
    capture_node,
    critic_node,
    draft_node,
    extract_ideas_node,
    pick_angles_node,
)
from postwright.platforms import get_adapter
from postwright.state import PostwrightState

logger = logging.getLogger("postwright.eval")

PROGRESS_FILE = Path("evals/.eval_progress.json")
CASES_FILE = Path("evals/cases.jsonl")
REPORTS_DIR = Path("evals/reports")


class JudgeEvaluationOutput(BaseModel):
    voice_match_score: int = Field(ge=1, le=5, description="Voice alignment (1-5)")
    clarity_score: int = Field(ge=1, le=5, description="Clarity & structure (1-5)")
    specificity_score: int = Field(ge=1, le=5, description="Concrete detail & numbers (1-5)")
    hook_strength_score: int = Field(ge=1, le=5, description="Hook & engagement (1-5)")
    critique_notes: str = Field(description="Detailed critique notes from independent judge")


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def load_eval_cases(cases_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = Path(cases_path) if cases_path else CASES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Eval cases file not found at {path}")

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                cases.append(json.loads(line_str))
    return cases


def check_specificity(note: str, draft_text: str) -> bool:
    """Check if the draft contains a concrete detail or number from the note."""
    note_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", note)
    if note_numbers:
        for num in note_numbers:
            if num in draft_text:
                return True
    if re.search(r"\d+", draft_text):
        return True
    note_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", note)}
    draft_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", draft_text)}
    overlap = note_words.intersection(draft_words)
    return len(overlap) >= 1


def run_eval_pipeline_on_case(
    case: dict[str, Any],
    llm: Any | None = None,
    threshold: int | None = None,
    max_revisions: int | None = None,
) -> dict[str, Any]:
    """Execute draft + critic pipeline loop on a single case in isolation."""
    settings = get_settings()
    eff_threshold = threshold if threshold is not None else settings.critic_score_threshold
    eff_max_revisions = max_revisions if max_revisions is not None else settings.max_revisions

    case_id = case["id"]
    note_text = case["note"]
    platform = case["platform"]

    state = PostwrightState(
        thread_id=f"eval-{case_id}",
        raw_note={"content": note_text, "source": "eval"},
        critic_score_threshold=eff_threshold,
        max_revisions=eff_max_revisions,
    )

    t0 = time.time()

    # 1. Capture
    cap_out = capture_node(state)
    state.raw_note = cap_out["raw_note"]

    # 2. Extract ideas
    ext_out = extract_ideas_node(state, llm=llm)
    state.extracted_ideas = ext_out.get("extracted_ideas", [])
    state.total_llm_calls += ext_out.get("total_llm_calls", 0)
    state.total_input_tokens += ext_out.get("total_input_tokens", 0)
    state.total_output_tokens += ext_out.get("total_output_tokens", 0)
    state.total_cost += ext_out.get("total_cost", 0.0)
    state.llm_retries += ext_out.get("llm_retries", 0)
    state.llm_wait_time_seconds += ext_out.get("llm_wait_time_seconds", 0.0)

    # 3. Pick angles
    pa_out = pick_angles_node(state)
    state.angled_ideas = [
        a for a in pa_out.get("angled_ideas", []) if a.target_platform == platform
    ]
    if not state.angled_ideas and pa_out.get("angled_ideas"):
        state.angled_ideas = pa_out.get("angled_ideas", [])[:1]

    # 4. Draft & Critic Loop
    while True:
        # Draft step
        draft_out = draft_node(state, llm=llm)
        state.candidate_drafts = draft_out.get("candidate_drafts", [])
        state.total_llm_calls = draft_out.get("total_llm_calls", state.total_llm_calls)
        state.total_input_tokens = draft_out.get("total_input_tokens", state.total_input_tokens)
        state.total_output_tokens = draft_out.get("total_output_tokens", state.total_output_tokens)
        state.total_cost = draft_out.get("total_cost", state.total_cost)
        state.llm_retries = draft_out.get("llm_retries", state.llm_retries)
        state.llm_wait_time_seconds = draft_out.get("llm_wait_time_seconds", state.llm_wait_time_seconds)

        # Critic step
        critic_out = critic_node(state, llm=llm)
        state.candidate_drafts = critic_out.get("candidate_drafts", [])
        state.critiques = critic_out.get("critiques", state.critiques)
        state.total_llm_calls = critic_out.get("total_llm_calls", state.total_llm_calls)
        state.total_input_tokens = critic_out.get("total_input_tokens", state.total_input_tokens)
        state.total_output_tokens = critic_out.get("total_output_tokens", state.total_output_tokens)
        state.total_cost = critic_out.get("total_cost", state.total_cost)
        state.llm_retries = critic_out.get("llm_retries", state.llm_retries)
        state.llm_wait_time_seconds = critic_out.get("llm_wait_time_seconds", state.llm_wait_time_seconds)
        state.below_threshold = critic_out.get("below_threshold", False)

        has_failing = any(
            d.score is None or d.score < eff_threshold
            for d in state.candidate_drafts
        )

        if not has_failing or state.revision_count >= eff_max_revisions:
            break

    latency = round(time.time() - t0, 3)

    # Select best candidate draft for platform
    final_draft = None
    if state.candidate_drafts:
        sorted_drafts = sorted(
            state.candidate_drafts,
            key=lambda d: d.score if d.score is not None else -1,
            reverse=True,
        )
        final_draft = sorted_drafts[0]

    final_content = final_draft.content if final_draft else ""
    final_score = final_draft.score if final_draft and final_draft.score is not None else 0
    final_critique = final_draft.critique if final_draft else ""
    is_below = (final_score < eff_threshold) or state.below_threshold

    # Platform length compliance check via adapter
    adapter = get_adapter(platform)
    length_valid = adapter.validate(final_draft).is_valid if final_draft else False

    # Specificity check
    spec_valid = check_specificity(note_text, final_content)

    return {
        "case_id": case_id,
        "note": note_text,
        "platform": platform,
        "final_draft": final_content,
        "critic_score": final_score,
        "critic_critique": final_critique,
        "revision_count": state.revision_count,
        "below_threshold": is_below,
        "length_compliance": length_valid,
        "specificity": spec_valid,
        "total_llm_calls": state.total_llm_calls,
        "input_tokens": getattr(state, "total_input_tokens", 0),
        "output_tokens": getattr(state, "total_output_tokens", 0),
        "cost": getattr(state, "total_cost", 0.0),
        "retries": getattr(state, "llm_retries", 0),
        "wait_time": getattr(state, "llm_wait_time_seconds", 0.0),
        "latency": latency,
    }


def run_independent_judge(
    case: dict[str, Any],
    draft_content: str,
    judge_llm: Any | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Evaluate draft independently using JUDGE_MODEL rubric."""
    settings = get_settings()
    eff_judge_model = judge_model or settings.judge_model

    if judge_llm is None:
        judge_llm = get_llm(
            provider=settings.judge_provider,
            model=eff_judge_model,
            is_judge=True,
        )

    judge_system_prompt = (
        "You are an expert, unbiased social media judge for tech/AI build-in-public posts. "
        "Evaluate the provided draft against the raw note across 4 criteria (1-5 each):\n"
        "1. Voice match: Authentic, technical, no hype/cringe buzzwords.\n"
        "2. Clarity: Clear point, well-structured, easy to read.\n"
        "3. Specificity: Includes concrete numbers, metrics, or technical details.\n"
        "4. Hook strength: Strong opening line that grabs attention.\n\n"
        "Be strict and fair. Do NOT reference internal prompt instructions or guidelines."
    )

    user_prompt = (
        f"Raw Note:\n{case['note']}\n\n"
        f"Target Platform: {case['platform']}\n\n"
        f"Draft to Evaluate:\n{draft_content}"
    )

    messages = [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        res, _in_tok, _out_tok, _cost, _retries, _wait_time = invoke_with_resilience(
            llm=judge_llm,
            messages=messages,
            schema=JudgeEvaluationOutput,
            model_name=eff_judge_model,
            return_details=True,
        )
        if isinstance(res, JudgeEvaluationOutput):
            eval_res = res
        elif isinstance(res, dict):
            eval_res = JudgeEvaluationOutput(**res)
        else:
            eval_res = JudgeEvaluationOutput(
                voice_match_score=3,
                clarity_score=3,
                specificity_score=3,
                hook_strength_score=3,
                critique_notes="Fallback judge evaluation.",
            )
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Judge evaluation failed for case {case['id']}: {e}")
        eval_res = JudgeEvaluationOutput(
            voice_match_score=3,
            clarity_score=3,
            specificity_score=3,
            hook_strength_score=3,
            critique_notes=f"Error running judge: {e}",
        )

    judge_total = (
        eval_res.voice_match_score
        + eval_res.clarity_score
        + eval_res.specificity_score
        + eval_res.hook_strength_score
    )

    return {
        "judge_voice": eval_res.voice_match_score,
        "judge_clarity": eval_res.clarity_score,
        "judge_specificity": eval_res.specificity_score,
        "judge_hook": eval_res.hook_strength_score,
        "judge_critique": eval_res.critique_notes,
        "judge_total": judge_total,
    }


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient handling zero variance or small sample size."""
    n = len(x)
    if n <= 1:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x <= 1e-9 or var_y <= 1e-9:
        return 0.0

    r = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return round(r, 4)


def compute_percentile(data: list[float], p: float) -> float:
    """Compute percentile value (0..100) from list of numbers."""
    if not data:
        return 0.0
    s_data = sorted(data)
    n = len(s_data)
    if n == 1:
        return s_data[0]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s_data[int(k)]
    d0 = s_data[int(f)] * (c - k)
    d1 = s_data[int(c)] * (k - f)
    return round(d0 + d1, 3)


def calculate_aggregates(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metrics from per-case eval results."""
    n = len(case_results)
    if n == 0:
        return {}

    critic_scores = [float(r["critic_score"]) for r in case_results]
    judge_scores = [float(r["judge_total"]) for r in case_results]
    below_thresh_count = sum(1 for r in case_results if r["below_threshold"])
    length_valid_count = sum(1 for r in case_results if r["length_compliance"])
    spec_valid_count = sum(1 for r in case_results if r["specificity"])
    revisions = [r["revision_count"] for r in case_results]
    costs = [r.get("cost", 0.0) for r in case_results]
    latencies = [r["latency"] for r in case_results]

    mean_critic = sum(critic_scores) / n
    mean_judge = sum(judge_scores) / n
    mean_bias = mean_critic - mean_judge  # Positive means critic scores higher than judge

    corr = compute_pearson_correlation(critic_scores, judge_scores)

    settings = get_settings()
    current_threshold = settings.critic_score_threshold
    suggested_threshold = current_threshold - round(mean_bias)

    return {
        "case_count": n,
        "mean_critic_score": round(mean_critic, 2),
        "mean_judge_score": round(mean_judge, 2),
        "below_threshold_rate": round((below_thresh_count / n) * 100, 1),
        "length_compliance_rate": round((length_valid_count / n) * 100, 1),
        "specificity_rate": round((spec_valid_count / n) * 100, 1),
        "avg_revisions": round(sum(revisions) / n, 2),
        "avg_cost": round(sum(costs) / n, 6),
        "p50_latency": compute_percentile(latencies, 50),
        "p95_latency": compute_percentile(latencies, 95),
        "mean_bias": round(mean_bias, 2),
        "correlation": corr,
        "suggested_threshold": suggested_threshold,
    }


def generate_markdown_report(
    case_results: list[dict[str, Any]],
    aggregates: dict[str, Any],
    config_info: dict[str, Any],
) -> str:
    """Format markdown evaluation report with config, same-model warning, table, and worst cases."""
    pipeline_model = config_info.get("pipeline_model", "")
    judge_model = config_info.get("judge_model", "")
    same_model = (
        pipeline_model.strip().lower() == judge_model.strip().lower()
        and bool(pipeline_model)
    )

    lines = []
    lines.append("# Postwright Evaluation Report")
    lines.append(f"**Generated at:** {config_info.get('timestamp', '')}")
    lines.append(f"**Git Commit:** `{config_info.get('git_commit', '')}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append(f"- **Pipeline Provider/Model:** `{config_info.get('pipeline_provider')}` / `{pipeline_model}`")
    lines.append(f"- **Judge Provider/Model:** `{config_info.get('judge_provider')}` / `{judge_model}`")
    lines.append(f"- **Critic Score Threshold:** `{config_info.get('critic_threshold')}`")
    lines.append(f"- **Max Revisions:** `{config_info.get('max_revisions')}`")
    lines.append("")

    if same_model:
        lines.append("> ⚠️ **WARNING:** Pipeline model and judge model are the same (`" + pipeline_model + "`). LLM-as-judge self-evaluation may exhibit self-enhancement bias.")
        lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append(f"- **Total Cases:** {aggregates.get('case_count', 0)}")
    lines.append(f"- **Mean Critic Total:** {aggregates.get('mean_critic_score', 0)} / 20")
    lines.append(f"- **Mean Judge Total:** {aggregates.get('mean_judge_score', 0)} / 20")
    lines.append(f"- **Mean Bias (Critic - Judge):** {aggregates.get('mean_bias', 0):+0.2f}")
    lines.append(f"- **Critic-Judge Correlation:** {aggregates.get('correlation', 0)}")
    lines.append(f"- **Below Threshold Rate:** {aggregates.get('below_threshold_rate', 0)}%")
    lines.append(f"- **Length Compliance Rate:** {aggregates.get('length_compliance_rate', 0)}%")
    lines.append(f"- **Specificity Rate:** {aggregates.get('specificity_rate', 0)}%")
    lines.append(f"- **Avg Revisions per Post:** {aggregates.get('avg_revisions', 0)}")
    lines.append(f"- **Avg Cost per Post:** ${aggregates.get('avg_cost', 0):.6f}")
    lines.append(f"- **p50 Latency:** {aggregates.get('p50_latency', 0)}s")
    lines.append(f"- **p95 Latency:** {aggregates.get('p95_latency', 0)}s")
    lines.append(f"- **Suggested Threshold Adjustment:** Current is {config_info.get('critic_threshold')}, suggested is {aggregates.get('suggested_threshold', config_info.get('critic_threshold'))} (suggestion only, config unchanged)")
    lines.append("")

    lines.append("## Per-Case Results")
    lines.append("| Case ID | Platform | Revisions | Critic Total | Judge Total | Below Thresh? | Length Valid? | Specificity? | Retries | Wait Time (s) | Cost ($) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for r in case_results:
        c_id = r["case_id"]
        plat = r["platform"].upper()
        revs = r["revision_count"]
        c_tot = r["critic_score"]
        j_tot = r["judge_total"]
        below = "Yes" if r["below_threshold"] else "No"
        len_val = "Yes" if r["length_compliance"] else "No"
        spec = "Yes" if r["specificity"] else "No"
        retries = r.get("retries", 0)
        wait_time = round(r.get("wait_time", 0.0), 2)
        cost = f"${r.get('cost', 0.0):.6f}"

        lines.append(
            f"| {c_id} | {plat} | {revs} | {c_tot}/20 | {j_tot}/20 | {below} | {len_val} | {spec} | {retries} | {wait_time}s | {cost} |"
        )

    lines.append("")

    # Worst 3 cases by critic score
    worst_3 = sorted(case_results, key=lambda r: (r["critic_score"], r["judge_total"]))[:3]
    lines.append("## Worst 3 Cases")
    for idx, w in enumerate(worst_3, start=1):
        lines.append(f"### {idx}. Case `{w['case_id']}` ({w['platform'].upper()})")
        lines.append(f"**Raw Note:** {w['note']}")
        lines.append(f"**Generated Draft:**\n```\n{w['final_draft']}\n```")
        lines.append(f"**In-Pipeline Critique (Score {w['critic_score']}/20):** {w['critic_critique']}")
        lines.append(f"**Judge Critique (Score {w['judge_total']}/20):** {w['judge_critique']}")
        lines.append("")

    return "\n".join(lines)


def run_eval_suite(
    cases_path: Path | str | None = None,
    limit: int | None = None,
    case_id_filter: str | None = None,
    resume: bool = False,
    compare_report_path: Path | str | None = None,
    llm: Any | None = None,
    judge_llm: Any | None = None,
    console_printer: Any | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    """Run full evaluation suite and generate report."""
    settings = get_settings()
    cases = load_eval_cases(cases_path)

    if case_id_filter:
        cases = [c for c in cases if c["id"] == case_id_filter]
    if limit is not None and limit > 0:
        cases = cases[:limit]

    completed_results: list[dict[str, Any]] = []

    if resume and PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                completed_results = json.load(f)
            completed_ids = {r["case_id"] for r in completed_results}
            cases = [c for c in cases if c["id"] not in completed_ids]
            if console_printer:
                console_printer(f"[yellow]Resuming run. Found {len(completed_results)} already evaluated cases.[/]")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load resume progress: {e}")

    total_cases = len(completed_results) + len(cases)

    for i, case in enumerate(cases, start=len(completed_results) + 1):
        if console_printer:
            console_printer(f"[blue]Evaluating Case {i}/{total_cases} ({case['id']})...[/]")

        # 1. Run pipeline
        res = run_eval_pipeline_on_case(case, llm=llm)

        # 2. Run judge
        j_res = run_independent_judge(case, res["final_draft"], judge_llm=judge_llm)
        res.update(j_res)

        completed_results.append(res)

        # Save progress incrementally
        try:
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(completed_results, f, indent=2)
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save eval progress: {e}")

    # Remove progress file upon completion
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink(missing_ok=True)

    aggregates = calculate_aggregates(completed_results)

    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()

    config_info = {
        "timestamp": timestamp_str,
        "git_commit": git_commit,
        "pipeline_provider": settings.llm_provider,
        "pipeline_model": settings.llm_model,
        "judge_provider": settings.judge_provider,
        "judge_model": settings.judge_model,
        "critic_threshold": settings.critic_score_threshold,
        "max_revisions": settings.max_revisions,
    }

    report_md = generate_markdown_report(completed_results, aggregates, config_info)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"{timestamp_str}.md"
    report_file.write_text(report_md, encoding="utf-8")

    # If compare report path provided, compute and return deltas
    compare_deltas = None
    if compare_report_path:
        compare_path = Path(compare_report_path)
        if compare_path.exists():
            compare_deltas = format_report_comparison(aggregates, compare_path.read_text(encoding="utf-8"))

    return report_md, report_file, {"aggregates": aggregates, "deltas": compare_deltas}


def format_report_comparison(current_aggs: dict[str, Any], earlier_report_md: str) -> str:
    """Parse earlier report markdown and calculate delta metrics."""
    earlier_critic = None
    earlier_judge = None
    earlier_below = None

    m_critic = re.search(r"Mean Critic Total:\*\* ([\d.]+)", earlier_report_md)
    if m_critic:
        earlier_critic = float(m_critic.group(1))

    m_judge = re.search(r"Mean Judge Total:\*\* ([\d.]+)", earlier_report_md)
    if m_judge:
        earlier_judge = float(m_judge.group(1))

    m_below = re.search(r"Below Threshold Rate:\*\* ([\d.]+)", earlier_report_md)
    if m_below:
        earlier_below = float(m_below.group(1))

    lines = ["\n### Delta Comparison against Earlier Report"]
    if earlier_critic is not None:
        diff_c = current_aggs.get("mean_critic_score", 0) - earlier_critic
        lines.append(f"- **Mean Critic Total:** {current_aggs.get('mean_critic_score', 0)} ({diff_c:+0.2f})")
    if earlier_judge is not None:
        diff_j = current_aggs.get("mean_judge_score", 0) - earlier_judge
        lines.append(f"- **Mean Judge Total:** {current_aggs.get('mean_judge_score', 0)} ({diff_j:+0.2f})")
    if earlier_below is not None:
        diff_b = current_aggs.get("below_threshold_rate", 0) - earlier_below
        lines.append(f"- **Below Threshold Rate:** {current_aggs.get('below_threshold_rate', 0)}% ({diff_b:+0.1f}%)")

    return "\n".join(lines)
