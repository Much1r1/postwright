# postwright

A human-in-the-loop LangGraph agent that turns a creator's rough build notes into ready-to-post social media drafts in their own voice.

## Overview

`postwright` is built for AI engineers and creators building in public. It processes rough notes, extracts core technical ideas, selects formats/angles, drafts candidate posts, evaluates quality using a critic loop, proposes an optimal posting schedule (converting audience timezones to `Africa/Nairobi`), and waits for explicit human approval via a LangGraph interrupt before scheduling or publishing.

## Features & Hard Rules

- **Python 3.11+ Managed with `uv`**: Modern, fast Python tooling.
- **Human-in-the-Loop Approval**: Every post must pass a LangGraph `interrupt` for approval. Nothing is published autonomously.
- **Dry-Run by Default**: Publishing to live platforms requires `POSTWRIGHT_LIVE=1` AND an approved draft.
- **Idiomatic LangGraph**: Stateful graph architecture with SQLite checkpointing and human-in-the-loop interrupts.
- **Configurable LLM Provider**: Groq by default (`LLM_PROVIDER=groq`), configurable via environment variables (Anthropic, Gemini, OpenAI).

## Quickstart

```bash
# Clone and install dependencies
uv sync

# Configure environment variables
cp .env.example .env

# Run pipeline on a build note
uv run postwright run --note "Fixed a race condition in SqliteSaver checkpointer."

# Review and approve/edit interrupted run
uv run postwright review

# View approved post history and cost tracking
uv run postwright history
```

## Running Tests & Checks

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

## Evals & Cost Tracking

`postwright` includes an offline evaluation harness (`evals/cases.jsonl`) and an independent LLM-as-judge system.

### Running Evaluations

```bash
# Run full evaluation suite
uv run postwright eval

# Run with limit or on a single case
uv run postwright eval --limit 5
uv run postwright eval --case case-01

# Resume an interrupted run (e.g. after a rate-limit 429 storm)
uv run postwright eval --resume

# Compare results against an earlier report
uv run postwright eval --compare evals/reports/20250223_120000.md
```

### Metrics Definitions

- **Mean Critic Total**: Average score (out of 20) assigned by the in-pipeline critic across all test cases.
- **Mean Judge Total**: Average score (out of 20) assigned by the independent LLM judge (`JUDGE_PROVIDER` / `JUDGE_MODEL`).
- **Mean Bias**: Difference between critic total and judge total (`mean_critic - mean_judge`). A positive bias indicates the in-pipeline critic scores higher than the judge.
- **Critic-Judge Correlation**: Pearson correlation coefficient between in-pipeline critic scores and independent judge scores.
- **Below Threshold Rate**: Percentage of cases where the candidate draft failed to reach the configured critic score threshold (`CRITIC_SCORE_THRESHOLD`).
- **Length Compliance Rate**: Percentage of drafts adhering to platform character limits (URL-weighted 280 chars for X, 3000 chars for LinkedIn).
- **Specificity Rate**: Percentage of drafts containing concrete details or numbers extracted from the raw build note.
- **Avg Revisions per Post**: Average revision iterations required before passing the critic threshold.
- **Avg Cost per Post**: Average API cost per post based on `config/pricing.yaml`.
- **p50 / p95 Latency**: 50th and 95th percentile pipeline execution time per case in seconds.
- **Suggested Threshold Adjustment**: Recommended adjustment to `CRITIC_SCORE_THRESHOLD` to align in-pipeline evaluation with independent judge standards (suggestion only; configuration is never modified automatically).

### Baseline Note & Limits of LLM-as-Judge

> **Note:** The first committed evaluation report in `evals/reports/` serves as an experimental baseline, not a formal quality claim.

**Limits of LLM-as-Judge:**
- **Self-Enhancement Bias:** If the pipeline model and judge model share the same underlying LLM, scores may be inflated due to self-preference bias. A warning is automatically emitted in reports when `LLM_MODEL` equals `JUDGE_MODEL`.
- **Non-Determinism:** LLM judge outputs can vary slightly across runs.
- **Domain Specificity:** The judge evaluates structural, voice, and specificity metrics, but cannot independently verify code correctness.

## Observability & Tracing

LangSmith tracing is disabled by default and enabled strictly via environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="postwright"
```

Runs are tagged with `thread_id`, `platform`, and the current `git_commit` hash. API keys are strictly masked and will never appear in console logs or generated reports.
