# postwright

A human-in-the-loop LangGraph agent that turns a creator's rough build notes into ready-to-post social media drafts in their own voice.

## Overview

`postwright` is built for AI engineers and creators building in public. It processes rough notes, extracts core technical ideas, selects formats/angles, drafts candidate posts, evaluates quality using a critic loop, proposes an optimal posting schedule (converting audience timezones to `Africa/Nairobi`), and waits for explicit human approval via a LangGraph interrupt before scheduling or publishing.

## Features & Hard Rules

- **Python 3.11+ Managed with `uv`**: Modern, fast Python tooling.
- **Human-in-the-Loop Approval**: Every post must pass a LangGraph `interrupt` for approval. Nothing is published autonomously.
- **Dry-Run by Default**: Publishing to live platforms requires `POSTWRIGHT_LIVE=1` AND an approved draft in status `queued`.
- **Idiomatic LangGraph**: Stateful graph architecture with SQLite checkpointing and human-in-the-loop interrupts.
- **Configurable LLM Provider**: Anthropic via `langchain-anthropic` by default, configurable via environment variables.

## Quickstart

```bash
# Clone and install dependencies
uv sync

# Configure environment variables
cp .env.example .env
```

## Going Live

By default, `postwright` operates in **dry-run mode** across all commands (`postwright publish`, `postwright publish-due`). In dry-run mode, no HTTP requests are sent to external social media platforms.

### Enabling Live Mode

To enable live publishing to external social media networks, you must explicitly set both required environment variables and meet the safety prerequisites:

1. **Set `POSTWRIGHT_LIVE=1`**:
   ```bash
   export POSTWRIGHT_LIVE=1
   ```
2. **Provide API Credentials**:
   - For X (Twitter): `X_BEARER_TOKEN` or `X_API_KEY`
   ```bash
   export X_BEARER_TOKEN="your_x_api_bearer_token"
   ```
3. **Draft Status Requirement**:
   - A post must be explicitly human-approved and present in the queue with status `queued` (or `failed` with less than 3 retries). Posts already marked `published` can never be sent twice.

### Platform Caveats & Terms

> [!IMPORTANT]
> **X / Twitter API Access & Tiers Notice**:
> X API access tiers (Free, Basic, Pro, Enterprise), rate limits, and Terms of Service change frequently.
> Before enabling `POSTWRIGHT_LIVE=1`, verify that your X API developer app credentials have active Write permissions for the v2 `/2/tweets` endpoint and comply with current developer policies.

## CLI Usage

- `postwright run --note "..."`: Run graph on build notes and pause at human review.
- `postwright review`: Inspect generated drafts, view thread splits / validation results, edit/approve/reject/rewrite.
- `postwright queue`: Show queued/published/failed status and platform post IDs.
- `postwright publish <id>`: Immediately publish a queued post by ID (dry-run unless `POSTWRIGHT_LIVE=1`).
- `postwright publish-due`: Process all queued posts whose scheduled slot time has passed. Meant to be run manually or via cron.

## Running Tests & Checks

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
