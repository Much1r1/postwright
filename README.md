# postwright

A human-in-the-loop LangGraph agent that turns a creator's rough build notes into ready-to-post social media drafts in their own voice.

## Overview

`postwright` is built for AI engineers and creators building in public. It processes rough notes, extracts core technical ideas, selects formats/angles, drafts candidate posts, evaluates quality using a critic loop, proposes an optimal posting schedule (converting audience timezones to `Africa/Nairobi`), and waits for explicit human approval via a LangGraph interrupt before scheduling or publishing.

## Features & Hard Rules

- **Python 3.11+ Managed with `uv`**: Modern, fast Python tooling.
- **Human-in-the-Loop Approval**: Every post must pass a LangGraph `interrupt` for approval. Nothing is published autonomously.
- **Dry-Run by Default**: Publishing to live platforms requires `POSTWRIGHT_LIVE=1` AND an approved draft.
- **Idiomatic LangGraph**: Stateful graph architecture with SQLite checkpointing and human-in-the-loop interrupts.
- **Configurable LLM Provider**: Anthropic via `langchain-anthropic` by default, configurable via environment variables.

## Quickstart

```bash
# Clone and install dependencies
uv sync

# Configure environment variables
cp .env.example .env
```

## Running Tests & Checks

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
