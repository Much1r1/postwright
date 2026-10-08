# postwright

A human-in-the-loop LangGraph agent that turns a creator's rough build notes into ready-to-post social media drafts in their own voice.

## Overview

`postwright` is built for AI engineers and creators building in public. It processes rough notes, extracts core technical ideas, selects formats/angles, drafts candidate posts, evaluates quality using a critic loop, proposes an optimal posting schedule (converting audience timezones to `Africa/Nairobi`), and waits for explicit human approval via a LangGraph interrupt before scheduling or publishing.

## Features & Hard Rules

- **Python 3.11+ Managed with `uv`**: Modern, fast Python tooling.
- **Human-in-the-Loop Approval**: Every post must pass a LangGraph `interrupt` for approval. Nothing is published autonomously.
- **Dry-Run by Default**: Publishing to live platforms requires `POSTWRIGHT_LIVE=1` AND an approved draft.
- **Idiomatic LangGraph**: Stateful graph architecture with SQLite checkpointing and human-in-the-loop interrupts.
- **Configurable LLM Provider**: Supports Groq (default), Gemini, and Anthropic via standard LangChain packages.

## Choosing a Provider

`postwright` supports multiple LLM providers: **Groq** (`groq`), **Gemini** (`gemini`), and **Anthropic** (`anthropic`). The default provider is `groq`.

### Provider Setup & Free Tiers

- **Groq** (`LLM_PROVIDER=groq`):
  - Get an API key from the [Groq Console](https://console.groq.com/keys).
  - Model list & supported models: See [Groq Model Documentation](https://console.groq.com/docs/models). Ensure you select a model supporting structured output (e.g. `llama-3.3-70b-versatile`).
  - Set `GROQ_API_KEY` in your `.env` file.
- **Gemini** (`LLM_PROVIDER=gemini`):
  - Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
  - Model list & supported models: See [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini). Ensure you select a model supporting structured output (e.g. `gemini-2.5-flash`).
  - Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in your `.env` file.
- **Anthropic** (`LLM_PROVIDER=anthropic`):
  - Get an API key from the [Anthropic Console](https://console.anthropic.com/).
  - Model list: See [Anthropic Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models). (e.g. `claude-3-5-sonnet-20241022`).
  - Set `ANTHROPIC_API_KEY` in your `.env` file.

> **Important Model Selection Note:** You must choose a model that explicitly supports structured output / function calling. Model default limits and support can change, so always verify available models in each provider's console.

### Rate Limits & Rate-Limit Resilience

Free tiers often enforce strict requests-per-minute (RPM) and tokens-per-minute (TPM) limits:
- `postwright` includes automatic retry logic with exponential backoff and jitter on HTTP `429` rate-limit errors and transient server errors, respecting `Retry-After` headers when returned by the API.
- You can enable an optional client-side throttle by setting `MAX_LLM_CALLS_PER_MINUTE` in your `.env` file. When set, requests will pause and wait rather than failing due to rapid requests.

### Privacy Warning

> ⚠️ **Privacy Warning:** Free-tier API services (such as Groq free tier or Google AI Studio free tier) may store or use submitted prompts for model training or quality assurance and may not guarantee data privacy. Do **not** input sensitive, proprietary, or confidential notes when using free-tier LLM services.

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
