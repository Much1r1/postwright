import logging
import os
import random
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from postwright.config import get_settings

logger = logging.getLogger("postwright.llm")

DEFAULT_PRICING_PATH = Path("config/pricing.yaml")


def load_pricing_config(pricing_path: Path | str | None = None) -> dict[str, Any]:
    """Load pricing rules from pricing.yaml."""
    path = Path(pricing_path) if pricing_path else DEFAULT_PRICING_PATH
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load pricing config from {path}: {e}")
        return {}


def get_model_pricing(
    model_name: str, pricing_path: Path | str | None = None
) -> tuple[float, float]:
    """Return (input_cost_per_1m, output_cost_per_1m) for a given model."""
    pricing = load_pricing_config(pricing_path)
    models = pricing.get("models", {}) if isinstance(pricing, dict) else {}

    if model_name in models:
        m_info = models[model_name]
        return (
            float(m_info.get("input_cost_per_1m", 0.0)),
            float(m_info.get("output_cost_per_1m", 0.0)),
        )

    # Check partial match (e.g. 'claude' or 'llama')
    for m_key, m_info in models.items():
        if m_key.lower() in model_name.lower() or model_name.lower() in m_key.lower():
            return (
                float(m_info.get("input_cost_per_1m", 0.0)),
                float(m_info.get("output_cost_per_1m", 0.0)),
            )

    msg = f"Model '{model_name}' not found in pricing.yaml. Defaulting cost to $0.00."
    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
    return (0.0, 0.0)


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing_path: Path | str | None = None,
) -> float:
    """Calculate total cost for token counts based on pricing table."""
    in_rate, out_rate = get_model_pricing(model_name, pricing_path=pricing_path)
    in_cost = (input_tokens / 1_000_000.0) * in_rate
    out_cost = (output_tokens / 1_000_000.0) * out_rate
    return round(in_cost + out_cost, 6)


def extract_token_counts(res: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from various LangChain result formats."""
    in_tokens = 0
    out_tokens = 0

    if hasattr(res, "usage_metadata") and res.usage_metadata:
        um = res.usage_metadata
        if isinstance(um, dict):
            in_tokens = um.get("input_tokens", 0) or um.get("prompt_tokens", 0)
            out_tokens = um.get("output_tokens", 0) or um.get("completion_tokens", 0)
        elif hasattr(um, "input_tokens"):
            in_tokens = getattr(um, "input_tokens", 0)
            out_tokens = getattr(um, "output_tokens", 0)

    if not in_tokens and hasattr(res, "response_metadata") and res.response_metadata:
        rm = res.response_metadata
        token_usage = rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(token_usage, dict):
            in_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            out_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)

    return (in_tokens, out_tokens)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    is_judge: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Returns a ChatModel instance based on settings or parameters.

    Supported providers: groq, gemini (or google), anthropic.
    Default provider is groq (or judge_provider if is_judge=True).
    """
    settings = get_settings()
    if is_judge:
        llm_provider = (provider or settings.judge_provider).lower()
        llm_model = model or settings.judge_model
    else:
        llm_provider = (provider or settings.llm_provider).lower()
        llm_model = model or settings.llm_model

    if llm_provider == "groq":
        api_key = kwargs.pop("api_key", None) or settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required for provider 'groq'."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=llm_model or "placeholder-model",
            api_key=api_key,  # type: ignore[arg-type]
            **kwargs,
        )

    elif llm_provider in ("gemini", "google"):
        api_key = (
            kwargs.pop("api_key", None)
            or settings.google_api_key
            or settings.gemini_api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for provider 'gemini'."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=llm_model or "placeholder-model",
            api_key=api_key,
            **kwargs,
        )

    elif llm_provider == "anthropic":
        api_key = kwargs.pop("api_key", None) or settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for provider 'anthropic'."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=llm_model or "placeholder-model",
            api_key=api_key,  # type: ignore[arg-type]
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{llm_provider}'. Supported providers are: groq, gemini, anthropic."
        )


class RateLimiter:
    """Client-side rate limiter for MAX_LLM_CALLS_PER_MINUTE."""

    def __init__(
        self,
        calls_per_minute: int | None = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.calls_per_minute = calls_per_minute
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.call_timestamps: list[float] = []

    def wait_if_needed(self) -> float:
        if not self.calls_per_minute or self.calls_per_minute <= 0:
            return 0.0

        now = self.time_fn()
        # Keep timestamps within the last 60 seconds
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 60.0]

        waited = 0.0
        if len(self.call_timestamps) >= self.calls_per_minute:
            oldest = self.call_timestamps[0]
            sleep_duration = 60.0 - (now - oldest)
            if sleep_duration > 0:
                self.sleep_fn(sleep_duration)
                waited = sleep_duration
                now = self.time_fn()
                self.call_timestamps = [t for t in self.call_timestamps if now - t < 60.0]

        self.call_timestamps.append(now)
        return waited


_GLOBAL_RATE_LIMITER: RateLimiter | None = None


def _get_rate_limiter(
    calls_per_minute: int | None,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RateLimiter:
    global _GLOBAL_RATE_LIMITER
    if (
        _GLOBAL_RATE_LIMITER is None
        or _GLOBAL_RATE_LIMITER.calls_per_minute != calls_per_minute
        or _GLOBAL_RATE_LIMITER.time_fn != time_fn
        or _GLOBAL_RATE_LIMITER.sleep_fn != sleep_fn
    ):
        _GLOBAL_RATE_LIMITER = RateLimiter(
            calls_per_minute=calls_per_minute,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
    return _GLOBAL_RATE_LIMITER


def is_transient_error(e: Exception) -> bool:
    """Checks if exception is a 429 rate-limit or transient server error."""
    status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True

    response = getattr(e, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int) and (resp_status == 429 or resp_status >= 500):
            return True

    err_str = str(e).lower()
    transient_keywords = [
        "429",
        "rate limit",
        "ratelimit",
        "resource_exhausted",
        "too many requests",
        "overloaded",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "timeout",
        "connection error",
        "connect error",
        "transient",
    ]
    return any(kw in err_str for kw in transient_keywords)


def extract_retry_after(e: Exception) -> float | None:
    """Extracts Retry-After header seconds if present on the exception."""
    headers = None
    if hasattr(e, "headers") and isinstance(e.headers, dict):
        headers = e.headers
    elif (
        hasattr(e, "response")
        and hasattr(e.response, "headers")
        and isinstance(e.response.headers, dict)
    ):
        headers = e.response.headers

    if headers:
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return None


def _execute_single_call_with_retries(
    fn: Callable[[], Any],
    max_retries: int,
    rate_limiter: RateLimiter | None = None,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Any, int, float]:
    retries = 0
    total_wait_time = 0.0

    while True:
        if rate_limiter:
            wait = rate_limiter.wait_if_needed()
            total_wait_time += wait

        try:
            res = fn()
            return res, retries, total_wait_time
        except Exception as e:
            if retries < max_retries and is_transient_error(e):
                retries += 1
                retry_after = extract_retry_after(e)
                base_backoff = (2 ** (retries - 1)) * 1.0 + random.uniform(0.0, 0.1)
                delay = (
                    retry_after
                    if (retry_after is not None and retry_after > 0)
                    else base_backoff
                )
                sleep_fn(delay)
                total_wait_time += delay
            else:
                raise


def invoke_with_resilience(
    llm: Any,
    messages: list[Any],
    schema: type[BaseModel] | None = None,
    model_name: str | None = None,
    max_retries: int | None = None,
    calls_per_minute: int | None = None,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    return_details: bool = False,
) -> tuple[Any, ...]:
    """Invokes LLM with rate-limit resilience, retry on 429/transient errors,
    optional client-side throttling, and 1 retry on structured-output parse failure.

    Returns:
        If return_details is False (default): (result, retries_count, total_wait_time_seconds)
        If return_details is True: (result, input_tokens, output_tokens, cost, retries_count, total_wait_time_seconds)
    """
    settings = get_settings()
    if max_retries is None:
        max_retries = settings.max_llm_retries
    if calls_per_minute is None:
        calls_per_minute = settings.max_llm_calls_per_minute

    rate_limiter = (
        _get_rate_limiter(calls_per_minute, time_fn=time_fn, sleep_fn=sleep_fn)
        if (calls_per_minute is not None and calls_per_minute > 0)
        else None
    )

    if schema is not None and hasattr(llm, "with_structured_output"):
        run_target = llm.with_structured_output(schema)
    else:
        run_target = llm

    total_retries = 0
    total_wait_time = 0.0

    def _resolve_cost_and_tokens(result_obj: Any) -> tuple[int, int, float]:
        in_tok, out_tok = extract_token_counts(result_obj)
        eff_model = (
            model_name
            or getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or settings.llm_model
        )
        cost = calculate_cost(eff_model, in_tok, out_tok)
        return in_tok, out_tok, cost

    try:
        res, retries, wait_time = _execute_single_call_with_retries(
            fn=lambda: run_target.invoke(messages),
            max_retries=max_retries,
            rate_limiter=rate_limiter,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        total_retries += retries
        total_wait_time += wait_time

        if return_details:
            in_tok, out_tok, cost = _resolve_cost_and_tokens(res)
            return res, in_tok, out_tok, cost, total_retries, total_wait_time

        return res, total_retries, total_wait_time
    except Exception as parse_err:
        if schema is None or is_transient_error(parse_err):
            raise

        total_retries += 1
        err_msg = str(parse_err)
        retry_prompt = HumanMessage(
            content=(
                f"The previous response failed schema validation with error:\n{err_msg}\n"
                "Please fix the error and output valid structured data matching the schema."
            )
        )
        retry_messages = list(messages) + [retry_prompt]

        try:
            res, retries, wait_time = _execute_single_call_with_retries(
                fn=lambda: run_target.invoke(retry_messages),
                max_retries=max_retries,
                rate_limiter=rate_limiter,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            )
            total_retries += retries
            total_wait_time += wait_time

            if return_details:
                in_tok, out_tok, cost = _resolve_cost_and_tokens(res)
                return res, in_tok, out_tok, cost, total_retries, total_wait_time

            return res, total_retries, total_wait_time
        except Exception as retry_err:
            raise ValueError(
                f"Structured output parsing failed after retry. Original error: {parse_err}. Retry error: {retry_err}"
            ) from retry_err
