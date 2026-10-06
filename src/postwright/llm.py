import os
import random
import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from postwright.config import get_settings


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
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
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
            settings.google_api_key
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
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
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
    llm: BaseChatModel,
    messages: list[Any],
    schema: type[BaseModel] | None = None,
    max_retries: int | None = None,
    calls_per_minute: int | None = None,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Any, int, float]:
    """Invokes LLM with rate-limit resilience, retry on 429/transient errors,
    optional client-side throttling, and 1 retry on structured-output parse failure.

    Returns:
        (result, retries_count, total_wait_time_seconds)
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

    run_target = (
        llm.with_structured_output(schema) if schema is not None else llm
    )

    total_retries = 0
    total_wait_time = 0.0

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
            return res, total_retries, total_wait_time
        except Exception as retry_err:
            raise ValueError(
                f"Structured output parsing failed after retry. Original error: {parse_err}. Retry error: {retry_err}"
            ) from retry_err
