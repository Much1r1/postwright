import logging
import time
import warnings
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from postwright.config import get_settings

logger = logging.getLogger("postwright.llm")

DEFAULT_PRICING_PATH = Path("config/pricing.yaml")


class RateThrottler:
    """Enforces max LLM calls per minute using sliding window."""

    def __init__(self, max_calls_per_minute: int = 30) -> None:
        self.max_calls_per_minute = max_calls_per_minute
        self.call_timestamps: list[float] = []

    def wait_if_needed(self) -> float:
        if self.max_calls_per_minute <= 0:
            return 0.0

        now = time.time()
        # Remove timestamps older than 60 seconds
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 60.0]

        wait_seconds = 0.0
        if len(self.call_timestamps) >= self.max_calls_per_minute:
            oldest_in_window = self.call_timestamps[0]
            wait_seconds = max(0.0, 60.0 - (now - oldest_in_window))
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.time()

        self.call_timestamps.append(now)
        return wait_seconds


_global_throttler: RateThrottler | None = None


def get_rate_throttler() -> RateThrottler:
    global _global_throttler
    if _global_throttler is None:
        settings = get_settings()
        _global_throttler = RateThrottler(max_calls_per_minute=settings.max_llm_calls_per_minute)
    return _global_throttler


def load_pricing_config(pricing_path: Path | str | None = None) -> dict[str, Any]:
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

    if llm_provider in ("groq",):
        try:
            from langchain_groq import ChatGroq

            api_key = kwargs.pop("api_key", settings.groq_api_key)
            return cast(
                BaseChatModel,
                ChatGroq(
                    model_name=llm_model,
                    groq_api_key=api_key,
                    **kwargs,
                ),
            )
        except ImportError:
            from langchain.chat_models import init_chat_model

            return cast(
                BaseChatModel,
                init_chat_model(llm_model, model_provider="groq", **kwargs),
            )

    elif llm_provider in ("google", "gemini", "google_genai"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = kwargs.pop("api_key", settings.google_api_key)
            return cast(
                BaseChatModel,
                ChatGoogleGenerativeAI(
                    model=llm_model,
                    google_api_key=api_key,
                    **kwargs,
                ),
            )
        except ImportError:
            from langchain.chat_models import init_chat_model

            return cast(
                BaseChatModel,
                init_chat_model(llm_model, model_provider="google_genai", **kwargs),
            )

    elif llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = kwargs.pop("api_key", settings.anthropic_api_key)
        return cast(
            BaseChatModel,
            ChatAnthropic(
                model_name=llm_model,
                api_key=api_key,
                **kwargs,
            ),
        )

    else:
        from langchain.chat_models import init_chat_model

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


def invoke_llm_with_resilience(
    runnable: Any,
    messages: Any,
    model_name: str | None = None,
    max_retries: int | None = None,
) -> tuple[Any, int, int, float, int, float]:
    """Invoke LLM/runnable with rate throttling, exponential backoff retries, and token/cost extraction.

    Returns:
        (result, input_tokens, output_tokens, cost, retry_count, total_wait_time_seconds)
    """
    settings = get_settings()
    eff_max_retries = max_retries if max_retries is not None else settings.max_llm_retries
    eff_model = model_name or settings.llm_model

    throttler = get_rate_throttler()

    retries = 0
    total_wait_time = 0.0

    # Apply rate limit throttle wait
    throttle_wait = throttler.wait_if_needed()
    total_wait_time += throttle_wait

    for attempt in range(eff_max_retries + 1):
        try:
            res = runnable.invoke(messages)
            in_tokens, out_tokens = extract_token_counts(res)
            cost = calculate_cost(eff_model, in_tokens, out_tokens)
            return (res, in_tokens, out_tokens, cost, retries, total_wait_time)

        except Exception as e:
            if attempt < eff_max_retries:
                retries += 1
                backoff_time = 2.0 ** attempt
                logger.warning(
                    f"LLM invocation failed (attempt {attempt + 1}/{eff_max_retries + 1}): {e}. Retrying in {backoff_time}s..."
                )
                time.sleep(backoff_time)
                total_wait_time += backoff_time
            else:
                logger.error(f"LLM invocation failed after {eff_max_retries} retries: {e}")
                raise

    raise RuntimeError("LLM invocation loop terminated unexpectedly.")
