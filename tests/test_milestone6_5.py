import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from postwright.llm import RateLimiter, get_llm, invoke_with_resilience


class SimpleOutput(BaseModel):
    message: str


class MockChatModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    call_count: int = 0
    received_messages: list[Any] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        return None

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.call_count += 1
        self.received_messages.append(input)
        if isinstance(self.responses, list) and self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return "Default Mock Response"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        mock = MagicMock()

        def _structured_invoke(messages: Any) -> Any:
            return self.invoke(messages)

        mock.invoke.side_effect = _structured_invoke
        return mock


def test_factory_returns_right_class_and_validates_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test Groq instantiation
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "secret_groq_key_12345")

    llm_groq = get_llm()
    assert isinstance(llm_groq, ChatGroq)

    # Test Gemini instantiation
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "secret_google_key_67890")

    llm_gemini = get_llm()
    assert isinstance(llm_gemini, ChatGoogleGenerativeAI)

    # Test Anthropic instantiation
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret_anthropic_key_99999")

    llm_anthropic = get_llm()
    assert isinstance(llm_anthropic, ChatAnthropic)

    # Test Judge LLM instantiation wiring
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "secret_judge_groq_key")
    llm_judge = get_llm(is_judge=True)
    assert isinstance(llm_judge, ChatGroq)


def test_factory_missing_key_error_does_not_leak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_value = "super_secret_key_value_do_not_leak"
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        get_llm()

    err_msg = str(exc_info.value)
    assert "GROQ_API_KEY" in err_msg
    assert secret_value not in err_msg


def test_backoff_retries_on_429_honors_retry_after_and_stops_at_max() -> None:
    slept_times: list[float] = []

    def mock_sleep(seconds: float) -> None:
        slept_times.append(seconds)

    class ErrorWith429(Exception):
        def __init__(self, msg: str, headers: dict[str, str] | None = None) -> None:
            super().__init__(msg)
            self.status_code = 429
            self.headers = headers or {}

    # Case 1: Succeeds after 2 transient failures
    err_429_header = ErrorWith429("Rate limit exceeded", headers={"Retry-After": "2.5"})
    err_transient = Exception("503 Service Unavailable")
    mock_llm = MockChatModel(responses=[err_429_header, err_transient, "Success Output"])

    res, retries, wait_time = invoke_with_resilience(
        llm=mock_llm,
        messages=["Hello"],
        max_retries=3,
        sleep_fn=mock_sleep,
    )

    assert res == "Success Output"
    assert retries == 2
    assert len(slept_times) == 2
    assert slept_times[0] == 2.5
    assert wait_time >= 2.5

    # Case 2: Max retries exceeded
    slept_times.clear()
    err_persistent = ErrorWith429("Rate limit persistent")
    mock_llm_fail = MockChatModel(
        responses=[err_persistent, err_persistent, err_persistent, err_persistent]
    )

    with pytest.raises(ErrorWith429):
        invoke_with_resilience(
            llm=mock_llm_fail,
            messages=["Hello"],
            max_retries=2,
            sleep_fn=mock_sleep,
        )

    assert len(slept_times) == 2


def test_throttle_spaces_calls_correctly_using_injected_clock() -> None:
    current_time = 1000.0
    slept_durations: list[float] = []

    def time_fn() -> float:
        return current_time

    def sleep_fn(seconds: float) -> None:
        nonlocal current_time
        slept_durations.append(seconds)
        current_time += seconds

    limiter = RateLimiter(calls_per_minute=2, time_fn=time_fn, sleep_fn=sleep_fn)

    # Call 1 at t=1000.0 (no wait)
    w1 = limiter.wait_if_needed()
    assert w1 == 0.0

    # Call 2 at t=1000.0 (no wait)
    w2 = limiter.wait_if_needed()
    assert w2 == 0.0

    # Call 3 at t=1000.0 (exceeds 2 calls/min -> should sleep 60.0s to t=1060.0)
    w3 = limiter.wait_if_needed()
    assert w3 == 60.0
    assert len(slept_durations) == 1
    assert slept_durations[0] == 60.0
    assert current_time == 1060.0


def test_structured_output_retry_once_then_raises() -> None:
    # Case 1: Structured output fails once with schema error, succeeds on retry
    mock_llm_fix = MockChatModel(
        responses=[
            ValueError("Invalid schema: missing field 'message'"),
            SimpleOutput(message="Fixed schema output"),
        ]
    )

    res, retries, _wait = invoke_with_resilience(
        llm=mock_llm_fix,
        messages=["Generate output"],
        schema=SimpleOutput,
    )

    assert res.message == "Fixed schema output"
    assert retries == 1
    assert len(mock_llm_fix.received_messages) == 2
    second_msg_batch = mock_llm_fix.received_messages[1]
    assert "The previous response failed schema validation" in str(second_msg_batch)

    # Case 2: Fails twice on schema error -> raises ValueError
    mock_llm_double_fail = MockChatModel(
        responses=[
            ValueError("Invalid schema error 1"),
            ValueError("Invalid schema error 2"),
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        invoke_with_resilience(
            llm=mock_llm_double_fail,
            messages=["Generate output"],
            schema=SimpleOutput,
        )

    assert "Structured output parsing failed after retry" in str(exc_info.value)


def test_only_llm_py_constructs_chat_models() -> None:
    src_dir = Path("src")
    llm_file = src_dir / "postwright" / "llm.py"

    chat_model_instantiations = [
        r"ChatGroq\(",
        r"ChatGoogleGenerativeAI\(",
        r"ChatAnthropic\(",
        r"init_chat_model\(",
    ]

    for py_file in src_dir.rglob("*.py"):
        if py_file.resolve() == llm_file.resolve():
            continue

        content = py_file.read_text(encoding="utf-8")
        for pattern in chat_model_instantiations:
            match = re.search(pattern, content)
            assert match is None, (
                f"Forbidden chat model instantiation pattern '{pattern}' found in {py_file}. "
                "Only llm.py is permitted to construct ChatModel instances."
            )


def test_only_one_resilience_helper_exists() -> None:
    src_dir = Path("src")
    resilience_patterns = [
        r"def invoke_llm_with_resilience",
        r"class RateThrottler",
    ]
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pattern in resilience_patterns:
            match = re.search(pattern, content)
            assert match is None, (
                f"Forbidden duplicate resilience component '{pattern}' found in {py_file}. "
                "Exactly one resilience helper (invoke_with_resilience) and RateLimiter in llm.py must exist."
            )
