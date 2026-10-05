from postwright.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-3-5-sonnet-20241022"
    assert settings.postwright_live is False
    assert settings.langchain_tracing_v2 is False


def test_settings_custom_env(monkeypatch: None) -> None:
    import os

    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["POSTWRIGHT_LIVE"] = "1"
    try:
        settings = Settings()
        assert settings.llm_provider == "openai"
        assert settings.postwright_live is True
    finally:
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("POSTWRIGHT_LIVE", None)
