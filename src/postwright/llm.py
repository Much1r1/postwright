from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel

from postwright.config import get_settings


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Returns a ChatModel instance based on settings or parameters."""
    settings = get_settings()
    llm_provider = (provider or settings.llm_provider).lower()
    llm_model = model or settings.llm_model

    if llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = settings.anthropic_api_key
        return ChatAnthropic(
            model_name=llm_model,
            api_key=api_key,  # type: ignore[arg-type]
            **kwargs,
        )
    else:
        # Fallback to init_chat_model for other providers
        from langchain.chat_models import init_chat_model

        return cast(
            BaseChatModel,
            init_chat_model(llm_model, model_provider=llm_provider, **kwargs),
        )
