from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    llm_model: str = Field(default="claude-3-5-sonnet-20241022", alias="LLM_MODEL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    critic_score_threshold: int = Field(default=14, alias="CRITIC_SCORE_THRESHOLD")
    max_revisions: int = Field(default=2, alias="MAX_REVISIONS")
    max_human_rewrites: int = Field(default=3, alias="MAX_HUMAN_REWRITES")

    postwright_checkpoint_db: str = Field(
        default="postwright_checkpoints.db", alias="POSTWRIGHT_CHECKPOINT_DB"
    )
    postwright_store_db: str = Field(
        default="postwright_store.db", alias="POSTWRIGHT_STORE_DB"
    )

    postwright_live: bool = Field(default=False, alias="POSTWRIGHT_LIVE")

    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="postwright", alias="LANGCHAIN_PROJECT")


def get_settings() -> Settings:
    return Settings()
