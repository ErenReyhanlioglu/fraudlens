"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    All values are read from environment variables or the `.env` file at the
    project root. Secrets are wrapped in `SecretStr` so they never appear in
    logs or error messages by accident.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # Postgres
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    # Redis
    redis_password: SecretStr = Field(alias="REDIS_PASSWORD")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # Qdrant
    qdrant_api_key: SecretStr = Field(alias="QDRANT_API_KEY")
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")

    # MLflow
    mlflow_host: str = Field(default="localhost", alias="MLFLOW_HOST")
    mlflow_port: int = Field(default=5000, alias="MLFLOW_PORT")

    # Anthropic
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    anthropic_model_haiku: str = Field(
        default="claude-haiku-4-5-20251001", alias="ANTHROPIC_MODEL_HAIKU"
    )
    anthropic_model_sonnet: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL_SONNET")

    # LangSmith
    langsmith_api_key: SecretStr = Field(default=SecretStr(""), alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="fraudlens", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy DSN for the Postgres instance."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis DSN, auth included."""
        password = self.redis_password.get_secret_value()
        return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def qdrant_url(self) -> str:
        """Qdrant HTTP endpoint."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking server URI."""
        return f"http://{self.mlflow_host}:{self.mlflow_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Using a cache avoids reparsing the environment on every call and keeps
    the configuration effectively immutable for the process lifetime.
    """
    return Settings()  # type: ignore[call-arg]
