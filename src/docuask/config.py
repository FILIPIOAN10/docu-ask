"""All configuration comes from environment variables (or a local `.env`), validated once at startup."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://docuask:docuask@localhost:5434/docuask"
    # Comma-separated list, so a key can be rotated without downtime (add the new one, remove the old one later).
    api_keys: SecretStr = SecretStr("dev-key")

    # "fake" needs no network and no key: tests, CI and the first demo. The others call a real provider.
    llm_provider: Literal["fake", "openai", "azure"] = "fake"
    # Must equal the width of chunks.embedding (migration 0001). Changing the model = new migration + reindex.
    embedding_dim: int = 1536

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_embedding_deployment: str | None = None
    azure_chat_deployment: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_bytes: int = 5_000_000

    # How many passages go into the prompt, how many candidates each search leg fetches before fusion, and the
    # cosine distance above which the nearest passage is "too far" and the question is refused without calling the
    # model. Calibrate it on YOUR embeddings and questions (python -m docuask.evaluate prints the distances).
    retrieval_k: int = 4
    candidate_k: int = 20
    max_distance: float = 0.95

    @property
    def keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
