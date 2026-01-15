from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field

ProviderName = Literal["openai", "openrouter", "mistral"]


class LLMSettings(BaseSettings):
    """
    Глобальные настройки клиента LLM/Embeddings.
    - Загружает значения из переменных окружения и из .env.
    - Отсутствие переменных окружения НЕ приводит к ошибкам — используются дефолты.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Общие ----
    default_provider: ProviderName = Field(default="openai")

    # ---- Логирование (одна переменная на функциональность) ----
    log_level: str = Field(default="INFO")

    # ---- OpenAI ----
    openai_chat_model_name: str = Field(default="gpt-4o-mini")
    openai_embedding_model_name: str = Field(default="text-embedding-3-small")
    openai_api_key: SecretStr | None = Field(default=None)
    openai_api_base: str | None = Field(default=None)

    # ---- OpenRouter ----
    openrouter_chat_model: str = Field(default="openrouter/auto")
    openrouter_emb_model: str = Field(default="text-embedding-3-small")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_api_key: SecretStr | None = Field(default=None)

    # Таймауты и ретраи
    request_timeout_s: float = 60.0
    connect_timeout_s: float = 10.0
    max_retries: int = 5
    retry_base_s: float = 1.0
    retry_max_s: float = 20.0
    retry_jitter_s: float = 0.5

    # Батч для эмбеддингов
    emb_batch_size: int = 64

    # ---- Mistral ----
    mistral_chat_model: str = Field(default="mistral-large-latest")
    mistral_emb_model: str = Field(default="mistral-embed")
    mistral_api_key: SecretStr | None = Field(default=None)


def get_settings() -> LLMSettings:
    """
    Возвращает НОВЫЙ экземпляр настроек.
    (Без синглтона: каждый вызов создаёт объект заново.)
    """
    return LLMSettings()


if __name__ == "__main__":

    import os

    print("=== LLMSettings mini-test ===")
    s = get_settings()
    print("default_provider:", s.default_provider)
    print("openai_chat_model_name:", s.openai_chat_model_name)
    print("openrouter_base_url:", s.openrouter_base_url)

    print("has OPENAI key?:", bool(s.openai_api_key and s.openai_api_key.get_secret_value()))
    print("has OPENROUTER key?:", bool(s.openrouter_api_key and s.openrouter_api_key.get_secret_value()))
    print("has MISTRAL key?:", bool(s.mistral_api_key and s.mistral_api_key.get_secret_value()))

    # env переопределяет дефолт
    os.environ["OPENAI_CHAT_MODEL_NAME"] = "gpt-4o-mini-2025"
    s2 = get_settings()
    print("override openai_chat_model_name:", s2.openai_chat_model_name)
