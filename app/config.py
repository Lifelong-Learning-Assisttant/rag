"""Конфигурация приложения RAG API."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigBase(BaseSettings):
    """Базовый класс для всех конфигураций."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )


class AppConfig(ConfigBase):
    """Настройки приложения."""

    model_config = SettingsConfigDict(env_prefix="app_")

    name: str = Field(default="RAG API Service", description="Название приложения")
    version: str = Field(default="1.0.0", description="Версия приложения")
    debug: bool = Field(default=False, description="Режим отладки")


class OpenAIConfig(ConfigBase):
    """Настройки OpenAI."""

    model_config = SettingsConfigDict(env_prefix="openai_")

    api_key: str | None = Field(default=None, description="API ключ OpenAI")
    api_base: str | None = Field(default=None, description="Базовый URL API OpenAI")
    embedding_model_name: str = Field(
        default="text-embedding-3-large",
        description="Модель для создания эмбеддингов",
    )
    chat_model_name: str = Field(
        default="gpt-4o-mini", description="Модель для генерации ответов"
    )


class QdrantConfig(ConfigBase):
    """Настройки Qdrant."""

    model_config = SettingsConfigDict(env_prefix="qdrant_")

    url: str = Field(
        default="http://localhost:6333", description="URL подключения к Qdrant"
    )
    collection_name: str = Field(
        default="yandex_handbook_child_chunks",
        description="Название коллекции в Qdrant",
    )


class RedisConfig(ConfigBase):
    """Настройки Redis."""

    model_config = SettingsConfigDict(env_prefix="redis_")

    url: str = Field(
        default="redis://localhost:6379", description="URL подключения к Redis"
    )


class RAGConfig(ConfigBase):
    """Настройки RAG."""

    model_config = SettingsConfigDict(env_prefix="rag_")

    retrieval_top_k: int = Field(
        default=10, description="Количество документов для поиска"
    )
    max_context_tokens: int = Field(
        default=4000, description="Максимальное количество токенов в контексте"
    )
    use_hyde_by_default: bool = Field(
        default=False, description="Использовать HyDE по умолчанию"
    )
    score_threshold: float = Field(
        default=0.7, 
        description="Порог скора для фильтрации результатов (0.0-1.0, меньше = строже)"
    )


class Settings(BaseSettings):
    """Главный класс настроек приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # Группы настроек
    app: AppConfig = Field(
        default_factory=AppConfig, description="Настройки приложения"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="Настройки OpenAI"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig, description="Настройки Qdrant"
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig, description="Настройки Redis"
    )
    rag: RAGConfig = Field(default_factory=RAGConfig, description="Настройки RAG")


settings = Settings()

