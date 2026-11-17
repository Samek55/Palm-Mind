from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Palm Mind RAG Service"
    debug: bool = False

    # SQL database (documents metadata + bookings)
    database_url: str = "sqlite:///./data/app.db"

    # Qdrant. If qdrant_url is empty we fall back to embedded (on-disk) mode,
    # which is handy for local development without a running server.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str = "./data/qdrant"
    qdrant_collection: str = "documents"

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_history_max_messages: int = 20
    chat_session_ttl_seconds: int = 60 * 60 * 24

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # Ingestion / retrieval
    max_upload_size_mb: int = 20
    fixed_chunk_size: int = 200  # words
    fixed_chunk_overlap: int = 40  # words
    semantic_similarity_threshold: float = 0.55
    semantic_max_chunk_words: int = 300
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_score_threshold: float = 0.25


@lru_cache
def get_settings() -> Settings:
    return Settings()
