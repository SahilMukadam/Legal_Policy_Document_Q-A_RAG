"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe config management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Legal Document Q&A RAG application."""

    # API Keys
    anthropic_api_key: str = Field(..., description="Anthropic API key for Claude")

    # App
    app_env: str = Field(default="development")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="legal_docs")

    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Claude Model
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=2048)

    # Retrieval
    top_k: int = Field(default=5, description="Number of chunks to retrieve")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this throughout the app
settings = Settings()
