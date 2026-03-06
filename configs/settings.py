"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe config management.

Supports multiple LLM providers — switch by changing LLM_PROVIDER in .env:
  - "gemini"    → Google Gemini (free tier)
  - "anthropic" → Claude API (paid, swap in for final demo)
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Legal Document Q&A RAG application."""

    # LLM Provider: "gemini" or "anthropic"
    llm_provider: str = Field(default="gemini")

    # API Keys (only the active provider's key is required)
    google_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

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

    # Model names per provider
    gemini_model: str = Field(default="gemini-2.5-flash-lite")
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