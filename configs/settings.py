"""
Application configuration loaded from environment variables.

Supports:
    - LLM providers: "gemini" or "anthropic"
    - Vector store providers: "chroma" or "pinecone"
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Legal Document Q&A RAG application."""

    # LLM Provider: "gemini" or "anthropic"
    llm_provider: str = Field(default="gemini")

    # API Keys
    google_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    # App
    app_env: str = Field(default="development")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # Vector Store Provider: "chroma" or "pinecone"
    vector_store_provider: str = Field(default="chroma")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="legal_docs")

    # Pinecone
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="legal-doc-qa")

    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Model names per provider
    gemini_model: str = Field(default="gemini-2.5-flash")
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=2048)

    # Retrieval
    top_k: int = Field(default=5)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
