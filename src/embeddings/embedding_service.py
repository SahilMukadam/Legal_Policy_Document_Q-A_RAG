"""
Embedding Module.

Generates vector embeddings from text chunks using Sentence-Transformers.
These embeddings capture the *meaning* of text as numerical vectors,
enabling semantic search (finding similar meaning, not just matching keywords).

How it works:
    "The tenant must pay rent by the 1st" → [0.023, -0.156, 0.891, ...]  (384 dimensions)
    "Monthly payments are due at the start of each month" → [0.019, -0.148, 0.887, ...]

    These two sentences have similar vectors because they mean similar things,
    even though they share almost no words. That's the power of embeddings.

Usage:
    from src.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService()
    vectors = service.embed_texts(["text one", "text two"])
    # Returns: list of 384-dimensional vectors
"""

from sentence_transformers import SentenceTransformer
from configs.settings import settings


class EmbeddingService:
    """Generate embeddings using Sentence-Transformers."""

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: HuggingFace model name. Default from settings.
                        'all-MiniLM-L6-v2' is fast, free, and runs on CPU.
                        Produces 384-dimensional vectors.
        """
        self.model_name = model_name or settings.embedding_model
        self._model = None  # Lazy load — only loads when first used

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model (first call downloads ~80MB, then cached)."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        # SentenceTransformer handles batching internally
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,  # Show progress for large batches
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        # Convert numpy arrays to plain lists for JSON serialization
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string. Convenience method for queries."""
        return self.embed_texts([text])[0]
