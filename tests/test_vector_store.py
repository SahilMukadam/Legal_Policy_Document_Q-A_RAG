"""
Tests for embedding service and vector store.
Run with: pytest tests/ -v
"""

import shutil
from pathlib import Path

import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore


# ---- Embedding Tests ----

class TestEmbeddingService:

    def test_embed_single_text(self):
        """Verify embedding generates a vector of correct dimension."""
        service = EmbeddingService()
        vector = service.embed_single("The tenant shall pay rent monthly.")

        assert isinstance(vector, list)
        assert len(vector) == 384  # all-MiniLM-L6-v2 dimension
        assert all(isinstance(v, float) for v in vector)

    def test_embed_multiple_texts(self):
        """Verify batch embedding works."""
        service = EmbeddingService()
        texts = [
            "The contract expires in 30 days.",
            "Payment is due on the first of each month.",
            "The landlord may terminate with written notice.",
        ]
        vectors = service.embed_texts(texts)

        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_empty_list(self):
        """Empty input should return empty output."""
        service = EmbeddingService()
        vectors = service.embed_texts([])
        assert vectors == []

    def test_similar_texts_have_close_embeddings(self):
        """Semantically similar texts should have similar vectors."""
        service = EmbeddingService()
        v1 = service.embed_single("The rent is due on the first of each month.")
        v2 = service.embed_single("Monthly payments must be made by the 1st.")
        v3 = service.embed_single("The weather forecast predicts rain tomorrow.")

        # Cosine similarity (vectors are normalized, so dot product = cosine sim)
        sim_related = sum(a * b for a, b in zip(v1, v2))
        sim_unrelated = sum(a * b for a, b in zip(v1, v3))

        # Related texts should be more similar than unrelated ones
        assert sim_related > sim_unrelated


# ---- Vector Store Tests ----

class TestVectorStore:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a clean vector store for each test."""
        # Use a test-specific directory
        import configs.settings as cfg
        original_dir = cfg.settings.chroma_persist_dir
        cfg.settings.chroma_persist_dir = "./test_chroma_data"
        cfg.settings.chroma_collection_name = "test_collection"

        self.store = VectorStore()
        self.store.delete_collection()  # Start fresh

        yield

        # Cleanup
        self.store.delete_collection()
        cfg.settings.chroma_persist_dir = original_dir
        test_dir = Path("./test_chroma_data")
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_add_and_search(self):
        """Add chunks and verify search returns relevant results."""
        chunks = [
            {"text": "The tenant must pay rent by the 1st of each month.",
             "metadata": {"source": "lease.pdf", "page": 1, "chunk_index": 0}},
            {"text": "The landlord shall maintain the property in good condition.",
             "metadata": {"source": "lease.pdf", "page": 2, "chunk_index": 1}},
            {"text": "Either party may terminate with 30 days written notice.",
             "metadata": {"source": "lease.pdf", "page": 3, "chunk_index": 2}},
        ]

        num_added = self.store.add_chunks(chunks)
        assert num_added == 3

        # Search for rent-related content
        results = self.store.search("When is rent due?", k=2)
        assert len(results) == 2
        assert "rent" in results[0]["text"].lower()  # Most relevant should mention rent

    def test_search_returns_scores(self):
        """Verify search results include similarity scores."""
        chunks = [
            {"text": "Privacy policy governs data collection.",
             "metadata": {"source": "policy.pdf", "page": 1, "chunk_index": 0}},
        ]
        self.store.add_chunks(chunks)

        results = self.store.search("data privacy", k=1)
        assert len(results) == 1
        assert "score" in results[0]
        assert 0 <= results[0]["score"] <= 1

    def test_get_stats(self):
        """Verify stats reflect stored data."""
        chunks = [
            {"text": "Test chunk one.", "metadata": {"source": "test.txt", "page": 1, "chunk_index": 0}},
            {"text": "Test chunk two.", "metadata": {"source": "test.txt", "page": 1, "chunk_index": 1}},
        ]
        self.store.add_chunks(chunks)

        stats = self.store.get_stats()
        assert stats["total_chunks"] == 2

    def test_empty_store_search(self):
        """Searching an empty store should return empty results."""
        results = self.store.search("anything", k=5)
        assert results == []
