"""
Tests for embedding service and vector store.
Run with: pytest tests/test_vector_store.py -v
"""

import shutil
from pathlib import Path

import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore


# ---- Embedding Tests ----

class TestEmbeddingService:

    def test_embed_single_text(self):
        service = EmbeddingService()
        vector = service.embed_single("The tenant shall pay rent monthly.")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    def test_embed_multiple_texts(self):
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
        service = EmbeddingService()
        vectors = service.embed_texts([])
        assert vectors == []

    def test_similar_texts_have_close_embeddings(self):
        service = EmbeddingService()
        v1 = service.embed_single("The rent is due on the first of each month.")
        v2 = service.embed_single("Monthly payments must be made by the 1st.")
        v3 = service.embed_single("The weather forecast predicts rain tomorrow.")

        sim_related = sum(a * b for a, b in zip(v1, v2))
        sim_unrelated = sum(a * b for a, b in zip(v1, v3))
        assert sim_related > sim_unrelated


# ---- Vector Store Tests ----

class TestVectorStore:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        import configs.settings as cfg
        original_dir = cfg.settings.chroma_persist_dir
        original_name = cfg.settings.chroma_collection_name
        cfg.settings.chroma_persist_dir = "./test_chroma_data"
        cfg.settings.chroma_collection_name = "test_collection"

        self.store = VectorStore()
        self.store.delete_collection()

        yield

        self.store.delete_collection()
        cfg.settings.chroma_persist_dir = original_dir
        cfg.settings.chroma_collection_name = original_name
        test_dir = Path("./test_chroma_data")
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    def _sample_chunks(self, source="lease.pdf"):
        return [
            {"text": "The tenant must pay rent by the 1st of each month.",
             "metadata": {"source": source, "page": 1, "chunk_index": 0}},
            {"text": "The landlord shall maintain the property in good condition.",
             "metadata": {"source": source, "page": 2, "chunk_index": 1}},
            {"text": "Either party may terminate with 30 days written notice.",
             "metadata": {"source": source, "page": 3, "chunk_index": 2}},
        ]

    def test_add_and_search(self):
        num_added = self.store.add_chunks(self._sample_chunks())
        assert num_added == 3

        results = self.store.search("When is rent due?", k=2)
        assert len(results) == 2
        assert "rent" in results[0]["text"].lower()

    def test_search_returns_scores(self):
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
        self.store.add_chunks(self._sample_chunks())
        stats = self.store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 1

    def test_empty_store_search(self):
        results = self.store.search("anything", k=5)
        assert results == []

    # ---- New Day 6 tests ----

    def test_search_with_source_filter(self):
        """Search filtered to a specific document."""
        self.store.add_chunks(self._sample_chunks("lease.pdf"))
        self.store.add_chunks([
            {"text": "Annual salary is 50000 pounds paid monthly.",
             "metadata": {"source": "employment.pdf", "page": 1, "chunk_index": 0}},
        ])

        # Search only in lease.pdf
        results = self.store.search("payment", k=5, source_filter="lease.pdf")
        for r in results:
            assert r["metadata"]["source"] == "lease.pdf"

    def test_list_documents(self):
        """List all uploaded documents."""
        self.store.add_chunks(self._sample_chunks("lease.pdf"))
        self.store.add_chunks([
            {"text": "Employment terms apply.",
             "metadata": {"source": "employment.pdf", "page": 1, "chunk_index": 0}},
        ])

        docs = self.store.list_documents()
        assert len(docs) == 2
        sources = [d["source"] for d in docs]
        assert "lease.pdf" in sources
        assert "employment.pdf" in sources

    def test_delete_document(self):
        """Delete a specific document's chunks."""
        self.store.add_chunks(self._sample_chunks("lease.pdf"))
        self.store.add_chunks([
            {"text": "Employment terms apply.",
             "metadata": {"source": "employment.pdf", "page": 1, "chunk_index": 0}},
        ])

        assert self.store.get_stats()["total_chunks"] == 4

        num_deleted = self.store.delete_document("lease.pdf")
        assert num_deleted == 3
        assert self.store.get_stats()["total_chunks"] == 1

        # Only employment.pdf should remain
        docs = self.store.list_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "employment.pdf"

    def test_document_exists(self):
        """Check if a document has been uploaded."""
        assert not self.store.document_exists("lease.pdf")

        self.store.add_chunks(self._sample_chunks("lease.pdf"))
        assert self.store.document_exists("lease.pdf")
        assert not self.store.document_exists("nonexistent.pdf")
