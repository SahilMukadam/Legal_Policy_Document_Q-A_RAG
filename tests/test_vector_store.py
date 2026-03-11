"""
Tests for embedding service and vector store with collection support.
Run with: pytest tests/test_vector_store.py -v
"""

import shutil
from pathlib import Path

import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore


class TestEmbeddingService:

    def test_embed_single_text(self):
        service = EmbeddingService()
        vector = service.embed_single("The tenant shall pay rent monthly.")
        assert isinstance(vector, list)
        assert len(vector) == 384

    def test_embed_multiple_texts(self):
        service = EmbeddingService()
        vectors = service.embed_texts(["Text one.", "Text two.", "Text three."])
        assert len(vectors) == 3

    def test_embed_empty_list(self):
        service = EmbeddingService()
        assert service.embed_texts([]) == []

    def test_similar_texts_have_close_embeddings(self):
        service = EmbeddingService()
        v1 = service.embed_single("The rent is due on the first of each month.")
        v2 = service.embed_single("Monthly payments must be made by the 1st.")
        v3 = service.embed_single("The weather forecast predicts rain tomorrow.")

        sim_related = sum(a * b for a, b in zip(v1, v2))
        sim_unrelated = sum(a * b for a, b in zip(v1, v3))
        assert sim_related > sim_unrelated


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

    def _add_lease(self):
        chunks = [
            {"text": "Rent is 1500 per month due on the first.",
             "metadata": {"source": "lease.pdf", "page": 1, "chunk_index": 0}},
            {"text": "Landlord maintains the property.",
             "metadata": {"source": "lease.pdf", "page": 2, "chunk_index": 1}},
        ]
        return self.store.add_chunks(chunks, doc_id="lease.pdf", collection_name="Real Estate")

    def _add_employment(self):
        chunks = [
            {"text": "Salary is 50000 per year paid monthly.",
             "metadata": {"source": "offer.pdf", "page": 1, "chunk_index": 0}},
        ]
        return self.store.add_chunks(chunks, doc_id="offer.pdf", collection_name="Employment")

    def test_add_and_search(self):
        self._add_lease()
        results = self.store.search("When is rent due?", k=2)
        assert len(results) >= 1
        assert "rent" in results[0]["text"].lower()

    def test_collection_metadata_stored(self):
        self._add_lease()
        results = self.store.search("rent", k=1)
        assert results[0]["metadata"]["collection"] == "Real Estate"

    def test_search_with_source_filters(self):
        self._add_lease()
        self._add_employment()

        results = self.store.search("payment", k=5, source_filters=["lease.pdf"])
        for r in results:
            assert r["metadata"]["source"] == "lease.pdf"

    def test_search_with_multiple_filters(self):
        self._add_lease()
        self._add_employment()

        results = self.store.search("payment", k=5, source_filters=["lease.pdf", "offer.pdf"])
        sources = {r["metadata"]["source"] for r in results}
        assert sources.issubset({"lease.pdf", "offer.pdf"})

    def test_list_collections(self):
        self._add_lease()
        self._add_employment()

        collections = self.store.list_collections()
        assert "Real Estate" in collections
        assert "Employment" in collections
        assert len(collections["Real Estate"]) == 1
        assert collections["Real Estate"][0]["source"] == "lease.pdf"

    def test_list_documents_flat(self):
        self._add_lease()
        self._add_employment()

        docs = self.store.list_documents()
        assert len(docs) == 2
        assert all("collection" in d for d in docs)

    def test_get_sources_for_collections(self):
        self._add_lease()
        self._add_employment()

        sources = self.store.get_sources_for_collections(["Real Estate"])
        assert "lease.pdf" in sources
        assert "offer.pdf" not in sources

    def test_delete_document(self):
        self._add_lease()
        self._add_employment()

        num_deleted = self.store.delete_document("lease.pdf")
        assert num_deleted == 2
        assert self.store.get_stats()["total_chunks"] == 1

    def test_delete_collection_group(self):
        self._add_lease()
        self._add_employment()

        num_deleted = self.store.delete_collection_group("Real Estate")
        assert num_deleted == 2

        collections = self.store.list_collections()
        assert "Real Estate" not in collections
        assert "Employment" in collections

    def test_document_exists(self):
        assert not self.store.document_exists("lease.pdf")
        self._add_lease()
        assert self.store.document_exists("lease.pdf")

    def test_get_stats(self):
        self._add_lease()
        self._add_employment()

        stats = self.store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert stats["total_collections"] == 2
        assert "Real Estate" in stats["collections"]

    def test_empty_store_search(self):
        results = self.store.search("anything", k=5)
        assert results == []
