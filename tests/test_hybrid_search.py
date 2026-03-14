"""
Tests for hybrid search (BM25 + semantic + RRF).
Run with: pytest tests/test_hybrid_search.py -v
"""

import shutil
from pathlib import Path

import pytest

from src.retrieval.hybrid_search import BM25, HybridSearch
from src.retrieval.vector_store import VectorStore


# ---- BM25 Tests ----

class TestBM25:

    def setup_method(self):
        self.bm25 = BM25()
        self.documents = [
            {"text": "The tenant must pay rent by the first of each month.",
             "metadata": {"source": "lease.pdf", "page": 1}},
            {"text": "The landlord is responsible for structural repairs and plumbing.",
             "metadata": {"source": "lease.pdf", "page": 2}},
            {"text": "Smoking is prohibited inside the property at all times.",
             "metadata": {"source": "lease.pdf", "page": 3}},
            {"text": "The security deposit of 3000 GBP is required upon signing.",
             "metadata": {"source": "lease.pdf", "page": 4}},
        ]
        self.bm25.index(self.documents)

    def test_basic_search(self):
        results = self.bm25.search("rent payment", k=2)
        assert len(results) >= 1
        assert "rent" in results[0]["text"].lower()

    def test_exact_term_match(self):
        """BM25 should excel at exact term matching."""
        results = self.bm25.search("smoking prohibited", k=1)
        assert len(results) == 1
        assert "smoking" in results[0]["text"].lower()

    def test_no_match_returns_empty(self):
        results = self.bm25.search("xyznonexistent", k=5)
        assert results == []

    def test_score_ordering(self):
        results = self.bm25.search("security deposit GBP", k=4)
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_empty_index(self):
        empty_bm25 = BM25()
        empty_bm25.index([])
        results = empty_bm25.search("anything", k=5)
        assert results == []


# ---- Hybrid Search Tests ----

class TestHybridSearch:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        import configs.settings as cfg
        original_dir = cfg.settings.chroma_persist_dir
        original_name = cfg.settings.chroma_collection_name
        cfg.settings.chroma_persist_dir = "./test_hybrid_chroma"
        cfg.settings.chroma_collection_name = "test_hybrid"

        self.store = VectorStore()
        self.store.delete_collection()

        # Add test documents
        chunks = [
            {"text": "The monthly rent is 1500 GBP due on the first day of each month.",
             "metadata": {"source": "lease.pdf", "page": 1, "chunk_index": 0}},
            {"text": "Section 6: Either party may terminate with 60 days written notice.",
             "metadata": {"source": "lease.pdf", "page": 3, "chunk_index": 1}},
            {"text": "The security deposit of 3000 GBP is held in a government-approved scheme.",
             "metadata": {"source": "lease.pdf", "page": 2, "chunk_index": 2}},
            {"text": "No pets are allowed without prior written approval from the landlord.",
             "metadata": {"source": "lease.pdf", "page": 4, "chunk_index": 3}},
        ]
        self.store.add_chunks(chunks, doc_id="lease.pdf", collection_name="Test")
        self.hybrid = HybridSearch(self.store)

        yield

        self.store.delete_collection()
        cfg.settings.chroma_persist_dir = original_dir
        cfg.settings.chroma_collection_name = original_name
        test_dir = Path("./test_hybrid_chroma")
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_hybrid_returns_results(self):
        results = self.hybrid.search("When is rent due?", k=3)
        assert len(results) >= 1
        assert "rent" in results[0]["text"].lower()

    def test_hybrid_marks_search_type(self):
        results = self.hybrid.search("rent payment", k=3)
        for r in results:
            assert r["search_type"] == "hybrid"

    def test_exact_term_found(self):
        """Hybrid should find exact section references that semantic might miss."""
        results = self.hybrid.search("Section 6", k=3)
        texts = [r["text"] for r in results]
        assert any("Section 6" in t for t in texts)

    def test_semantic_meaning_found(self):
        """Hybrid should still find semantic matches."""
        results = self.hybrid.search("Can I keep a dog?", k=3)
        texts = [r["text"].lower() for r in results]
        assert any("pets" in t for t in texts)

    def test_source_filter_works(self):
        results = self.hybrid.search("rent", k=5, source_filters=["lease.pdf"])
        for r in results:
            assert r["metadata"]["source"] == "lease.pdf"

    def test_empty_store(self):
        self.store.delete_collection()
        empty_hybrid = HybridSearch(self.store)
        results = empty_hybrid.search("anything", k=5)
        assert results == []
