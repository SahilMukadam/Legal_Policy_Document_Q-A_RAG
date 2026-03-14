"""
Hybrid Search Module.

Combines semantic search (ChromaDB vectors) with keyword search (BM25)
for better retrieval. Uses Reciprocal Rank Fusion to merge results.

Why hybrid search?
    - Semantic: "What's the penalty for late payment?" finds "5% fee" (meaning match)
    - Keyword: "Section 6" finds exact section (exact match)
    - Hybrid: Gets both — best of both worlds

How Reciprocal Rank Fusion works:
    Each result gets a score based on its RANK in each search method:
        RRF_score = 1/(k + rank_semantic) + 1/(k + rank_keyword)
    Results that rank high in BOTH methods get the highest combined score.

Usage:
    from src.retrieval.hybrid_search import HybridSearch

    searcher = HybridSearch(vector_store)
    results = searcher.search("When is rent due?", k=5)
"""

import re
import math
from collections import defaultdict

from src.retrieval.vector_store import VectorStore


class BM25:
    """
    Simple BM25 keyword search implementation.

    BM25 is the classic text retrieval algorithm used by search engines.
    It scores documents based on term frequency and document length.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization (0=no normalization, 1=full).
        """
        self.k1 = k1
        self.b = b
        self.documents: list[dict] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0
        self.doc_freqs: dict[str, int] = defaultdict(int)  # How many docs contain each term
        self.total_docs: int = 0
        self._tokenized_docs: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def index(self, documents: list[dict]):
        """
        Build the BM25 index from document chunks.

        Args:
            documents: List of dicts with "text" and "metadata" keys.
        """
        self.documents = documents
        self.total_docs = len(documents)
        self._tokenized_docs = []
        self.doc_freqs = defaultdict(int)

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self._tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count unique terms per document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search documents using BM25 scoring.

        Returns:
            List of result dicts with text, metadata, and BM25 score.
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._tokenized_docs):
            score = self._score_document(query_tokens, doc_tokens, i)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            if score > 0:
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": score,
                })

        return results

    def _score_document(self, query_tokens: list[str], doc_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a single document."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # Term frequency in this document
            tf = doc_tokens.count(term)
            if tf == 0:
                continue

            # Inverse document frequency
            df = self.doc_freqs[term]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score


class HybridSearch:
    """
    Hybrid search combining semantic (vector) and keyword (BM25) search.
    Uses Reciprocal Rank Fusion to merge results.
    """

    def __init__(self, vector_store: VectorStore, rrf_k: int = 60):
        """
        Args:
            vector_store: The ChromaDB vector store for semantic search.
            rrf_k: RRF constant (default 60, standard in literature).
        """
        self.vector_store = vector_store
        self.bm25 = BM25()
        self.rrf_k = rrf_k
        self._indexed = False

    def _ensure_bm25_index(self):
        """Build BM25 index from all documents in vector store."""
        if self.vector_store.collection.count() == 0:
            self._indexed = False
            return

        # Get all documents from ChromaDB
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        documents = []
        for text, metadata in zip(all_data["documents"], all_data["metadatas"]):
            documents.append({"text": text, "metadata": metadata})

        self.bm25.index(documents)
        self._indexed = True

    def search(
        self,
        query: str,
        k: int = 5,
        source_filters: list[str] | None = None,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid search combining semantic and keyword results.

        Args:
            query: Search query.
            k: Number of results to return.
            source_filters: Optional list of filenames to filter.
            semantic_weight: Weight for semantic search scores (0-1).
            keyword_weight: Weight for keyword search scores (0-1).

        Returns:
            List of result dicts with text, metadata, and fused score.
        """
        # Rebuild BM25 index if needed
        self._ensure_bm25_index()

        if not self._indexed:
            return []

        # Get more results than k from each method for better fusion
        fetch_k = min(k * 3, self.vector_store.collection.count())

        # Semantic search
        semantic_results = self.vector_store.search(
            query=query,
            k=fetch_k,
            source_filters=source_filters,
        )

        # Keyword search
        keyword_results = self.bm25.search(query=query, k=fetch_k)

        # Apply source filter to BM25 results (vector store handles its own)
        if source_filters:
            keyword_results = [
                r for r in keyword_results
                if r["metadata"].get("source") in source_filters
            ]

        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight,
        )

        return fused[:k]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = w1 * 1/(k + rank_semantic) + w2 * 1/(k + rank_keyword)
        """
        # Build a map: text -> result data
        result_map: dict[str, dict] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        # Score semantic results by rank
        for rank, result in enumerate(semantic_results, start=1):
            key = result["text"][:200]  # Use text prefix as unique key
            result_map[key] = result
            rrf_scores[key] += semantic_weight * (1.0 / (self.rrf_k + rank))

        # Score keyword results by rank
        for rank, result in enumerate(keyword_results, start=1):
            key = result["text"][:200]
            if key not in result_map:
                result_map[key] = result
            rrf_scores[key] += keyword_weight * (1.0 / (self.rrf_k + rank))

        # Sort by fused RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # Build final results
        fused_results = []
        for key in sorted_keys:
            result = result_map[key].copy()
            result["score"] = round(rrf_scores[key], 4)
            result["search_type"] = "hybrid"
            fused_results.append(result)

        return fused_results
