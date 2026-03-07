"""
Vector Store Module.

Manages document storage and retrieval using ChromaDB.
ChromaDB stores embeddings locally (in a folder) and lets you search
by semantic similarity — find chunks whose *meaning* is closest to your query.

How retrieval works:
    1. User asks: "When is rent due?"
    2. We embed the question → [0.023, -0.156, ...]
    3. ChromaDB finds the stored chunks with the most similar vectors
    4. We return those chunks (with their original text + metadata)

Usage:
    from src.retrieval.vector_store import VectorStore

    store = VectorStore()
    store.add_chunks(chunks)                          # Store document chunks
    results = store.search("When is rent due?", k=5)  # Find relevant chunks
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from configs.settings import settings
from src.embeddings.embedding_service import EmbeddingService


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(self):
        """Initialize ChromaDB client and embedding service."""
        self.embedding_service = EmbeddingService()

        # Persistent storage — data survives server restarts
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection (like a "table" in a database)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add_chunks(self, chunks: list[dict], doc_id: str | None = None) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dicts from TextChunker, each with
                    "text" and "metadata" keys.
            doc_id: Optional document identifier prefix for chunk IDs.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Extract texts and generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            prefix = doc_id or chunk["metadata"].get("source", "doc")
            chunk_id = f"{prefix}_chunk_{i}"
            ids.append(chunk_id)

            # ChromaDB metadata must be str, int, float, or bool
            metadata = {}
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)

        # Add to ChromaDB (handles batching internally)
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """
        Search for chunks most relevant to the query.

        Args:
            query: Natural language question or search text.
            k: Number of results to return (default from settings).

        Returns:
            List of result dicts, each with:
                - "text": The chunk text
                - "metadata": Source info (filename, page, chunk_index)
                - "score": Similarity score (0 to 1, higher = more relevant)
        """
        k = k or settings.top_k

        # Embed the query
        query_embedding = self.embedding_service.embed_single(query)

        # Search ChromaDB
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
        )

        # Format results
        formatted = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                })

        return formatted

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": settings.chroma_collection_name,
            "persist_dir": settings.chroma_persist_dir,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": self.embedding_service.dimension,
        }

    def delete_collection(self):
        """Delete all data in the collection. Use with caution."""
        self.client.delete_collection(settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
