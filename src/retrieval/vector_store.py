"""
Vector Store Module.

Manages document storage and retrieval using ChromaDB.
Now supports:
    - Metadata filtering (search within specific documents)
    - Document listing (see all uploaded documents)
    - Document deletion (remove a single document's chunks)
    - Duplicate detection (prevent re-uploading same file)

Usage:
    from src.retrieval.vector_store import VectorStore

    store = VectorStore()
    store.add_chunks(chunks)
    results = store.search("rent due", k=5, source_filter="lease.pdf")
    docs = store.list_documents()
    store.delete_document("old_contract.pdf")
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from configs.settings import settings
from src.embeddings.embedding_service import EmbeddingService


class VectorStore:
    """ChromaDB-backed vector store with metadata filtering."""

    def __init__(self):
        """Initialize ChromaDB client and embedding service."""
        self.embedding_service = EmbeddingService()

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], doc_id: str | None = None) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dicts from TextChunker.
            doc_id: Optional document identifier prefix for chunk IDs.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            prefix = doc_id or chunk["metadata"].get("source", "doc")
            chunk_id = f"{prefix}_chunk_{i}"
            ids.append(chunk_id)

            metadata = {}
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    def search(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for chunks most relevant to the query.

        Args:
            query: Natural language question or search text.
            k: Number of results to return.
            source_filter: Optional filename to restrict search to a
                          specific document (e.g., "lease.pdf").

        Returns:
            List of result dicts with text, metadata, and similarity score.
        """
        k = k or settings.top_k

        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query)

        # Build optional metadata filter
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            where=where_filter,
        )

        formatted = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i],
                })

        return formatted

    def list_documents(self) -> list[dict]:
        """
        List all uploaded documents with their chunk counts.

        Returns:
            List of dicts with source filename and number of chunks.
        """
        if self.collection.count() == 0:
            return []

        # Get all metadata to find unique sources
        all_data = self.collection.get(include=["metadatas"])

        doc_counts: dict[str, dict] = {}
        for metadata in all_data["metadatas"]:
            source = metadata.get("source", "Unknown")
            if source not in doc_counts:
                doc_counts[source] = {
                    "source": source,
                    "chunks": 0,
                    "pages": set(),
                }
            doc_counts[source]["chunks"] += 1
            page = metadata.get("page")
            if page is not None:
                doc_counts[source]["pages"].add(page)

        # Convert sets to counts for JSON serialization
        documents = []
        for doc_info in doc_counts.values():
            documents.append({
                "source": doc_info["source"],
                "chunks": doc_info["chunks"],
                "pages": len(doc_info["pages"]),
            })

        return sorted(documents, key=lambda d: d["source"])

    def delete_document(self, source: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        Args:
            source: The filename of the document to delete.

        Returns:
            Number of chunks deleted.
        """
        if self.collection.count() == 0:
            return 0

        # Find all chunk IDs for this document
        all_data = self.collection.get(include=["metadatas"])

        ids_to_delete = []
        for chunk_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
            if metadata.get("source") == source:
                ids_to_delete.append(chunk_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def document_exists(self, source: str) -> bool:
        """Check if a document has already been uploaded."""
        if self.collection.count() == 0:
            return False

        results = self.collection.get(
            where={"source": source},
            limit=1,
            include=[],
        )

        return len(results["ids"]) > 0

    def get_stats(self) -> dict:
        """Return collection statistics."""
        documents = self.list_documents()
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(documents),
            "documents": documents,
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
