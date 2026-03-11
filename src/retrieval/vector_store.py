"""
Vector Store Module.

Manages document storage and retrieval using ChromaDB.
Supports:
    - Collections (folders) for organizing documents
    - Multi-source filtering (search across selected files/collections)
    - Document and collection management

Usage:
    store = VectorStore()
    store.add_chunks(chunks, doc_id="lease.pdf", collection="Real Estate")
    results = store.search("rent", source_filters=["lease.pdf", "addendum.pdf"])
    collections = store.list_collections()
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from configs.settings import settings
from src.embeddings.embedding_service import EmbeddingService


class VectorStore:
    """ChromaDB-backed vector store with collections and multi-filter search."""

    def __init__(self):
        self.embedding_service = EmbeddingService()

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict],
        doc_id: str | None = None,
        collection_name: str = "General",
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dicts from TextChunker.
            doc_id: Document identifier prefix for chunk IDs.
            collection_name: Collection/folder this document belongs to.

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

            # Add collection metadata
            metadata["collection"] = collection_name
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
        source_filters: list[str] | None = None,
    ) -> list[dict]:
        """
        Search for chunks most relevant to the query.

        Args:
            query: Natural language question or search text.
            k: Number of results to return.
            source_filters: Optional list of filenames to restrict search.
                           Pass specific filenames to filter.

        Returns:
            List of result dicts with text, metadata, and similarity score.
        """
        k = k or settings.top_k

        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query)

        # Build metadata filter
        where_filter = None
        if source_filters and len(source_filters) == 1:
            where_filter = {"source": source_filters[0]}
        elif source_filters and len(source_filters) > 1:
            where_filter = {"source": {"$in": source_filters}}

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

    def list_collections(self) -> dict[str, list[dict]]:
        """
        List all collections with their documents.

        Returns:
            Dict mapping collection names to lists of document info:
            {
                "Real Estate": [
                    {"source": "lease.pdf", "chunks": 5, "pages": 3},
                    {"source": "addendum.pdf", "chunks": 2, "pages": 1},
                ],
                "Employment": [...]
            }
        """
        if self.collection.count() == 0:
            return {}

        all_data = self.collection.get(include=["metadatas"])

        # Build nested structure: collection → source → stats
        tree: dict[str, dict[str, dict]] = {}
        for metadata in all_data["metadatas"]:
            coll = metadata.get("collection", "General")
            source = metadata.get("source", "Unknown")

            if coll not in tree:
                tree[coll] = {}
            if source not in tree[coll]:
                tree[coll][source] = {"source": source, "chunks": 0, "pages": set()}

            tree[coll][source]["chunks"] += 1
            page = metadata.get("page")
            if page is not None:
                tree[coll][source]["pages"].add(page)

        # Convert sets to counts
        result = {}
        for coll_name, docs in sorted(tree.items()):
            result[coll_name] = []
            for doc_info in sorted(docs.values(), key=lambda d: d["source"]):
                result[coll_name].append({
                    "source": doc_info["source"],
                    "chunks": doc_info["chunks"],
                    "pages": len(doc_info["pages"]),
                })

        return result

    def list_documents(self) -> list[dict]:
        """List all documents as a flat list (backward compatible)."""
        collections = self.list_collections()
        documents = []
        for coll_name, docs in collections.items():
            for doc in docs:
                documents.append({
                    **doc,
                    "collection": coll_name,
                })
        return sorted(documents, key=lambda d: d["source"])

    def get_sources_for_collections(self, collection_names: list[str]) -> list[str]:
        """
        Get all source filenames belonging to the given collections.

        Args:
            collection_names: List of collection names.

        Returns:
            List of source filenames.
        """
        collections = self.list_collections()
        sources = []
        for coll_name in collection_names:
            if coll_name in collections:
                sources.extend([doc["source"] for doc in collections[coll_name]])
        return sources

    def delete_document(self, source: str) -> int:
        """Delete all chunks belonging to a specific document."""
        if self.collection.count() == 0:
            return 0

        all_data = self.collection.get(include=["metadatas"])

        ids_to_delete = []
        for chunk_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
            if metadata.get("source") == source:
                ids_to_delete.append(chunk_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def delete_collection_group(self, collection_name: str) -> int:
        """Delete all chunks belonging to a collection/folder."""
        if self.collection.count() == 0:
            return 0

        all_data = self.collection.get(include=["metadatas"])

        ids_to_delete = []
        for chunk_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
            if metadata.get("collection") == collection_name:
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

    def get_document_collection(self, source: str) -> str | None:
        """Get the collection name for a document."""
        if self.collection.count() == 0:
            return None

        results = self.collection.get(
            where={"source": source},
            limit=1,
            include=["metadatas"],
        )

        if results["ids"] and results["metadatas"]:
            return results["metadatas"][0].get("collection")
        return None

    def get_stats(self) -> dict:
        """Return collection statistics."""
        collections = self.list_collections()
        total_docs = sum(len(docs) for docs in collections.values())
        return {
            "total_chunks": self.collection.count(),
            "total_documents": total_docs,
            "total_collections": len(collections),
            "collections": {
                name: {"documents": len(docs), "chunks": sum(d["chunks"] for d in docs)}
                for name, docs in collections.items()
            },
            "collection_name": settings.chroma_collection_name,
            "persist_dir": settings.chroma_persist_dir,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": self.embedding_service.dimension,
        }

    def delete_collection(self):
        """Delete all data. Use with caution."""
        self.client.delete_collection(settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
