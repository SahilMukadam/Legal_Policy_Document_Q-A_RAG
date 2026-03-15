"""
Pinecone Vector Store Module.

Cloud-based vector storage using Pinecone. Same interface as the
ChromaDB VectorStore so they're interchangeable.

Pinecone advantages over ChromaDB:
    - Cloud-hosted (no local storage needed)
    - Scales to millions of vectors
    - Production-grade reliability
    - Built-in metadata filtering

Usage:
    from src.retrieval.pinecone_store import PineconeVectorStore

    store = PineconeVectorStore()
    store.add_chunks(chunks, doc_id="lease.pdf", collection_name="Real Estate")
    results = store.search("When is rent due?", k=5)
"""

from pinecone import Pinecone, ServerlessSpec

from configs.settings import settings
from src.embeddings.embedding_service import EmbeddingService


class PineconeVectorStore:
    """Pinecone-backed vector store — same interface as ChromaDB VectorStore."""

    def __init__(self):
        self.embedding_service = EmbeddingService()

        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY not set in .env. "
                "Get your free key at: https://app.pinecone.io/"
            )

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self.dimension = self.embedding_service.dimension

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        self.index = self.pc.Index(self.index_name)

    def add_chunks(
        self,
        chunks: list[dict],
        doc_id: str | None = None,
        collection_name: str = "General",
    ) -> int:
        """Add document chunks to Pinecone."""
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        vectors = []
        for i, chunk in enumerate(chunks):
            prefix = doc_id or chunk["metadata"].get("source", "doc")
            chunk_id = f"{prefix}_chunk_{i}"

            # Build metadata (Pinecone supports string, number, boolean, list of strings)
            metadata = {"text": chunk["text"], "collection": collection_name}
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)

            vectors.append({
                "id": chunk_id,
                "values": embeddings[i],
                "metadata": metadata,
            })

        # Pinecone upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        return len(vectors)

    def search(
        self,
        query: str,
        k: int | None = None,
        source_filters: list[str] | None = None,
    ) -> list[dict]:
        """Search Pinecone for relevant chunks."""
        k = k or settings.top_k

        query_embedding = self.embedding_service.embed_single(query)

        # Build metadata filter
        filter_dict = None
        if source_filters and len(source_filters) == 1:
            filter_dict = {"source": {"$eq": source_filters[0]}}
        elif source_filters and len(source_filters) > 1:
            filter_dict = {"source": {"$in": source_filters}}

        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter_dict,
        )

        formatted = []
        for match in results.matches:
            metadata = dict(match.metadata)
            text = metadata.pop("text", "")
            formatted.append({
                "text": text,
                "metadata": metadata,
                "score": match.score,
            })

        return formatted

    def list_collections(self) -> dict[str, list[dict]]:
        """List all collections with their documents."""
        # Pinecone doesn't have a "list all" — we query with a large limit
        # For production, you'd maintain a separate metadata store
        # This is a workaround using Pinecone's list endpoint
        try:
            # Fetch a sample to discover collections
            # Note: this is limited — for large datasets, maintain a separate index
            sample = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=10000,
                include_metadata=True,
            )

            tree: dict[str, dict[str, dict]] = {}
            for match in sample.matches:
                coll = match.metadata.get("collection", "General")
                source = match.metadata.get("source", "Unknown")

                if coll not in tree:
                    tree[coll] = {}
                if source not in tree[coll]:
                    tree[coll][source] = {"source": source, "chunks": 0, "pages": set()}

                tree[coll][source]["chunks"] += 1
                page = match.metadata.get("page")
                if page is not None:
                    tree[coll][source]["pages"].add(page)

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
        except Exception:
            return {}

    def list_documents(self) -> list[dict]:
        """List all documents as a flat list."""
        collections = self.list_collections()
        documents = []
        for coll_name, docs in collections.items():
            for doc in docs:
                documents.append({**doc, "collection": coll_name})
        return sorted(documents, key=lambda d: d["source"])

    def get_sources_for_collections(self, collection_names: list[str]) -> list[str]:
        """Get all source filenames for given collections."""
        collections = self.list_collections()
        sources = []
        for coll_name in collection_names:
            if coll_name in collections:
                sources.extend([doc["source"] for doc in collections[coll_name]])
        return sources

    def delete_document(self, source: str) -> int:
        """Delete all chunks for a specific document."""
        try:
            # Find IDs to delete by querying with filter
            results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=10000,
                include_metadata=True,
                filter={"source": {"$eq": source}},
            )

            ids_to_delete = [match.id for match in results.matches]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def delete_collection_group(self, collection_name: str) -> int:
        """Delete all chunks in a collection."""
        try:
            results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=10000,
                include_metadata=True,
                filter={"collection": {"$eq": collection_name}},
            )

            ids_to_delete = [match.id for match in results.matches]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def document_exists(self, source: str) -> bool:
        """Check if a document exists."""
        try:
            results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=1,
                include_metadata=True,
                filter={"source": {"$eq": source}},
            )
            return len(results.matches) > 0
        except Exception:
            return False

    def get_document_collection(self, source: str) -> str | None:
        """Get the collection name for a document."""
        try:
            results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=1,
                include_metadata=True,
                filter={"source": {"$eq": source}},
            )
            if results.matches:
                return results.matches[0].metadata.get("collection")
        except Exception:
            pass
        return None

    def get_stats(self) -> dict:
        """Return index statistics."""
        collections = self.list_collections()
        total_docs = sum(len(docs) for docs in collections.values())
        index_stats = self.index.describe_index_stats()

        return {
            "total_chunks": index_stats.total_vector_count,
            "total_documents": total_docs,
            "total_collections": len(collections),
            "collections": {
                name: {"documents": len(docs), "chunks": sum(d["chunks"] for d in docs)}
                for name, docs in collections.items()
            },
            "index_name": self.index_name,
            "provider": "pinecone",
            "embedding_model": settings.embedding_model,
            "embedding_dimension": self.dimension,
        }

    def delete_collection(self):
        """Delete all vectors in the index."""
        try:
            self.index.delete(delete_all=True)
        except Exception:
            pass
