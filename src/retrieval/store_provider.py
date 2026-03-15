"""
Vector Store Provider Factory.

Returns the configured vector store based on VECTOR_STORE_PROVIDER in .env.
Same pattern as llm_provider.py — swap providers by changing one config line.

Usage:
    from src.retrieval.store_provider import get_vector_store

    store = get_vector_store()  # Returns ChromaDB or Pinecone based on .env

To switch:
    # In .env, change:
    VECTOR_STORE_PROVIDER=pinecone
    PINECONE_API_KEY=your-key-here
"""

from configs.settings import settings


def get_vector_store():
    """
    Returns a vector store instance based on the configured provider.

    Supported providers:
        - "chroma"   → ChromaDB (local, free, great for development)
        - "pinecone" → Pinecone (cloud, scalable, production-grade)

    Both providers implement the same interface:
        - add_chunks()
        - search()
        - list_collections()
        - list_documents()
        - delete_document()
        - delete_collection_group()
        - document_exists()
        - get_stats()
        - delete_collection()
    """
    provider = settings.vector_store_provider.lower()

    if provider == "chroma":
        from src.retrieval.vector_store import VectorStore
        return VectorStore()

    elif provider == "pinecone":
        from src.retrieval.pinecone_store import PineconeVectorStore

        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY not set in .env. "
                "Get your free key at: https://app.pinecone.io/"
            )

        return PineconeVectorStore()

    else:
        raise ValueError(
            f"Unknown VECTOR_STORE_PROVIDER: '{provider}'. "
            f"Supported: 'chroma', 'pinecone'"
        )
