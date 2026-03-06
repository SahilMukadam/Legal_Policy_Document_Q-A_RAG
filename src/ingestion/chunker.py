"""
Text Chunking Module.

Splits parsed documents into smaller chunks suitable for embedding
and retrieval. Maintains metadata (source, page) through the split.

Why chunking matters:
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't cut sentences mid-thought

Usage:
    from src.ingestion.chunker import TextChunker

    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(documents)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.settings import settings


class TextChunker:
    """Split documents into overlapping chunks for vector storage."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Args:
            chunk_size: Max characters per chunk (default from settings).
            chunk_overlap: Overlap between chunks (default from settings).
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on these
        # separators in order — paragraphs first, then sentences,
        # then words. This keeps chunks semantically coherent.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: list[dict]) -> list[dict]:
        """
        Split documents into chunks, preserving metadata.

        Args:
            documents: List from DocumentParser.parse(), each with
                       "text" and "metadata" keys.

        Returns:
            List of chunk dicts, each with:
                - "text": The chunk text
                - "metadata": Original metadata + chunk_index
        """
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Split the text into chunks
            text_chunks = self.splitter.split_text(text)

            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                })

        return all_chunks
