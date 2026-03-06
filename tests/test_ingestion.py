"""
Tests for document parsing and chunking.
Run with: pytest tests/ -v
"""

import os
from pathlib import Path

import pytest

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker


# ---- Setup: create sample test files ----

SAMPLE_DIR = Path("data/sample_docs")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_txt_file():
    """Create a sample .txt file for testing."""
    content = (
        "TERMS OF SERVICE\n\n"
        "Section 1: Acceptance of Terms\n"
        "By accessing this service, you agree to be bound by these terms. "
        "If you do not agree, do not use the service. "
        "These terms apply to all users.\n\n"
        "Section 2: Privacy Policy\n"
        "We collect personal data as described in our Privacy Policy. "
        "Your data will not be sold to third parties. "
        "We use industry-standard encryption to protect your information.\n\n"
        "Section 3: Limitation of Liability\n"
        "In no event shall the company be liable for any indirect, "
        "incidental, or consequential damages arising from your use of "
        "the service. The total liability shall not exceed the amount "
        "paid by you in the twelve months prior to the claim."
    )
    file_path = SAMPLE_DIR / "test_terms.txt"
    file_path.write_text(content, encoding="utf-8")
    yield str(file_path)
    # Cleanup
    file_path.unlink(missing_ok=True)


# ---- Parser Tests ----

class TestDocumentParser:

    def test_parse_txt(self, sample_txt_file):
        parser = DocumentParser()
        result = parser.parse(sample_txt_file)

        assert len(result) == 1
        assert "TERMS OF SERVICE" in result[0]["text"]
        assert result[0]["metadata"]["source"] == "test_terms.txt"
        assert result[0]["metadata"]["page"] == 1

    def test_unsupported_file_type(self):
        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse("document.xlsx")

    def test_supported_extensions(self):
        assert ".pdf" in DocumentParser.SUPPORTED_EXTENSIONS
        assert ".docx" in DocumentParser.SUPPORTED_EXTENSIONS
        assert ".txt" in DocumentParser.SUPPORTED_EXTENSIONS


# ---- Chunker Tests ----

class TestTextChunker:

    def test_basic_chunking(self, sample_txt_file):
        parser = DocumentParser()
        documents = parser.parse(sample_txt_file)

        # Use small chunk size to force multiple chunks
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(documents)

        assert len(chunks) > 1  # Should produce multiple chunks
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_chunk_metadata_preserved(self, sample_txt_file):
        parser = DocumentParser()
        documents = parser.parse(sample_txt_file)

        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(documents)

        # Every chunk should have the source filename
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test_terms.txt"
            assert "chunk_index" in chunk["metadata"]
            assert "total_chunks" in chunk["metadata"]

    def test_small_text_single_chunk(self):
        """Text smaller than chunk_size should produce exactly 1 chunk."""
        chunker = TextChunker(chunk_size=5000, chunk_overlap=200)
        documents = [{"text": "Short text.", "metadata": {"source": "test.txt", "page": 1}}]
        chunks = chunker.chunk(documents)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."
