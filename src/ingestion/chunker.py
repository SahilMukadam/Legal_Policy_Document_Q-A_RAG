"""
Text Chunking Module with Section Detection.

Splits parsed documents into smaller chunks suitable for embedding
and retrieval. Detects section headers and attaches them to each
chunk so citations can show WHERE in the document the answer came from.

Example output metadata:
    {
        "source": "lease.pdf",
        "page": 1,
        "chunk_index": 2,
        "section": "Section 3: Rent",
        "context_before": "...previous text ending...",
        "context_after": "...next text beginning..."
    }
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.settings import settings


# Patterns that indicate a section/heading in legal documents
SECTION_PATTERNS = [
    # "Section 1:", "Section 1.", "SECTION 1:"
    re.compile(r'^(Section\s+\d+[\.\:].*)$', re.IGNORECASE | re.MULTILINE),
    # "Article 1:", "ARTICLE I."
    re.compile(r'^(Article\s+[\dIVXLCDM]+[\.\:].*)$', re.IGNORECASE | re.MULTILINE),
    # "1. Title", "1.1 Title", "1.1.1 Title"
    re.compile(r'^(\d+(?:\.\d+)*[\.\)]\s+[A-Z].{2,80})$', re.MULTILINE),
    # "DEFINITIONS", "TERMINATION", "PRIVACY POLICY" (all-caps lines)
    re.compile(r'^([A-Z][A-Z\s\&\-]{4,80})$', re.MULTILINE),
    # Markdown headings: "# Title", "## Subtitle"
    re.compile(r'^(#{1,4}\s+.{2,80})$', re.MULTILINE),
    # "Part I:", "Part 1:", "PART ONE"
    re.compile(r'^(Part\s+[\dIVXLCDM]+[\.\:]?.*)$', re.IGNORECASE | re.MULTILINE),
    # "Schedule A", "Appendix 1", "Annex B"
    re.compile(r'^((?:Schedule|Appendix|Annex|Exhibit)\s+[\dA-Z]+[\.\:]?.*)$', re.IGNORECASE | re.MULTILINE),
    # "(a)", "(b)", "(i)", "(ii)" — clause-level markers
    re.compile(r'^(\([a-z]+\)\s+.{5,80})$', re.MULTILINE),
]


class TextChunker:
    """Split documents into overlapping chunks with section detection."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _detect_sections(self, text: str) -> list[dict]:
        """
        Find all section headers in the text with their positions.

        Returns:
            List of {"title": "Section 3: Rent", "start": 450, "end": 470}
        """
        sections = []
        seen_titles = set()

        for pattern in SECTION_PATTERNS:
            for match in pattern.finditer(text):
                title = match.group(1).strip()

                # Skip very short matches or duplicates
                if len(title) < 3 or title in seen_titles:
                    continue

                # Skip if it looks like a sentence (has lowercase after first word)
                words = title.split()
                if len(words) > 2 and all(w[0].islower() for w in words[1:] if w[0].isalpha()):
                    continue

                seen_titles.add(title)
                sections.append({
                    "title": title,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Sort by position
        sections.sort(key=lambda s: s["start"])
        return sections

    def _find_section_for_position(self, position: int, sections: list[dict]) -> str:
        """Find the nearest preceding section header for a text position."""
        current_section = "Introduction"

        for section in sections:
            if section["start"] <= position:
                current_section = section["title"]
            else:
                break

        return current_section

    def _get_context_snippet(self, full_text: str, chunk_text: str, before: int = 50, after: int = 50) -> dict:
        """Get a brief text snippet before and after the chunk for context."""
        pos = full_text.find(chunk_text)
        if pos == -1:
            return {"context_before": "", "context_after": ""}

        # Get text before chunk
        start = max(0, pos - before)
        before_text = full_text[start:pos].strip()
        if start > 0:
            before_text = "..." + before_text

        # Get text after chunk
        end = pos + len(chunk_text)
        after_end = min(len(full_text), end + after)
        after_text = full_text[end:after_end].strip()
        if after_end < len(full_text):
            after_text = after_text + "..."

        return {
            "context_before": before_text,
            "context_after": after_text,
        }

    def chunk(self, documents: list[dict]) -> list[dict]:
        """
        Split documents into chunks with section metadata.

        Args:
            documents: List from DocumentParser.parse().

        Returns:
            List of chunk dicts with enriched metadata including section info.
        """
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Detect sections in the full document
            sections = self._detect_sections(text)

            # Split the text into chunks
            text_chunks = self.splitter.split_text(text)

            for i, chunk_text in enumerate(text_chunks):
                # Find which section this chunk belongs to
                chunk_position = text.find(chunk_text)
                section = self._find_section_for_position(
                    chunk_position if chunk_position >= 0 else 0,
                    sections,
                )

                # Get surrounding context
                context = self._get_context_snippet(text, chunk_text)

                # Estimate line numbers
                if chunk_position >= 0:
                    line_start = text[:chunk_position].count("\n") + 1
                    line_end = line_start + chunk_text.count("\n")
                else:
                    line_start = None
                    line_end = None

                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "section": section,
                        "line_start": line_start,
                        "line_end": line_end,
                        "context_before": context["context_before"],
                        "context_after": context["context_after"],
                    },
                })

        return all_chunks
