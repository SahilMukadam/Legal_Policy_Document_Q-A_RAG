"""
Document Ingestion Module.

Handles parsing of uploaded documents (PDF, DOCX, TXT) and extracting
clean text content with metadata (page numbers, source filename).

Supported formats:
    - PDF  (.pdf)  → via PyPDF2
    - DOCX (.docx) → via python-docx
    - TXT  (.txt)  → direct read

Usage:
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    documents = parser.parse("contract.pdf")
    # Returns: [{"text": "...", "metadata": {"source": "contract.pdf", "page": 1}}]
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument


class DocumentParser:
    """Parse documents and extract text with metadata."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def parse(self, file_path: str) -> list[dict]:
        """
        Parse a document and return a list of page/section dicts.

        Args:
            file_path: Path to the document file.

        Returns:
            List of dicts, each with:
                - "text": The extracted text content
                - "metadata": {"source": filename, "page": page_number}
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{extension}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if extension == ".pdf":
            return self._parse_pdf(path)
        elif extension == ".docx":
            return self._parse_docx(path)
        elif extension == ".txt":
            return self._parse_txt(path)

    def _parse_pdf(self, path: Path) -> list[dict]:
        """Extract text from each page of a PDF."""
        reader = PdfReader(str(path))
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():  # Skip empty pages
                documents.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                        "total_pages": len(reader.pages),
                    },
                })

        if not documents:
            raise ValueError(f"No text could be extracted from '{path.name}'.")

        return documents

    def _parse_docx(self, path: Path) -> list[dict]:
        """Extract text from a DOCX file, treating each paragraph as content."""
        doc = DocxDocument(str(path))
        full_text = []

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        if not full_text:
            raise ValueError(f"No text could be extracted from '{path.name}'.")

        # DOCX doesn't have page numbers, so we return as a single document
        return [{
            "text": "\n\n".join(full_text),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
            },
        }]

    def _parse_txt(self, path: Path) -> list[dict]:
        """Read a plain text file."""
        text = path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"File '{path.name}' is empty.")

        return [{
            "text": text.strip(),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
            },
        }]
