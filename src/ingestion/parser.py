"""
Document Ingestion Module.

Handles parsing of uploaded documents and extracting clean text
content with metadata (page numbers, source filename).

Supported formats:
    - PDF  (.pdf)          → via pypdf
    - DOCX (.docx)         → via python-docx
    - TXT  (.txt)          → direct read
    - Markdown (.md)       → direct read (preserves structure)
    - JSON (.json)         → pretty-printed key-value extraction
    - YAML (.yaml, .yml)   → pretty-printed key-value extraction
    - XML  (.xml)          → text content extraction
    - HTML (.html, .htm)   → text extraction via BeautifulSoup

Usage:
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    documents = parser.parse("contract.pdf")
"""

import json
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


class DocumentParser:
    """Parse documents and extract text with metadata."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".txt", ".md",
        ".json", ".yaml", ".yml",
        ".xml", ".html", ".htm",
    }

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
                f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if extension == ".pdf":
            return self._parse_pdf(path)
        elif extension == ".docx":
            return self._parse_docx(path)
        elif extension in {".txt", ".md"}:
            return self._parse_txt(path)
        elif extension == ".json":
            return self._parse_json(path)
        elif extension in {".yaml", ".yml"}:
            return self._parse_yaml(path)
        elif extension == ".xml":
            return self._parse_xml(path)
        elif extension in {".html", ".htm"}:
            return self._parse_html(path)

    def _parse_pdf(self, path: Path) -> list[dict]:
        """Extract text from each page of a PDF."""
        reader = PdfReader(str(path))
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                        "total_pages": len(reader.pages),
                        "file_type": "pdf",
                    },
                })

        if not documents:
            raise ValueError(f"No text could be extracted from '{path.name}'.")

        return documents

    def _parse_docx(self, path: Path) -> list[dict]:
        """Extract text from a DOCX file."""
        doc = DocxDocument(str(path))
        full_text = []

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        if not full_text:
            raise ValueError(f"No text could be extracted from '{path.name}'.")

        return [{
            "text": "\n\n".join(full_text),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "docx",
            },
        }]

    def _parse_txt(self, path: Path) -> list[dict]:
        """Read a plain text or markdown file."""
        # Try utf-8 first, fall back to latin-1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        if not text.strip():
            raise ValueError(f"File '{path.name}' is empty.")

        return [{
            "text": text.strip(),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": path.suffix.lstrip("."),
            },
        }]

    def _parse_json(self, path: Path) -> list[dict]:
        """Parse JSON file into readable text."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{path.name}': {str(e)}")

        # Convert to readable text
        readable = self._json_to_text(data)

        if not readable.strip():
            raise ValueError(f"No content could be extracted from '{path.name}'.")

        return [{
            "text": readable.strip(),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "json",
            },
        }]

    def _json_to_text(self, data, indent: int = 0) -> str:
        """Recursively convert JSON to readable text."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}Item {i + 1}:")
                    lines.append(self._json_to_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)

    def _parse_yaml(self, path: Path) -> list[dict]:
        """Parse YAML file into readable text."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for .yaml/.yml files. "
                "Install it: pip install pyyaml"
            )

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in '{path.name}': {str(e)}")

        if data is None:
            raise ValueError(f"File '{path.name}' is empty.")

        # Convert to readable text (reuse JSON converter — works for dicts/lists)
        readable = self._json_to_text(data)

        if not readable.strip():
            raise ValueError(f"No content could be extracted from '{path.name}'.")

        return [{
            "text": readable.strip(),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "yaml",
            },
        }]

    def _parse_xml(self, path: Path) -> list[dict]:
        """Extract text content from XML file."""
        import xml.etree.ElementTree as ET

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in '{path.name}': {str(e)}")

        # Extract all text content from XML elements
        texts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                texts.append(f"{tag}: {elem.text.strip()}")
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())

        if not texts:
            raise ValueError(f"No text content found in '{path.name}'.")

        return [{
            "text": "\n".join(texts),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "xml",
            },
        }]

    def _parse_html(self, path: Path) -> list[dict]:
        """Extract text from HTML file using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required for .html/.htm files. "
                "Install it: pip install beautifulsoup4"
            )

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract text
        clean_text = soup.get_text(separator="\n", strip=True)

        if not clean_text.strip():
            raise ValueError(f"No text content found in '{path.name}'.")

        return [{
            "text": clean_text.strip(),
            "metadata": {
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "html",
            },
        }]
